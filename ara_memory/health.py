from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ara_memory.backup import verify_backup
from ara_memory.models import MemoryStatus


@dataclass(slots=True)
class HealthSignal:
    name: str
    passed: bool
    detail: str
    severity: str = "error"
    value: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
            "value": self.value,
        }


@dataclass(slots=True)
class HealthReport:
    scope: str
    passed: bool
    status: str
    score: int
    stats: dict[str, Any]
    signals: list[HealthSignal]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "status": self.status,
            "score": self.score,
            "stats": self.stats,
            "signals": [signal.as_dict() for signal in self.signals],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [f"# Ara Memory Health: {self.scope}", f"status: {self.status}", f"score: {self.score}/100"]
        lines.append("## Signals")
        for signal in self.signals:
            marker = "OK" if signal.passed else signal.severity.upper()
            lines.append(f"- [{marker}] {signal.name}: {signal.detail}")
        lines.append("## Recommendations")
        for recommendation in self.recommendations:
            lines.append(f"- {recommendation}")
        return "\n".join(lines)


def run_health_check(
    memory: Any,
    *,
    scope: str = "global",
    query: str = "current memory health",
    recall_budget: int = 1600,
    hot_budget: int = 1200,
    review_limit: int = 500,
    backup_max_age_hours: float = 72.0,
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
) -> HealthReport:
    stats = memory.stats()
    spool = memory.spool_stats()
    doctor = memory.doctor(scope=scope, recall_query=query, recall_budget=recall_budget, hot_budget=hot_budget)
    retention = memory.retention(scope=scope, cold_limit=5)
    triage = memory.review_triage(scope=scope, limit=review_limit, examples_per_group=1)
    latest_backup = _latest_backup(memory.store.root)
    latest_retention_cycle = _latest_retention_cycle(memory.store.root, scope=scope)

    signals = [
        HealthSignal(
            name="doctor",
            passed=doctor.passed,
            detail="all doctor checks passed" if doctor.passed else _failed_doctor_detail(doctor.as_dict()),
            value=doctor.as_dict(),
        ),
        _spool_signal(spool),
        _review_pressure_signal(triage.as_dict()),
        _candidate_ratio_signal(retention.as_dict()),
        _cold_ratio_signal(retention.as_dict()),
        _retention_cycle_signal(latest_retention_cycle, retention.as_dict()),
        _backup_signal(latest_backup, max_age_hours=backup_max_age_hours),
    ]
    if regression_cases is not None:
        regression = memory.recall_regression(regression_cases, baseline=regression_baseline)
        signals.append(
            HealthSignal(
                name="recall_regression",
                passed=regression.passed,
                detail="recall regression passed" if regression.passed else "recall regression failed",
                value=regression.as_dict(),
            )
        )

    score = _score(signals)
    hard_fail = any(not signal.passed and signal.severity == "error" for signal in signals)
    passed = not hard_fail
    status = "pass" if passed and score >= 90 else "watch" if passed else "fail"
    return HealthReport(
        scope=scope,
        passed=passed,
        status=status,
        score=score,
        stats={
            **stats,
            "spool": spool,
            "retention": retention.totals,
            "review_open": triage.total_open,
            "latest_backup": latest_backup,
            "latest_retention_cycle": latest_retention_cycle,
        },
        signals=signals,
        recommendations=_recommend(signals),
    )


def _spool_signal(spool: dict[str, int]) -> HealthSignal:
    failed = spool.get("failed", 0)
    pending = spool.get("pending", 0)
    processing = spool.get("processing", 0)
    if failed:
        return HealthSignal("spool", False, f"{failed} failed spool item(s) require replay or rejection", value=spool)
    if pending + processing > 25:
        return HealthSignal("spool", False, f"{pending} pending and {processing} processing items exceed backlog budget", severity="warning", value=spool)
    return HealthSignal("spool", True, f"pending={pending}, processing={processing}, failed={failed}", value=spool)


def _review_pressure_signal(triage: dict[str, Any]) -> HealthSignal:
    total_open = int(triage.get("total_open", 0))
    if total_open > 1000:
        return HealthSignal("review_pressure", False, f"{total_open} open review items exceed hard budget", value=total_open)
    if total_open > 250:
        return HealthSignal("review_pressure", False, f"{total_open} open review items need triage", severity="warning", value=total_open)
    return HealthSignal("review_pressure", True, f"{total_open} open review items", value=total_open)


def _candidate_ratio_signal(retention: dict[str, Any]) -> HealthSignal:
    status_counts = {row["status"]: row["count"] for row in retention.get("by_status", [])}
    stable = max(1, int(status_counts.get(MemoryStatus.STABLE.value, 0)))
    candidate = int(status_counts.get(MemoryStatus.CANDIDATE.value, 0))
    ratio = candidate / stable
    if ratio > 4:
        return HealthSignal("candidate_ratio", False, f"candidate/stable ratio is {ratio:.2f}", value=ratio)
    if ratio > 2:
        return HealthSignal("candidate_ratio", False, f"candidate/stable ratio is {ratio:.2f}", severity="warning", value=ratio)
    return HealthSignal("candidate_ratio", True, f"candidate/stable ratio is {ratio:.2f}", value=ratio)


def _cold_ratio_signal(retention: dict[str, Any]) -> HealthSignal:
    totals = retention.get("totals", {})
    capsules = max(1, int(totals.get("capsules", 0)))
    cold = int(totals.get("cold_capsules", 0))
    ratio = cold / capsules
    if ratio > 0.5:
        return HealthSignal("cold_ratio", False, f"cold capsules are {ratio:.0%} of scope", severity="warning", value=ratio)
    return HealthSignal("cold_ratio", True, f"cold capsules are {ratio:.0%} of scope", value=ratio)


def _backup_signal(latest: dict[str, Any] | None, *, max_age_hours: float) -> HealthSignal:
    if latest is None:
        return HealthSignal("backup", False, "no verified backup found", severity="warning")
    if not latest["verified"]:
        return HealthSignal("backup", False, f"latest backup failed verification: {latest['path']}", value=latest)
    age = float(latest["age_hours"])
    if age > max_age_hours:
        return HealthSignal("backup", False, f"latest verified backup is {age:.1f}h old", severity="warning", value=latest)
    return HealthSignal("backup", True, f"latest verified backup is {age:.1f}h old", value=latest)


def _retention_cycle_signal(latest: dict[str, Any] | None, retention: dict[str, Any]) -> HealthSignal:
    totals = retention.get("totals", {})
    capsules = max(1, int(totals.get("capsules", 0)))
    cold = int(totals.get("cold_capsules", 0))
    ratio = cold / capsules
    if ratio <= 0.5:
        return HealthSignal("retention_cycle", True, "cold pressure is below retention-cycle threshold", value=latest)
    if latest is None:
        return HealthSignal(
            "retention_cycle",
            False,
            "no recent retention-cycle evidence for elevated cold pressure",
            severity="warning",
            value=latest,
        )
    if not latest.get("passed"):
        return HealthSignal(
            "retention_cycle",
            False,
            f"latest retention-cycle is blocked: {latest.get('path')}",
            severity="warning",
            value=latest,
        )
    shadow = latest.get("shadow_prune") or {}
    deletion = shadow.get("deletion") or {}
    if shadow and not shadow.get("passed"):
        return HealthSignal(
            "retention_cycle",
            False,
            f"latest retention-cycle shadow-prune failed: {latest.get('path')}",
            severity="warning",
            value=latest,
        )
    return HealthSignal(
        "retention_cycle",
        True,
        "latest retention-cycle passed; "
        f"shadow capsules_removed={deletion.get('capsules_removed', 0)}, events_removed={deletion.get('events_removed', 0)}",
        value=latest,
    )


def _latest_backup(root: Path) -> dict[str, Any] | None:
    backup_dir = root / "backups"
    if not backup_dir.exists():
        return None
    candidates = sorted(backup_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            verification = verify_backup(path)
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError, KeyError):
            continue
        created_at = _backup_created_at(path, verification)
        age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
        return {
            "path": str(path),
            "verified": bool(verification.get("passed")),
            "age_hours": round(age_hours, 2),
            "created_at": created_at.isoformat(),
            "entries": verification.get("entries", 0),
        }
    return None


def _latest_retention_cycle(root: Path, *, scope: str) -> dict[str, Any] | None:
    cycle_dir = root / "archive" / "retention-cycles"
    if not cycle_dir.exists():
        return None
    candidates = sorted(cycle_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("scope") not in (scope, None):
            continue
        created_at = _cycle_created_at(path, payload)
        return _summarize_retention_cycle(path, payload, created_at)
    return None


def _summarize_retention_cycle(path: Path, payload: dict[str, Any], created_at: datetime) -> dict[str, Any]:
    plan = payload.get("prune_plan") or {}
    shadow = payload.get("shadow_prune") or {}
    backup = payload.get("backup") or {}
    cold_export = payload.get("cold_export") or {}
    return {
        "path": str(path),
        "report_path": payload.get("report_path") or str(path),
        "scope": payload.get("scope"),
        "passed": bool(payload.get("passed")),
        "created_at": created_at.isoformat(),
        "age_hours": round((datetime.now(timezone.utc) - created_at).total_seconds() / 3600, 2),
        "backup_path": backup.get("path"),
        "cold_export_path": cold_export.get("path"),
        "plan_totals": plan.get("totals", {}),
        "shadow_prune": {
            "passed": shadow.get("passed"),
            "deletion": (shadow.get("deletion") or {}),
        }
        if shadow
        else None,
    }


def _cycle_created_at(path: Path, payload: dict[str, Any]) -> datetime:
    raw = payload.get("created_at")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _backup_created_at(path: Path, verification: dict[str, Any]) -> datetime:
    raw = verification.get("manifest", {}).get("created_at")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _failed_doctor_detail(report: dict[str, Any]) -> str:
    failed = [check for check in report.get("checks", []) if not check.get("passed") and check.get("severity") != "warning"]
    if not failed:
        return "doctor warnings present"
    return "; ".join(f"{check['name']}: {check['detail']}" for check in failed[:3])


def _score(signals: list[HealthSignal]) -> int:
    score = 100
    for signal in signals:
        if signal.passed:
            continue
        score -= 25 if signal.severity == "error" else 10
    return max(0, score)


def _recommend(signals: list[HealthSignal]) -> list[str]:
    recommendations = []
    for signal in signals:
        if signal.passed:
            continue
        if signal.name == "doctor":
            recommendations.append("Run doctor with JSON output and fix failing checks before relying on recall.")
        elif signal.name == "spool":
            recommendations.append("Inspect .ara-memory/spool/failed and drain or explicitly reject failed envelopes.")
        elif signal.name == "review_pressure":
            recommendations.append("Run review-triage, then review-worker dry-run before applying queue changes.")
        elif signal.name == "candidate_ratio":
            recommendations.append("Run candidate-pressure to identify dominant candidate kinds before promoting, merging, or cooling memories.")
        elif signal.name == "cold_ratio":
            cycle = next((item for item in signals if item.name == "retention_cycle"), None)
            if cycle and cycle.passed:
                recommendations.append("Cold pressure is high but retention-cycle evidence exists; review it before any explicit live-prune approval.")
            else:
                recommendations.append("Run retention-cycle with representative recall queries before considering any destructive cleanup.")
        elif signal.name == "retention_cycle":
            recommendations.append("Run retention-cycle with representative recall queries to prove backup, cold export, prune-plan, and shadow-prune.")
        elif signal.name == "backup":
            recommendations.append("Create and verify a fresh backup, then run restore-drill.")
        elif signal.name == "recall_regression":
            recommendations.append("Inspect recall-regression drift before changing ranking, compression, or pruning.")
    if not recommendations:
        recommendations.append("Memory health is within current operating budgets; continue scheduled worker-loop and milestone backups.")
    return recommendations
