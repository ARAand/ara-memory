from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ara_memory.backup import verify_backup
from ara_memory.backup_stewardship import DEFAULT_TARGET_BACKUP_BYTES
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
    backup_target_bytes: int | None = DEFAULT_TARGET_BACKUP_BYTES,
    retention_cycle_max_age_hours: float = 72.0,
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
) -> HealthReport:
    stats = memory.stats()
    spool = memory.spool_stats()
    doctor = memory.doctor(scope=scope, recall_query=query, recall_budget=recall_budget, hot_budget=hot_budget)
    retention = memory.retention(scope=scope, cold_limit=5)
    triage = memory.review_triage(scope=scope, limit=review_limit, examples_per_group=1)
    latest_backup = _latest_backup(memory.store.root)
    backup_stewardship = _backup_stewardship_report(memory, stats, target_backup_bytes=backup_target_bytes)
    latest_retention_cycle = _latest_retention_cycle(memory.store.root, scope=scope)
    cold_signal = _cold_ratio_signal(retention.as_dict())
    cold_stewardship = (
        memory.cold_stewardship(
            scope=scope,
            group_limit=3,
            examples_per_group=0,
            max_cycle_age_hours=retention_cycle_max_age_hours,
        )
        if not cold_signal.passed
        else None
    )

    signals = [
        HealthSignal(
            name="doctor",
            passed=doctor.passed,
            detail="all doctor checks passed" if doctor.passed else _failed_doctor_detail(doctor.as_dict()),
            value=doctor.as_dict(),
        ),
        _spool_signal(spool),
        _review_pressure_signal(triage.as_dict()),
        _relation_review_pressure_signal(memory.store, scope=scope, limit=review_limit),
        _candidate_ratio_signal(retention.as_dict()),
        cold_signal,
        _retention_cycle_signal(
            latest_retention_cycle,
            retention.as_dict(),
            cold_stewardship.as_dict() if cold_stewardship else None,
            max_age_hours=retention_cycle_max_age_hours,
        ),
        _backup_signal(latest_backup, max_age_hours=backup_max_age_hours),
        _backup_pressure_signal(stats, backup_stewardship, target_backup_bytes=backup_target_bytes),
    ]
    if regression_cases is not None:
        regression = memory.recall_regression(regression_cases, baseline=regression_baseline)
        signals.append(_recall_regression_signal(regression.as_dict()))

    score = _score(signals)
    hard_fail = any(not signal.passed and signal.severity == "error" for signal in signals)
    has_warning = any(not signal.passed and signal.severity == "warning" for signal in signals)
    passed = not hard_fail
    status = "pass" if passed and not has_warning and score >= 90 else "watch" if passed else "fail"
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
            "backup_stewardship": backup_stewardship,
            "latest_retention_cycle": latest_retention_cycle,
            "cold_stewardship": cold_stewardship.as_dict() if cold_stewardship else None,
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


def _relation_review_pressure_signal(store: Any, *, scope: str, limit: int) -> HealthSignal:
    from ara_memory.relation_merge import list_relation_merge_review_queue

    report = list_relation_merge_review_queue(store, scope=scope, status="open", limit=limit)
    items = report.open_items
    total_open = len(items)
    fail_count = sum(1 for item in items if item.get("review_status") == "fail")
    watch_count = sum(1 for item in items if item.get("review_status") == "watch")
    value = {
        "total_open": total_open,
        "fail_count": fail_count,
        "watch_count": watch_count,
        "items": items[:10],
    }
    if fail_count:
        return HealthSignal(
            "relation_review_pressure",
            False,
            f"{fail_count} failed relation merge review item(s) require repair before further graph merges",
            value=value,
        )
    if total_open:
        return HealthSignal(
            "relation_review_pressure",
            False,
            f"{total_open} open relation merge review item(s) need inspection",
            severity="warning",
            value=value,
        )
    return HealthSignal("relation_review_pressure", True, "0 open relation merge review items", value=value)


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


def _recall_regression_signal(report: dict[str, Any]) -> HealthSignal:
    cases = [item for item in report.get("cases", []) if isinstance(item, dict)]
    baseline = [item for item in report.get("baseline_comparison", []) if isinstance(item, dict)]
    cases_passed = all(bool(item.get("passed", False)) for item in cases)
    baseline_passed = all(bool(item.get("passed", False)) for item in baseline)
    value = {
        **report,
        "cases_passed": cases_passed,
        "baseline_passed": baseline_passed,
    }
    if cases_passed and baseline_passed:
        return HealthSignal(
            "recall_regression",
            True,
            "recall regression passed",
            value=value,
        )
    if cases_passed:
        return HealthSignal(
            "recall_regression",
            False,
            "recall regression cases passed; baseline drift needs review",
            severity="warning",
            value=value,
        )
    return HealthSignal(
        "recall_regression",
        False,
        "recall regression cases failed",
        value=value,
    )


def _backup_stewardship_report(
    memory: Any,
    stats: dict[str, Any],
    *,
    target_backup_bytes: int | None,
) -> dict[str, Any] | None:
    if target_backup_bytes is None:
        return None
    backup_bytes = int(stats.get("backup_bytes", 0) or 0)
    if backup_bytes <= target_backup_bytes:
        return None
    return memory.backup_stewardship(
        target_backup_bytes=target_backup_bytes,
        quarantine_failed=True,
        write_cache=False,
    ).as_dict()


def _backup_pressure_signal(
    stats: dict[str, Any],
    stewardship: dict[str, Any] | None,
    *,
    target_backup_bytes: int | None,
) -> HealthSignal:
    backup_bytes = int(stats.get("backup_bytes", 0) or 0)
    if target_backup_bytes is None:
        return HealthSignal("backup_pressure", True, "backup byte target disabled", value=backup_bytes)
    if backup_bytes <= target_backup_bytes:
        return HealthSignal(
            "backup_pressure",
            True,
            f"backup bytes {backup_bytes} are within target {target_backup_bytes}",
            value={"backup_bytes": backup_bytes, "target_backup_bytes": target_backup_bytes},
        )
    if stewardship is None:
        return HealthSignal(
            "backup_pressure",
            False,
            f"backup bytes {backup_bytes} exceed target {target_backup_bytes}",
            severity="warning",
            value={"backup_bytes": backup_bytes, "target_backup_bytes": target_backup_bytes},
        )
    totals = stewardship.get("totals", {})
    if not stewardship.get("passed"):
        return HealthSignal(
            "backup_pressure",
            False,
            "backup stewardship is blocked; inspect backup directory safety before cleanup",
            severity="warning",
            value=stewardship,
        )
    quarantine_candidates = int(totals.get("quarantine_candidates", 0) or 0)
    if quarantine_candidates:
        after_quarantine = int(totals.get("bytes_after_quarantine_candidates", backup_bytes) or backup_bytes)
        return HealthSignal(
            "backup_pressure",
            False,
            f"backup bytes {backup_bytes} exceed target {target_backup_bytes}; "
            f"{quarantine_candidates} failed-verification backups can be quarantined to leave {after_quarantine} live backup bytes",
            severity="warning",
            value=stewardship,
        )
    candidates = int(totals.get("delete_candidates", 0) or 0)
    if candidates:
        return HealthSignal(
            "backup_pressure",
            False,
            f"backup bytes {backup_bytes} exceed target {target_backup_bytes}; "
            f"{candidates} redundant verified backups can leave {totals.get('bytes_after_candidates')} bytes",
            severity="warning",
            value=stewardship,
        )
    return HealthSignal(
        "backup_pressure",
        False,
        f"backup bytes {backup_bytes} exceed target {target_backup_bytes}, but no redundant verified backups are eligible under the keep policy",
        severity="warning",
        value=stewardship,
    )


def _retention_cycle_signal(
    latest: dict[str, Any] | None,
    retention: dict[str, Any],
    cold_stewardship: dict[str, Any] | None = None,
    *,
    max_age_hours: float = 72.0,
) -> HealthSignal:
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
    age = float(latest.get("age_hours") or 0.0)
    if age > max_age_hours:
        return HealthSignal(
            "retention_cycle",
            False,
            f"latest retention-cycle is stale: {age:.1f}h old",
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
    if cold_stewardship:
        evidence = cold_stewardship.get("cycle_evidence") or {}
        if evidence.get("required"):
            if not evidence.get("matches_current"):
                current = evidence.get("current") or {}
                cycle = evidence.get("cycle") or {}
                if not evidence.get("identity_available"):
                    return HealthSignal(
                        "retention_cycle",
                        False,
                        "latest retention-cycle lacks a cold identity fingerprint; rerun retention-cycle evidence",
                        severity="warning",
                        value=latest,
                    )
                if evidence.get("count_match") and not evidence.get("identity_match"):
                    mismatched = ", ".join(evidence.get("identity_mismatched_sets") or []) or "unknown"
                    return HealthSignal(
                        "retention_cycle",
                        False,
                        f"cold identity fingerprint drifted since latest retention-cycle: mismatched_sets={mismatched}",
                        severity="warning",
                        value=latest,
                    )
                if evidence.get("protected_only_drift"):
                    current = evidence.get("current") or {}
                    cycle = evidence.get("cycle") or {}
                    return HealthSignal(
                        "retention_cycle",
                        True,
                        "latest retention-cycle passed; prunable source-event set is unchanged while protected cold evidence grew: "
                        f"current cold={current.get('cold_capsules')}, protected={current.get('protected_events')}, "
                        f"prunable={current.get('prunable_events')} vs cycle cold={cycle.get('cold_capsules')}, "
                        f"protected={cycle.get('protected_events')}, prunable={cycle.get('prunable_events')}",
                        value=latest,
                    )
                return HealthSignal(
                    "retention_cycle",
                    False,
                    "cold totals drifted since latest retention-cycle: "
                    f"current cold={current.get('cold_capsules')}, protected={current.get('protected_events')}, "
                    f"prunable={current.get('prunable_events')} vs cycle cold={cycle.get('cold_capsules')}, "
                    f"protected={cycle.get('protected_events')}, prunable={cycle.get('prunable_events')}",
                    severity="warning",
                    value=latest,
                )
            if not evidence.get("shadow_events_preserved"):
                return HealthSignal(
                    "retention_cycle",
                    False,
                    "latest retention-cycle did not prove source-event preservation",
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
            verification = verify_backup(path, trust_root=root)
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
        "cold_identity": plan.get("cold_identity"),
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
        elif signal.name == "relation_review_pressure":
            value = signal.value if isinstance(signal.value, dict) else {}
            if int(value.get("fail_count", 0) or 0):
                recommendations.append(
                    "Run relation-merge-review-queue and repair or explain failed relation merge witnesses before applying more graph merges."
                )
            else:
                recommendations.append(
                    "Run relation-merge-review-queue, inspect open relation merge review items, then rerun relation-merge-review --record-queue."
                )
        elif signal.name == "candidate_ratio":
            recommendations.append("Run candidate-pressure to identify dominant candidate kinds before promoting, merging, or cooling memories.")
        elif signal.name == "cold_ratio":
            cycle = next((item for item in signals if item.name == "retention_cycle"), None)
            if cycle and cycle.passed:
                recommendations.append(
                    "Cold pressure is high but retention-cycle evidence exists; run cold-stewardship before any explicit live-prune approval."
                )
            else:
                recommendations.append(
                    "Run cold-stewardship, then retention-cycle with representative recall queries before considering any destructive cleanup."
                )
        elif signal.name == "retention_cycle":
            recommendations.append("Run retention-cycle with representative recall queries to prove backup, cold export, prune-plan, and shadow-prune.")
        elif signal.name == "backup":
            recommendations.append("Create and verify a fresh backup, then run restore-drill.")
        elif signal.name == "backup_pressure":
            totals = (signal.value or {}).get("totals", {}) if isinstance(signal.value, dict) else {}
            if int(totals.get("quarantine_candidates", 0) or 0) > 0:
                recommendations.append(
                    "Run backup-stewardship dry-run with --quarantine-failed, then apply only reviewed failed-backup moves with exact quarantine confirmation."
                )
            elif int(totals.get("delete_candidates", 0) or 0) > 0:
                recommendations.append(
                    "Run backup-stewardship dry-run, then apply only reviewed redundant verified backup deletions with exact confirmation."
                )
            else:
                recommendations.append(
                    "Backup bytes exceed target but no redundant verified backups are eligible; review the target or protected backup policy."
                )
        elif signal.name == "recall_regression":
            value = signal.value if isinstance(signal.value, dict) else {}
            if value.get("cases_passed") and not value.get("baseline_passed"):
                recommendations.append(
                    "Recall cases still pass but baseline drifted; review selected capsules and refresh the baseline only after confirming the new pack is better."
                )
            else:
                recommendations.append("Recall regression cases failed; inspect recall output before changing ranking, compression, or pruning.")
    if not recommendations:
        recommendations.append("Memory health is within current operating budgets; continue scheduled worker-loop and milestone backups.")
    return recommendations
