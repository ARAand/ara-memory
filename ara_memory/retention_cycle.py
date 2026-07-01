from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ara_memory.backup import create_backup, verify_backup
from ara_memory.cold_export import export_cold_capsules, verify_cold_export
from ara_memory.lock import FileLock
from ara_memory.models import utc_now
from ara_memory.prune import PrunePlanner, ShadowPruner
from ara_memory.storage import MemoryStore


@dataclass(slots=True)
class RetentionCycleReport:
    scope: str | None
    passed: bool
    backup: dict[str, Any]
    backup_verification: dict[str, Any]
    cold_export: dict[str, Any]
    cold_export_verification: dict[str, Any]
    prune_plan: dict[str, Any]
    shadow_prune: dict[str, Any] | None
    recommendations: list[str]
    report_path: str | None = None
    lock: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "backup": self.backup,
            "backup_verification": self.backup_verification,
            "cold_export": self.cold_export,
            "cold_export_verification": self.cold_export_verification,
            "prune_plan": self.prune_plan,
            "shadow_prune": self.shadow_prune,
            "recommendations": self.recommendations,
            "report_path": self.report_path,
            "lock": self.lock,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        plan_totals = self.prune_plan.get("totals", {})
        shadow_deletion = (self.shadow_prune or {}).get("deletion", {})
        lines = [f"# Ara Memory Retention Cycle: {scope}"]
        lines.append(f"status: {'pass' if self.passed else 'blocked'}")
        if self.report_path:
            lines.append(f"report: {self.report_path}")
        if self.lock:
            lines.append(
                "lock: "
                f"acquired={self.lock.get('acquired')}, "
                f"reason={self.lock.get('reason', '')}"
            )
        lines.append(f"backup: {self.backup.get('path')}")
        lines.append(f"cold_export: {self.cold_export.get('path')}")
        lines.append(
            "plan: "
            f"cold_capsules={plan_totals.get('cold_capsules', 0)}, "
            f"protected_events={plan_totals.get('protected_events', 0)}, "
            f"prunable_events={plan_totals.get('prunable_events', 0)}"
        )
        if self.shadow_prune is None:
            lines.append("shadow_prune: skipped")
        else:
            lines.append(
                "shadow_prune: "
                f"passed={self.shadow_prune.get('passed')}, "
                f"capsules_removed={shadow_deletion.get('capsules_removed', 0)}, "
                f"events_removed={shadow_deletion.get('events_removed', 0)}"
            )
        lines.append("## Gates")
        lines.append(f"- backup_verification: {self.backup_verification.get('passed')}")
        lines.append(f"- cold_export_verification: {self.cold_export_verification.get('passed')}")
        lines.append(f"- prune_plan: {self.prune_plan.get('passed')}")
        if self.shadow_prune is not None:
            lines.append(f"- shadow_prune: {self.shadow_prune.get('passed')}")
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


class RetentionCycleRunner:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        backup_output: Path | None = None,
        cold_output: Path | None = None,
        limit: int | None = None,
        recall_queries: list[str] | None = None,
        recall_budget: int = 1200,
        include_global: bool = True,
        doctor_query: str = "current memory state",
        shadow: bool = True,
        report_output: Path | None = None,
        use_lock: bool = True,
        lock_stale_seconds: int = 3600,
    ) -> RetentionCycleReport:
        self.store.init()
        recall_queries = recall_queries or []
        if use_lock:
            lock = FileLock(self.store.root, "worker", stale_seconds=lock_stale_seconds)
            lock_result = lock.acquire()
            if not lock_result.acquired:
                return RetentionCycleReport(
                    scope=scope,
                    passed=False,
                    backup={},
                    backup_verification={},
                    cold_export={},
                    cold_export_verification={},
                    prune_plan={},
                    shadow_prune=None,
                    recommendations=[
                        "Retention-cycle skipped because the memory worker lock is already held.",
                        "Rerun after the active worker finishes, or use --no-lock only when the store is otherwise quiescent.",
                    ],
                    lock=lock_result.as_dict(),
                )
            try:
                return self._run_unlocked(
                    scope=scope,
                    backup_output=backup_output,
                    cold_output=cold_output,
                    limit=limit,
                    recall_queries=recall_queries,
                    recall_budget=recall_budget,
                    include_global=include_global,
                    doctor_query=doctor_query,
                    shadow=shadow,
                    report_output=report_output,
                    lock=lock_result.as_dict(),
                )
            finally:
                lock.release()
        return self._run_unlocked(
            scope=scope,
            backup_output=backup_output,
            cold_output=cold_output,
            limit=limit,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
            doctor_query=doctor_query,
            shadow=shadow,
            report_output=report_output,
            lock=None,
        )

    def _run_unlocked(
        self,
        *,
        scope: str | None,
        backup_output: Path | None,
        cold_output: Path | None,
        limit: int | None,
        recall_queries: list[str],
        recall_budget: int,
        include_global: bool,
        doctor_query: str,
        shadow: bool,
        report_output: Path | None,
        lock: dict[str, Any] | None,
    ) -> RetentionCycleReport:

        backup = create_backup(self.store, output=backup_output)
        backup_payload = backup.as_dict()
        backup_verification = verify_backup(backup.path, trust_root=self.store.root)

        cold = export_cold_capsules(self.store, output=cold_output, scope=scope, limit=limit, order="oldest")
        cold_payload = cold.as_dict()
        cold_verification = verify_cold_export(cold.path, trust_root=self.store.root)

        plan = PrunePlanner(self.store).run(
            scope=scope,
            limit=limit,
            export_path=cold.path,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
        )
        plan_payload = plan.as_dict()

        shadow_payload = None
        if shadow and backup_verification.get("passed") and cold_verification.get("passed") and plan.passed:
            shadow_report = ShadowPruner(self.store).run(
                backup_path=backup.path,
                export_path=cold.path,
                scope=scope,
                limit=limit,
                recall_queries=recall_queries,
                recall_budget=recall_budget,
                include_global=include_global,
                doctor_query=doctor_query,
            )
            shadow_payload = shadow_report.as_dict()

        passed = (
            bool(backup_verification.get("passed"))
            and bool(cold_verification.get("passed"))
            and plan.passed
            and shadow_payload is not None
            and bool(shadow_payload.get("passed"))
        )
        report = RetentionCycleReport(
            scope=scope,
            passed=passed,
            backup=backup_payload,
            backup_verification=backup_verification,
            cold_export=cold_payload,
            cold_export_verification=cold_verification,
            prune_plan=plan_payload,
            shadow_prune=shadow_payload,
            recommendations=_recommend(passed, plan_payload, shadow_payload, recall_queries=recall_queries),
            lock=lock,
        )
        report.report_path = str(_write_report(self.store, report, output=report_output))
        return report


def _recommend(
    passed: bool,
    plan: dict[str, Any],
    shadow: dict[str, Any] | None,
    *,
    recall_queries: list[str],
) -> list[str]:
    recommendations: list[str] = []
    totals = plan.get("totals", {})
    if not recall_queries:
        recommendations.append("Add representative recall queries; retention-cycle will block pruning without them.")
    if totals.get("protected_events", 0):
        recommendations.append(f"Keep {totals['protected_events']} protected source events cited by active memories.")
    if shadow is None:
        recommendations.append("Shadow prune did not run; do not prepare live pruning from this cycle.")
    elif shadow.get("passed"):
        recommendations.append("Shadow prune passed in a restored sandbox; live pruning still requires explicit approval.")
    if not passed:
        recommendations.append("Do not prune; at least one retention-cycle gate is blocked.")
    elif not recommendations:
        recommendations.append("Retention cycle passed; keep the generated backup and cold export as review evidence.")
    return recommendations


def _write_report(store: MemoryStore, report: RetentionCycleReport, *, output: Path | None) -> Path:
    path = output or _default_report_path(store.root, scope=report.scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["report_path"] = str(path)
    payload["created_at"] = utc_now()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _default_report_path(root: Path, *, scope: str | None) -> Path:
    stamp = utc_now().replace(":", "").replace("+", "Z")
    scope_part = _safe_name(scope or "all")
    return root / "archive" / "retention-cycles" / f"{scope_part}-retention-cycle-{stamp}.json"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-") or "all"
