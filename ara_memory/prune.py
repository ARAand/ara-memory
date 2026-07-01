from __future__ import annotations

import json
import secrets
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from ara_memory.backup import restore_backup
from ara_memory.cold_export import verify_cold_export
from ara_memory.doctor import MemoryDoctor
from ara_memory.models import MemoryStatus, new_id, utc_now
from ara_memory.recall import RecallCompiler
from ara_memory.storage import MemoryStore, row_to_capsule


PRUNABLE_STATUSES = (
    MemoryStatus.SUPERSEDED.value,
    MemoryStatus.REJECTED.value,
    MemoryStatus.QUARANTINED.value,
)
ACTIVE_STATUSES = (
    MemoryStatus.CANDIDATE.value,
    MemoryStatus.STABLE.value,
)
LIVE_PRUNE_CONFIRMATION = "DELETE COLD CAPSULES"


@dataclass(slots=True)
class PrunePlanReport:
    scope: str | None
    limit: int | None
    dry_run: bool
    passed: bool
    gates: list[dict[str, Any]]
    totals: dict[str, int]
    recall_checks: list[dict[str, Any]]
    protected_event_ids: list[str]
    prunable_event_ids: list[str]
    candidate_capsule_ids: list[str]
    candidate_capsules: list[dict[str, Any]]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "limit": self.limit,
            "dry_run": self.dry_run,
            "passed": self.passed,
            "gates": self.gates,
            "totals": self.totals,
            "recall_checks": self.recall_checks,
            "protected_event_ids": self.protected_event_ids,
            "prunable_event_ids": self.prunable_event_ids,
            "candidate_capsule_ids": self.candidate_capsule_ids,
            "candidate_capsules": self.candidate_capsules,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Memory Prune Plan: {scope}"]
        lines.append(f"status: {'pass' if self.passed else 'blocked'}")
        lines.append(
            "totals: "
            f"cold_capsules={self.totals['cold_capsules']}, "
            f"prunable_capsules={self.totals['prunable_capsules']}, "
            f"candidate_source_events={self.totals['candidate_source_events']}, "
            f"prunable_events={self.totals['prunable_events']}, "
            f"protected_events={self.totals['protected_events']}"
        )
        lines.append("## Gates")
        for gate in self.gates:
            status = "OK" if gate["passed"] else "BLOCKED"
            lines.append(f"- [{status}] {gate['name']}: {gate['message']}")
        if self.recall_checks:
            lines.append("## Recall Checks")
            for check in self.recall_checks:
                status = "OK" if check["passed"] else "BLOCKED"
                lines.append(
                    f"- [{status}] {check['query']}: tokens={check['estimated_tokens']}, "
                    f"selected={check['selected_capsules']}, cold_overlap={check['cold_overlap']}"
                )
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        if self.candidate_capsules:
            lines.append("## Candidate Capsules")
            for row in self.candidate_capsules[:10]:
                lines.append(f"- {row['id']} [{row['kind']}/{row['status']}] {row['title']}")
        return "\n".join(lines)


@dataclass(slots=True)
class ShadowPruneReport:
    scope: str | None
    passed: bool
    plan: PrunePlanReport
    backup_restore: dict[str, Any]
    deletion: dict[str, Any]
    doctor: dict[str, Any]
    recall_checks: list[dict[str, Any]]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "plan": self.plan.as_dict(),
            "backup_restore": self.backup_restore,
            "deletion": self.deletion,
            "doctor": self.doctor,
            "recall_checks": self.recall_checks,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Memory Shadow Prune: {scope}"]
        lines.append(f"status: {'pass' if self.passed else 'blocked'}")
        lines.append(
            "deletion: "
            f"capsules_removed={self.deletion.get('capsules_removed', 0)}, "
            f"fts_removed={self.deletion.get('fts_removed', 0)}, "
            f"edges_removed={self.deletion.get('edges_removed', 0)}, "
            f"events_removed={self.deletion.get('events_removed', 0)}"
        )
        lines.append("## Plan")
        lines.append(
            f"- passed={self.plan.passed}; cold_capsules={self.plan.totals['cold_capsules']}; "
            f"protected_events={self.plan.totals['protected_events']}; "
            f"prunable_events={self.plan.totals['prunable_events']}"
        )
        lines.append("## Doctor")
        lines.append(f"- passed={self.doctor.get('passed')}")
        for check in self.doctor.get("checks", []):
            status = "OK" if check["passed"] else "FAIL"
            lines.append(f"- [{status}] {check['name']}: {check['detail']}")
        if self.recall_checks:
            lines.append("## Recall Checks")
            for check in self.recall_checks:
                status = "OK" if check["passed"] else "FAIL"
                lines.append(
                    f"- [{status}] {check['query']}: before={check['before_tokens']} tokens, "
                    f"after={check['after_tokens']} tokens, after_selected={check['after_selected_capsules']}"
                )
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


@dataclass(slots=True)
class LivePruneApproval:
    approval_id: str
    token: str
    expires_at: str
    shadow: ShadowPruneReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "token": self.token,
            "expires_at": self.expires_at,
            "shadow": self.shadow.as_dict(),
            "warning": "Store this token only long enough to run live-prune; it authorizes one irreversible cold-capsule deletion.",
        }


@dataclass(slots=True)
class LivePruneReport:
    scope: str | None
    passed: bool
    operation_id: str | None
    approval_id: str | None
    deletion: dict[str, Any]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "operation_id": self.operation_id,
            "approval_id": self.approval_id,
            "deletion": self.deletion,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Memory Live Prune: {scope}"]
        lines.append(f"status: {'pass' if self.passed else 'blocked'}")
        if self.operation_id:
            lines.append(f"operation_id: {self.operation_id}")
        lines.append(
            "deletion: "
            f"capsules_removed={self.deletion.get('capsules_removed', 0)}, "
            f"edges_removed={self.deletion.get('edges_removed', 0)}, "
            f"events_removed={self.deletion.get('events_removed', 0)}"
        )
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


class PrunePlanner:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        limit: int | None = None,
        export_path: Path | None = None,
        recall_queries: list[str] | None = None,
        recall_budget: int = 1200,
        include_global: bool = True,
    ) -> PrunePlanReport:
        self.store.init()
        candidates = _cold_capsules(self.store, scope=scope, limit=limit)
        candidate_ids = {row["id"] for row in candidates}
        ordered_candidate_ids = [row["id"] for row in candidates]
        source_event_ids = sorted({event_id for row in candidates for event_id in row["source_event_ids"]})
        active_event_ids = _active_event_ids(self.store, scope=scope)
        protected_event_ids = sorted(set(source_event_ids).intersection(active_event_ids))
        prunable_event_ids = sorted(set(source_event_ids) - set(protected_event_ids))

        gates = []
        gates.append(_gate(bool(candidates), "cold_candidates", f"{len(candidates)} cold capsules selected."))
        gates.append(
            _export_gate(
                export_path,
                scope=scope,
                expected_capsules=len(candidates),
                limit=limit,
                expected_ids=candidate_ids,
            )
        )
        recall_checks = _recall_checks(
            self.store,
            recall_queries or [],
            scope=scope or "global",
            budget=recall_budget,
            include_global=include_global,
            candidate_ids=candidate_ids,
        )
        if recall_queries:
            gates.append(
                _gate(
                    all(check["passed"] for check in recall_checks),
                    "recall_regression",
                    "Recall checks do not select planned cold capsules."
                    if all(check["passed"] for check in recall_checks)
                    else "At least one recall check still depends on a planned cold capsule.",
                )
            )
        else:
            gates.append(_gate(False, "recall_regression", "No recall queries supplied for pruning simulation."))

        totals = {
            "cold_capsules": len(candidates),
            "prunable_capsules": len(candidates),
            "candidate_source_events": len(source_event_ids),
            "prunable_events": len(prunable_event_ids),
            "protected_events": len(protected_event_ids),
        }
        recommendations = _recommend(totals, gates, recall_queries=recall_queries or [])
        return PrunePlanReport(
            scope=scope,
            limit=limit,
            dry_run=True,
            passed=all(gate["passed"] for gate in gates),
            gates=gates,
            totals=totals,
            recall_checks=recall_checks,
            protected_event_ids=protected_event_ids,
            prunable_event_ids=prunable_event_ids,
            candidate_capsule_ids=ordered_candidate_ids,
            candidate_capsules=[
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "status": row["status"],
                    "title": row["title"],
                    "updated_at": row["updated_at"],
                }
                for row in candidates[:20]
            ],
            recommendations=recommendations,
        )


class ShadowPruner:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        backup_path: Path,
        export_path: Path,
        scope: str | None = None,
        limit: int | None = None,
        recall_queries: list[str] | None = None,
        recall_budget: int = 1200,
        include_global: bool = True,
        doctor_query: str = "current memory state",
    ) -> ShadowPruneReport:
        recall_queries = recall_queries or []
        planner = PrunePlanner(self.store)
        plan = planner.run(
            scope=scope,
            limit=limit,
            export_path=export_path,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
        )
        if not plan.passed:
            return ShadowPruneReport(
                scope=scope,
                passed=False,
                plan=plan,
                backup_restore={},
                deletion={},
                doctor={"passed": False, "checks": []},
                recall_checks=[],
                recommendations=["Shadow prune blocked because prune-plan did not pass."],
            )

        with tempfile.TemporaryDirectory() as tmp:
            shadow_root = Path(tmp) / "shadow-memory"
            backup_restore = restore_backup(backup_path, shadow_root, trust_root=self.store.root)
            shadow_store = MemoryStore(shadow_root)
            before_recalls = _recall_snapshot(
                self.store,
                queries=recall_queries,
                scope=scope or "global",
                budget=recall_budget,
                include_global=include_global,
            )
            candidate_ids = [row["id"] for row in _cold_capsules(self.store, scope=scope, limit=limit)]
            deletion = _delete_capsules_only(shadow_store, candidate_ids)
            after_recalls = _recall_snapshot(
                shadow_store,
                queries=recall_queries,
                scope=scope or "global",
                budget=recall_budget,
                include_global=include_global,
            )
            doctor_report = MemoryDoctor(shadow_store).run(
                scope=scope or "global",
                recall_query=doctor_query,
                recall_budget=recall_budget,
                hot_budget=1200,
                include_global=include_global,
            )
            recall_checks = _compare_recalls(before_recalls, after_recalls)
            passed = (
                bool(backup_restore.get("passed"))
                and deletion.get("capsules_removed", 0) == len(candidate_ids)
                and deletion.get("events_removed", 0) == 0
                and doctor_report.passed
                and all(check["passed"] for check in recall_checks)
            )
            return ShadowPruneReport(
                scope=scope,
                passed=passed,
                plan=plan,
                backup_restore=backup_restore,
                deletion=deletion,
                doctor=doctor_report.as_dict(),
                recall_checks=recall_checks,
                recommendations=_shadow_recommend(passed, deletion, plan),
            )


class LivePruneController:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def prepare(
        self,
        *,
        backup_path: Path,
        export_path: Path,
        scope: str | None = None,
        limit: int | None = None,
        recall_queries: list[str] | None = None,
        recall_budget: int = 1200,
        include_global: bool = True,
        doctor_query: str = "current memory state",
        ttl_minutes: int = 30,
    ) -> LivePruneApproval:
        self.store.init()
        shadow = ShadowPruner(self.store).run(
            backup_path=backup_path,
            export_path=export_path,
            scope=scope,
            limit=limit,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
            doctor_query=doctor_query,
        )
        if not shadow.passed:
            raise ValueError("Cannot prepare live prune because shadow-prune did not pass.")
        token = secrets.token_urlsafe(24)
        approval_id = new_id("prune_approval")
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
        with self.store.session() as conn:
            conn.execute(
                """
                INSERT INTO prune_approvals(
                  id, token_hash, scope, backup_path, cold_export_path, recall_queries_json,
                  plan_json, shadow_json, expires_at, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    _token_hash(token),
                    scope,
                    str(backup_path),
                    str(export_path),
                    json.dumps(recall_queries or [], ensure_ascii=False, sort_keys=True),
                    json.dumps(shadow.plan.as_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(shadow.as_dict(), ensure_ascii=False, sort_keys=True),
                    expires_at,
                    "prepared",
                    utc_now(),
                ),
            )
        return LivePruneApproval(approval_id=approval_id, token=token, expires_at=expires_at, shadow=shadow)

    def run(self, *, approval_token: str, confirmation: str) -> LivePruneReport:
        self.store.init()
        if confirmation != LIVE_PRUNE_CONFIRMATION:
            return LivePruneReport(
                scope=None,
                passed=False,
                operation_id=None,
                approval_id=None,
                deletion={},
                recommendations=[f"Confirmation must exactly match: {LIVE_PRUNE_CONFIRMATION}"],
            )
        approval = self._load_approval(approval_token)
        if approval is None:
            return LivePruneReport(
                scope=None,
                passed=False,
                operation_id=None,
                approval_id=None,
                deletion={},
                recommendations=["Approval token was not found."],
            )
        if approval["status"] != "prepared":
            return _blocked_live_report(approval, f"Approval status is {approval['status']}, not prepared.")
        if _is_expired(approval["expires_at"]):
            self._mark_approval_status(approval["id"], "expired")
            return _blocked_live_report(approval, "Approval token is expired.")

        scope = approval["scope"]
        shadow = json.loads(approval["shadow_json"])
        plan = shadow["plan"]
        approved_candidate_ids = plan.get("candidate_capsule_ids")
        if not approved_candidate_ids:
            return _blocked_live_report(
                approval,
                "Approved prune plan does not contain exact capsule IDs; rerun prepare-live-prune.",
            )
        current_candidate_ids = [
            row["id"] for row in _cold_capsules(self.store, scope=scope, limit=plan.get("limit"))
        ]
        if set(current_candidate_ids) != set(approved_candidate_ids):
            return _blocked_live_report(
                approval,
                "Approved cold capsule set changed or no longer matches live memory; rerun retention-cycle and prepare-live-prune.",
            )
        verification = verify_cold_export(Path(approval["cold_export_path"]))
        if not verification.get("passed"):
            return _blocked_live_report(approval, "Approved cold export no longer verifies.")
        exported_ids = _exported_capsule_ids(Path(approval["cold_export_path"]))
        missing_export_ids = sorted(set(approved_candidate_ids) - exported_ids)
        if missing_export_ids:
            return _blocked_live_report(
                approval,
                "Approved cold export no longer covers the approved capsule IDs; rerun retention-cycle and prepare-live-prune.",
            )

        deletion = _delete_capsules_only(self.store, approved_candidate_ids)
        if deletion.get("blocked") or deletion.get("capsules_removed", 0) != len(approved_candidate_ids):
            return _blocked_live_report(
                approval,
                deletion.get(
                    "block_reason",
                    "Approved cold capsule set changed during deletion; rerun retention-cycle and prepare-live-prune.",
                ),
            )
        operation_id = new_id("irrev_op")
        manifest = {
            "approval_id": approval["id"],
            "scope": scope,
            "backup_path": approval["backup_path"],
            "cold_export_path": approval["cold_export_path"],
            "deletion": deletion,
            "approved_plan_totals": plan["totals"],
            "approved_candidate_ids": approved_candidate_ids,
            "event_deletion_policy": "events are never deleted by live-prune",
        }
        now = utc_now()
        with self.store.session() as conn:
            conn.execute(
                """
                INSERT INTO irreversible_operations(
                  id, operation, scope, approval_id, status, manifest_json, created_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    operation_id,
                    "live-prune-cold-capsules",
                    scope,
                    approval["id"],
                    "completed",
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE prune_approvals SET status = ?, used_at = ? WHERE id = ?",
                ("used", now, approval["id"]),
            )
        return LivePruneReport(
            scope=scope,
            passed=True,
            operation_id=operation_id,
            approval_id=approval["id"],
            deletion=deletion,
            recommendations=[
                "Live prune completed and was recorded in irreversible_operations.",
                "Run doctor, maintenance, backup, and restore-drill immediately after live pruning.",
            ],
        )

    def list_operations(self, *, limit: int = 20) -> list[dict[str, Any]]:
        self.store.init()
        with self.store.session() as conn:
            return [
                {
                    **dict(row),
                    "manifest": json.loads(row["manifest_json"]),
                }
                for row in conn.execute(
                    """
                    SELECT *
                    FROM irreversible_operations
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            ]

    def _load_approval(self, token: str) -> dict[str, Any] | None:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT * FROM prune_approvals WHERE token_hash = ?",
                (_token_hash(token),),
            ).fetchone()
            return dict(row) if row else None

    def _mark_approval_status(self, approval_id: str, status: str) -> None:
        with self.store.session() as conn:
            conn.execute(
                "UPDATE prune_approvals SET status = ? WHERE id = ?",
                (status, approval_id),
            )


def _gate(passed: bool, name: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "passed": passed,
        "message": message,
    }
    if details:
        payload["details"] = details
    return payload


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _is_expired(expires_at: str) -> bool:
    return datetime.fromisoformat(expires_at) <= datetime.now(timezone.utc)


def _blocked_live_report(approval: dict[str, Any], message: str) -> LivePruneReport:
    return LivePruneReport(
        scope=approval.get("scope"),
        passed=False,
        operation_id=None,
        approval_id=approval.get("id"),
        deletion={},
        recommendations=[message],
    )


def _cold_capsules(store: MemoryStore, *, scope: str | None, limit: int | None) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in PRUNABLE_STATUSES)
    clauses = [f"status IN ({placeholders})"]
    args: list[Any] = [*PRUNABLE_STATUSES]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    sql = f"""
        SELECT *
        FROM capsules
        WHERE {' AND '.join(clauses)}
        ORDER BY updated_at ASC, id ASC
    """
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    with store.session() as conn:
        return [row_to_capsule(row) for row in conn.execute(sql, args)]


def _active_event_ids(store: MemoryStore, *, scope: str | None) -> set[str]:
    return store.source_event_ids_for_statuses(ACTIVE_STATUSES, scope=scope)


def _export_gate(
    export_path: Path | None,
    *,
    scope: str | None,
    expected_capsules: int,
    limit: int | None,
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    if export_path is None:
        return _gate(False, "cold_export", "No verified cold export supplied.")
    verification = verify_cold_export(export_path)
    manifest = verification.get("manifest", {})
    exported_ids = _exported_capsule_ids(export_path) if verification.get("passed") else set()
    scope_ok = manifest.get("scope") == scope
    count_ok = manifest.get("capsule_count", 0) >= expected_capsules
    limit_ok = limit is None or manifest.get("limit") in (None, limit)
    coverage_ok = expected_ids is None or expected_ids.issubset(exported_ids)
    passed = bool(verification.get("passed")) and scope_ok and count_ok and limit_ok and coverage_ok
    return _gate(
        passed,
        "cold_export",
        "Verified cold export covers the planned capsule set."
        if passed
        else "Cold export is missing, invalid, scoped differently, smaller than, or missing capsules from the planned set.",
        details={
            "verification": verification,
            "scope_ok": scope_ok,
            "count_ok": count_ok,
            "limit_ok": limit_ok,
            "coverage_ok": coverage_ok,
            "missing_capsule_ids": sorted((expected_ids or set()) - exported_ids)[:20],
        },
    )


def _recall_checks(
    store: MemoryStore,
    queries: list[str],
    *,
    scope: str,
    budget: int,
    include_global: bool,
    candidate_ids: set[str],
) -> list[dict[str, Any]]:
    checks = []
    compiler = RecallCompiler(store)
    for query in queries:
        result = compiler.recall(query, scope=scope, budget=budget, include_global=include_global)
        selected_ids = set(result.diagnostics.get("selected_capsule_ids", []))
        overlap = sorted(selected_ids.intersection(candidate_ids))
        checks.append(
            {
                "query": query,
                "passed": not overlap,
                "estimated_tokens": result.diagnostics["estimated_tokens_after"],
                "selected_capsules": len(selected_ids),
                "cold_overlap": overlap,
            }
        )
    return checks


def _exported_capsule_ids(export_path: Path) -> set[str]:
    with zipfile.ZipFile(export_path, "r") as zf:
        if "capsules.jsonl" not in zf.namelist():
            return set()
        raw = zf.read("capsules.jsonl").decode("utf-8")
    ids = set()
    for line in raw.splitlines():
        if line.strip():
            payload = json.loads(line)
            if "id" in payload:
                ids.add(payload["id"])
    return ids


def _recommend(totals: dict[str, int], gates: list[dict[str, Any]], *, recall_queries: list[str]) -> list[str]:
    recommendations = []
    if any(not gate["passed"] for gate in gates):
        recommendations.append("Do not prune; at least one safety gate is blocked.")
    if totals["protected_events"]:
        recommendations.append(
            f"Keep {totals['protected_events']} source events because active candidate/stable capsules still cite them."
        )
    if totals["prunable_events"]:
        recommendations.append(
            f"{totals['prunable_events']} source events appear to be cited only by planned cold capsules."
        )
    if not recall_queries:
        recommendations.append("Add representative recall queries before turning this dry-run into a destructive prune.")
    if not recommendations:
        recommendations.append("Dry-run passed; destructive pruning still requires a separate explicit command and fresh backup.")
    return recommendations


def _delete_capsules_only(store: MemoryStore, capsule_ids: list[str]) -> dict[str, Any]:
    store.init()
    if not capsule_ids:
        return {
            "blocked": False,
            "capsules_requested": 0,
            "capsules_removed": 0,
            "fts_removed": 0,
            "edges_removed": 0,
            "actions_removed": 0,
            "quality_scores_removed": 0,
            "review_queue_removed": 0,
            "source_links_removed": 0,
            "events_removed": 0,
        }
    placeholders = ",".join("?" for _ in capsule_ids)
    with store.session() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            f"SELECT id, status FROM capsules WHERE id IN ({placeholders})",
            capsule_ids,
        ).fetchall()
        found_ids = {row["id"] for row in rows}
        missing_ids = sorted(set(capsule_ids) - found_ids)
        non_prunable_ids = sorted(row["id"] for row in rows if row["status"] not in PRUNABLE_STATUSES)
        if missing_ids or non_prunable_ids:
            return {
                "blocked": True,
                "block_reason": (
                    "Approved cold capsule set changed during deletion; "
                    "rerun retention-cycle and prepare-live-prune."
                ),
                "missing_capsule_ids": missing_ids,
                "non_prunable_capsule_ids": non_prunable_ids,
                "capsules_requested": len(capsule_ids),
                "capsules_removed": 0,
                "fts_removed": 0,
                "edges_removed": 0,
                "actions_removed": 0,
                "quality_scores_removed": 0,
                "review_queue_removed": 0,
                "source_links_removed": 0,
                "events_removed": 0,
            }
        before_capsules = conn.execute(
            f"SELECT COUNT(*) FROM capsules WHERE id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        before_fts = conn.execute(
            f"SELECT COUNT(*) FROM capsules_fts WHERE id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        before_edges = conn.execute(
            f"SELECT COUNT(*) FROM temporal_edges WHERE source_capsule_id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        before_actions = conn.execute(
            f"SELECT COUNT(*) FROM memory_actions WHERE capsule_id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        before_quality = conn.execute(
            f"SELECT COUNT(*) FROM memory_quality_scores WHERE capsule_id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        before_review_queue = conn.execute(
            f"SELECT COUNT(*) FROM memory_review_queue WHERE capsule_id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        before_source_links = conn.execute(
            f"SELECT COUNT(*) FROM capsule_source_events WHERE capsule_id IN ({placeholders})",
            capsule_ids,
        ).fetchone()[0]
        conn.execute(f"DELETE FROM temporal_edges WHERE source_capsule_id IN ({placeholders})", capsule_ids)
        conn.execute(f"DELETE FROM memory_actions WHERE capsule_id IN ({placeholders})", capsule_ids)
        conn.execute(f"DELETE FROM memory_quality_scores WHERE capsule_id IN ({placeholders})", capsule_ids)
        conn.execute(f"DELETE FROM memory_review_queue WHERE capsule_id IN ({placeholders})", capsule_ids)
        conn.execute(f"DELETE FROM capsules_fts WHERE id IN ({placeholders})", capsule_ids)
        status_placeholders = ",".join("?" for _ in PRUNABLE_STATUSES)
        cur = conn.execute(
            f"""
            DELETE FROM capsules
            WHERE id IN ({placeholders})
              AND status IN ({status_placeholders})
            """,
            [*capsule_ids, *PRUNABLE_STATUSES],
        )
        if cur.rowcount != before_capsules:
            raise RuntimeError("Approved cold capsule set changed during deletion.")
        conn.execute("INSERT INTO capsules_fts(capsules_fts) VALUES ('optimize')")
        return {
            "blocked": False,
            "capsules_requested": len(capsule_ids),
            "capsules_removed": before_capsules,
            "fts_removed": before_fts,
            "edges_removed": before_edges,
            "actions_removed": before_actions,
            "quality_scores_removed": before_quality,
            "review_queue_removed": before_review_queue,
            "source_links_removed": before_source_links,
            "events_removed": 0,
        }


def _recall_snapshot(
    store: MemoryStore,
    *,
    queries: list[str],
    scope: str,
    budget: int,
    include_global: bool,
) -> dict[str, dict[str, Any]]:
    compiler = RecallCompiler(store)
    snapshots = {}
    for query in queries:
        result = compiler.recall(query, scope=scope, budget=budget, include_global=include_global)
        snapshots[query] = {
            "estimated_tokens": result.diagnostics["estimated_tokens_after"],
            "selected_capsule_ids": result.diagnostics.get("selected_capsule_ids", []),
            "capsules_selected": result.diagnostics["capsules_selected"],
            "pack_head": result.pack[:400],
        }
    return snapshots


def _compare_recalls(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for query, before_item in before.items():
        after_item = after.get(query, {})
        after_selected = set(after_item.get("selected_capsule_ids", []))
        before_selected = set(before_item.get("selected_capsule_ids", []))
        retained_any = bool(after_selected) or bool(after_item.get("pack_head", "").strip())
        overlap_ratio = _overlap_ratio(before_selected, after_selected)
        token_delta = int(after_item.get("estimated_tokens", 0)) - int(before_item.get("estimated_tokens", 0))
        checks.append(
            {
                "query": query,
                "passed": retained_any and overlap_ratio >= 0.35,
                "before_tokens": before_item.get("estimated_tokens", 0),
                "after_tokens": after_item.get("estimated_tokens", 0),
                "token_delta": token_delta,
                "before_selected_capsules": len(before_selected),
                "after_selected_capsules": len(after_selected),
                "selected_overlap_ratio": overlap_ratio,
            }
        )
    return checks


def _overlap_ratio(before: set[str], after: set[str]) -> float:
    if not before:
        return 1.0 if not after else 0.0
    return len(before.intersection(after)) / len(before)


def _shadow_recommend(passed: bool, deletion: dict[str, Any], plan: PrunePlanReport) -> list[str]:
    recommendations = []
    if not passed:
        recommendations.append("Do not run destructive pruning; shadow prune failed at least one gate.")
    if deletion.get("events_removed", 0) == 0:
        recommendations.append("Shadow prune removed only cold capsules; source events remain preserved.")
    if plan.totals["protected_events"]:
        recommendations.append(
            f"Keep {plan.totals['protected_events']} protected source events in the live store."
        )
    if passed:
        recommendations.append("Shadow prune passed in a temporary restore; live pruning still needs an explicit separate command.")
    return recommendations
