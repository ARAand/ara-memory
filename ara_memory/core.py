from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ara_memory.advisor import MemoryAdvisor, MemoryReviewEngine, build_memory_advisor
from ara_memory.archive_crypto import migrate_archive_objects
from ara_memory.audit import MemoryAuditor
from ara_memory.backup import BackupResult, create_backup, drill_restore_backup, restore_backup, verify_backup
from ara_memory.backup_stewardship import BackupStewardshipReport, run_backup_stewardship
from ara_memory.cold_export import ColdExportResult, export_cold_capsules, verify_cold_export
from ara_memory.curator import MemoryCurator
from ara_memory.doctor import DoctorReport, MemoryDoctor
from ara_memory.hot import HotState, HotStateBuilder
from ara_memory.maintenance import MaintenanceReport, run_maintenance
from ara_memory.memory_lifecycle import MemoryLifecycleReport
from ara_memory.models import Event, EventKind, MemoryStatus
from ara_memory.mutation_preflight import MutationPreflight
from ara_memory.privacy import guard_event_payload, privacy_safe_label
from ara_memory.prune import (
    LivePruneApproval,
    LivePruneController,
    LivePruneReport,
    PrunePlanner,
    PrunePlanReport,
    ShadowPruner,
    ShadowPruneReport,
)
from ara_memory.quality import QualityReport, QualityScorer, ReviewWorkerReport
from ara_memory.recall import RecallCandidateResult, RecallCompiler, RecallResult
from ara_memory.retention import RetentionAnalyzer, RetentionReport
from ara_memory.retention_cycle import RetentionCycleReport, RetentionCycleRunner
from ara_memory.promotion import can_promote_capsule
from ara_memory.risk import MemoryRiskAssessor
from ara_memory.sleep import SleepConsolidator, SleepReport
from ara_memory.storage import MemoryStore, row_to_capsule


class AraMemory:
    def __init__(self, root: Path | None = None) -> None:
        root = root or Path(os.environ.get("ARA_MEMORY_HOME", ".ara-memory"))
        self.store = MemoryStore(root)
        self.advisor: MemoryAdvisor | None = None

    def init(self) -> None:
        self.store.init()

    def retain(
        self,
        *,
        kind: str,
        text: str,
        source: str = "manual",
        scope: str = "global",
        metadata: dict[str, Any] | None = None,
        allow_raw_private: bool = False,
    ) -> Event:
        guarded = guard_event_payload(
            text=text,
            metadata=metadata,
            source=source,
            scope=scope,
            allow_raw_private=allow_raw_private,
        )
        event_metadata = dict(guarded.metadata)
        if guarded.reasons or guarded.redacted or guarded.allow_raw_private:
            event_metadata["privacy"] = guarded.metadata_note()
        event = Event.create(
            kind=EventKind(kind),
            text=guarded.text,
            source=guarded.source or source,
            scope=guarded.scope or scope,
            metadata=event_metadata,
        )
        return self.store.append_event(event)

    def consolidate(self, *, limit: int = 100) -> int:
        return len(MemoryCurator(self.store).consolidate(limit=limit))

    def recall(
        self,
        query: str,
        *,
        scope: str = "global",
        budget: int = 4000,
        include_global: bool = True,
        include_hot: bool = False,
    ) -> str:
        return self.recall_result(
            query,
            scope=scope,
            budget=budget,
            include_global=include_global,
            include_hot=include_hot,
        ).pack

    def recall_result(
        self,
        query: str,
        *,
        scope: str = "global",
        budget: int = 4000,
        include_global: bool = True,
        include_hot: bool = False,
    ) -> RecallResult:
        scope = _canonical_scope(scope) or "global"
        hot = self.read_hot(scope=scope) if include_hot else None
        return RecallCompiler(self.store).recall(
            query,
            scope=scope,
            budget=budget,
            include_global=include_global,
            hot_state=hot.text if hot else None,
        )

    def recall_candidates(
        self,
        query: str,
        *,
        scope: str = "global",
        budget: int = 4000,
        include_global: bool = True,
        include_hot: bool = False,
        candidate_limit: int = 18,
    ) -> RecallCandidateResult:
        scope = _canonical_scope(scope) or "global"
        hot = self.read_hot(scope=scope) if include_hot else None
        return RecallCompiler(self.store).recall_candidates(
            query,
            scope=scope,
            budget=budget,
            include_global=include_global,
            hot_state=hot.text if hot else None,
            candidate_limit=candidate_limit,
        )

    def audit(self) -> str:
        return MemoryAuditor(self.store).audit()

    def stats(self) -> dict[str, Any]:
        stats = self.store.stats()
        storage = self.store.storage_breakdown()
        stats.update(storage)
        stats["storage_breakdown"] = dict(storage)
        return stats

    def promote(self, capsule_id: str, *, actor: str = "manual", reason: str = "") -> bool:
        row = self.store.get_capsule(capsule_id)
        if row is None:
            return False
        cap = row_to_capsule(row)
        gate = can_promote_capsule(self.store, cap, require_provenance=False)
        if gate.should_quarantine:
            self.store.update_capsule_status(
                capsule_id,
                MemoryStatus.QUARANTINED,
                actor="memory-auditor",
                reason="blocked unsafe manual promotion",
            )
            return False
        if not gate.allowed:
            return False
        if cap["status"] == MemoryStatus.CANDIDATE.value:
            return self.store.update_capsule_status_if_current(
                capsule_id,
                MemoryStatus.CANDIDATE,
                MemoryStatus.STABLE,
                actor=actor,
                reason=reason,
            )
        return self.store.update_capsule_status(capsule_id, MemoryStatus.STABLE, actor=actor, reason=reason)

    def reject(self, capsule_id: str, *, actor: str = "manual", reason: str = "") -> bool:
        return self.store.update_capsule_status(capsule_id, MemoryStatus.REJECTED, actor=actor, reason=reason)

    def quarantine(self, capsule_id: str, *, actor: str = "manual", reason: str = "") -> bool:
        return self.store.update_capsule_status(capsule_id, MemoryStatus.QUARANTINED, actor=actor, reason=reason)

    def list_capsules(
        self,
        *,
        scope: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        status_value = MemoryStatus(status) if status else None
        scope = _canonical_scope(scope)
        return [
            row_to_capsule(row)
            for row in self.store.list_capsules(scope=scope, status=status_value, kind=kind, limit=limit)
        ]

    def list_actions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(row) for row in self.store.list_actions(limit=limit)]

    def review_witnesses(
        self,
        *,
        scope: str | None = None,
        action: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        scope = _canonical_scope(scope)
        return [dict(row) for row in self.store.list_review_witnesses(scope=scope, action=action, limit=limit)]

    def prepare_review_rollback(self, witness_id: str, *, ttl_minutes: int = 30) -> dict[str, Any]:
        return self.store.prepare_review_rollback_approval(witness_id, ttl_minutes=ttl_minutes)

    def live_review_rollback(self, approval_token: str, *, confirm: str) -> dict[str, Any]:
        return self.store.live_review_rollback(approval_token, confirm=confirm)

    def mutation_preflight(
        self,
        *,
        capsule_id: str,
        action: str,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        return MutationPreflight(self.store).run(
            capsule_id=capsule_id,
            action=action,
            title=title,
            body=body,
            tags=tags,
        )

    def prepare_mutation(
        self,
        *,
        capsule_id: str,
        action: str,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
        ttl_minutes: int = 30,
    ) -> dict[str, Any]:
        preflight = self.mutation_preflight(
            capsule_id=capsule_id,
            action=action,
            title=title,
            body=body,
            tags=tags,
        )
        return self.store.prepare_mutation_approval(preflight.as_dict(), ttl_minutes=ttl_minutes)

    def live_mutation_apply(self, approval_token: str, *, confirm: str) -> dict[str, Any]:
        return self.store.live_mutation_apply(approval_token, confirm=confirm)

    def prepare_mutation_rollback(self, witness_id: str, *, ttl_minutes: int = 30) -> dict[str, Any]:
        return self.store.prepare_mutation_rollback_approval(witness_id, ttl_minutes=ttl_minutes)

    def live_mutation_rollback(self, approval_token: str, *, confirm: str) -> dict[str, Any]:
        return self.store.live_mutation_rollback(approval_token, confirm=confirm)

    def sleep(self, *, scope: str = "global", dry_run: bool = False) -> SleepReport:
        scope = _canonical_scope(scope) or "global"
        return SleepConsolidator(self.store, advisor=self._advisor()).run(scope=scope, dry_run=dry_run)

    def list_sleep_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(row) for row in self.store.list_consolidation_runs(limit=limit)]

    def risk_report(self, *, scope: str = "global", limit: int = 200) -> list[dict[str, Any]]:
        scope = _canonical_scope(scope) or "global"
        return [verdict.as_dict() for verdict in MemoryRiskAssessor(self.store).assess_scope(scope=scope, limit=limit)]

    def review(self, *, scope: str = "global", limit: int = 500) -> list[dict[str, Any]]:
        scope = _canonical_scope(scope) or "global"
        return [
            rec.as_dict()
            for rec in MemoryReviewEngine(self.store, advisor=self._advisor()).review_scope(scope=scope, limit=limit)
        ]

    def quality(self, *, scope: str | None = None, limit: int = 500, persist: bool = False) -> QualityReport:
        scope = _canonical_scope(scope)
        return QualityScorer(self.store).run(scope=scope, limit=limit, persist=persist)

    def promotion_candidates(self, *, scope: str | None = None, limit: int = 500, threshold: float | None = None) -> Any:
        scope = _canonical_scope(scope)
        kwargs: dict[str, Any] = {"scope": scope, "limit": limit}
        if threshold is not None:
            kwargs["threshold"] = threshold
        return QualityScorer(self.store).promotion_candidates(**kwargs)

    def review_queue(self, *, scope: str | None = None, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        scope = _canonical_scope(scope)
        return QualityScorer(self.store).list_queue(scope=scope, status=status, limit=limit)

    def resolve_review(self, queue_id: str) -> bool:
        return QualityScorer(self.store).resolve_queue_item(queue_id)

    def review_triage(
        self,
        *,
        scope: str | None = None,
        status: str = "open",
        limit: int = 500,
        examples_per_group: int = 3,
    ) -> Any:
        scope = _canonical_scope(scope)
        return QualityScorer(self.store).triage_queue(
            scope=scope,
            status=status,
            limit=limit,
            examples_per_group=examples_per_group,
        )

    def review_worker(self, *, scope: str | None = None, limit: int = 25, dry_run: bool = True) -> ReviewWorkerReport:
        scope = _canonical_scope(scope)
        return QualityScorer(self.store).work_queue(scope=scope, limit=limit, dry_run=dry_run)

    def review_compact(self, *, scope: str | None = None, limit: int = 250, dry_run: bool = True) -> Any:
        scope = _canonical_scope(scope)
        return QualityScorer(self.store).compact_queue(scope=scope, limit=limit, dry_run=dry_run)

    def review_redact(self, *, scope: str | None = None, limit: int = 50, dry_run: bool = True) -> Any:
        scope = _canonical_scope(scope)
        return QualityScorer(self.store).redact_sensitive_reviews(scope=scope, limit=limit, dry_run=dry_run)

    def build_hot(self, *, scope: str = "global", budget: int = 1200) -> HotState:
        scope = _canonical_scope(scope) or "global"
        return HotStateBuilder(self.store).build(scope=scope, budget=budget)

    def read_hot(self, *, scope: str = "global") -> HotState | None:
        scope = _canonical_scope(scope) or "global"
        return HotStateBuilder(self.store).read(scope=scope)

    def evaluate(self) -> Any:
        from ara_memory.evaluation import run_builtin_evaluation

        return run_builtin_evaluation()

    def contextual_evaluate(self) -> Any:
        from ara_memory.evaluation import run_contextual_evaluation

        return run_contextual_evaluation()

    def spool_turn(self, turn: dict[str, Any], **kwargs: Any) -> Any:
        from ara_memory.spool import enqueue_turn

        return enqueue_turn(self, turn, **kwargs)

    def execute_turn_ingress(self, turn: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        from ara_memory.turn import execute_turn_ingress

        return execute_turn_ingress(self, turn, **kwargs)

    def drain_spool(self, **kwargs: Any) -> Any:
        from ara_memory.spool import drain_spool

        return drain_spool(self, **kwargs)

    def spool_stats(self) -> dict[str, int]:
        from ara_memory.spool import spool_stats

        return spool_stats(self)

    def worker(self, **kwargs: Any) -> Any:
        from ara_memory.worker import run_memory_worker

        return run_memory_worker(self, **kwargs)

    def worker_loop(self, **kwargs: Any) -> Any:
        from ara_memory.worker import run_worker_loop

        return run_worker_loop(self, **kwargs)

    def write_worker_schedule(self, **kwargs: Any) -> Any:
        from ara_memory.schedule import write_windows_worker_task_script

        return write_windows_worker_task_script(**kwargs)

    def verify_worker_schedule(self, **kwargs: Any) -> Any:
        from ara_memory.schedule import verify_windows_worker_task_script

        return verify_windows_worker_task_script(**kwargs)

    def recall_regression(
        self,
        cases: list[Any],
        *,
        baseline: dict[str, Any] | None = None,
        max_token_growth: float = 0.25,
        min_overlap: float = 0.35,
    ) -> Any:
        from ara_memory.regression import run_recall_regression

        return run_recall_regression(
            self,
            cases,
            baseline=baseline,
            max_token_growth=max_token_growth,
            min_overlap=min_overlap,
        )

    def recall_plan(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.recall_plan import build_recall_plan

        return build_recall_plan(self, query, **kwargs)

    def recall_context(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.recall_plan import build_recall_context

        return build_recall_context(self, query, **kwargs)

    def recall_policy(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.recall_policy import build_recall_policy

        return build_recall_policy(self, query, **kwargs)

    def reconsolidation_frame(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import build_reconsolidation_frame

        return build_reconsolidation_frame(self, query, **kwargs)

    def prepare_reconsolidation(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import prepare_reconsolidation_approval

        return prepare_reconsolidation_approval(self, query, **kwargs)

    def apply_reconsolidation(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import apply_reconsolidation_approval

        return apply_reconsolidation_approval(self, **kwargs)

    def review_reconsolidation(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import review_reconsolidation_witnesses

        return review_reconsolidation_witnesses(self, **kwargs)

    def record_reconsolidation_review_queue(self, report: Any, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import record_reconsolidation_review_queue

        return record_reconsolidation_review_queue(self, report, **kwargs)

    def reconsolidation_review_queue(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import list_reconsolidation_review_queue

        return list_reconsolidation_review_queue(self, **kwargs)

    def strong_reconsolidation_preflight(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import strong_reconsolidation_preflight

        return strong_reconsolidation_preflight(self, query, **kwargs)

    def prepare_live_reconsolidation_action(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import prepare_live_reconsolidation_action

        return prepare_live_reconsolidation_action(self, query, **kwargs)

    def live_reconsolidation_cool(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import live_reconsolidation_cool

        return live_reconsolidation_cool(self, **kwargs)

    def review_reconsolidation_action_witnesses(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import review_reconsolidation_action_witnesses

        return review_reconsolidation_action_witnesses(self, **kwargs)

    def shadow_reconsolidation_action_rollback(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import shadow_reconsolidation_action_rollback

        return shadow_reconsolidation_action_rollback(self, **kwargs)

    def prepare_live_reconsolidation_action_rollback(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import prepare_live_reconsolidation_action_rollback

        return prepare_live_reconsolidation_action_rollback(self, **kwargs)

    def review_reconsolidation_action_rollback_approvals(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import review_reconsolidation_action_rollback_approvals

        return review_reconsolidation_action_rollback_approvals(self, **kwargs)

    def live_reconsolidation_action_rollback(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import live_reconsolidation_action_rollback

        return live_reconsolidation_action_rollback(self, **kwargs)

    def shadow_reconsolidation_rollback(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import shadow_reconsolidation_rollback

        return shadow_reconsolidation_rollback(self, **kwargs)

    def prepare_live_reconsolidation_rollback(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import prepare_live_reconsolidation_rollback

        return prepare_live_reconsolidation_rollback(self, **kwargs)

    def live_reconsolidation_rollback(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import live_reconsolidation_rollback

        return live_reconsolidation_rollback(self, **kwargs)

    def record_reconsolidation_exception_witness(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import record_reconsolidation_exception_witness

        return record_reconsolidation_exception_witness(self, **kwargs)

    def list_reconsolidation_exception_witnesses(self, **kwargs: Any) -> Any:
        from ara_memory.reconsolidation import list_reconsolidation_exception_witnesses

        return list_reconsolidation_exception_witnesses(self, **kwargs)

    def record_recall_policy_impact(self, **kwargs: Any) -> Event:
        from ara_memory.recall_policy import record_recall_policy_impact

        return record_recall_policy_impact(self, **kwargs)

    def evaluate_recall_policy(self, **kwargs: Any) -> Any:
        from ara_memory.recall_policy import evaluate_recall_policy

        return evaluate_recall_policy(self, **kwargs)

    def working_memory(self, **kwargs: Any) -> Any:
        from ara_memory.working_memory import build_working_memory

        return build_working_memory(self, **kwargs)

    def record_memory_impact(self, **kwargs: Any) -> Any:
        from ara_memory.working_memory import record_memory_impact

        return record_memory_impact(self, **kwargs)

    def evaluate_memory_impact(self, **kwargs: Any) -> Any:
        from ara_memory.working_memory import evaluate_memory_impact

        return evaluate_memory_impact(self, **kwargs)

    def recall_quality(self, query: str, **kwargs: Any) -> Any:
        from ara_memory.recall_quality import build_recall_quality_gate

        return build_recall_quality_gate(self, query, **kwargs)

    def govern_turn(self, turn: dict[str, Any], **kwargs: Any) -> Any:
        from ara_memory.governor import govern_turn

        return govern_turn(self, turn, **kwargs)

    def agency_review(self, **kwargs: Any) -> Any:
        from ara_memory.agency import build_agency_review

        return build_agency_review(self, **kwargs)

    def purpose_check(self, **kwargs: Any) -> Any:
        from ara_memory.purpose import run_purpose_check

        return run_purpose_check(self, **kwargs)

    def identity_check(self, **kwargs: Any) -> Any:
        from ara_memory.identity import run_identity_check

        return run_identity_check(self, **kwargs)

    def failure_kind_audit(self, **kwargs: Any) -> Any:
        from ara_memory.failure_audit import audit_failure_kinds

        return audit_failure_kinds(self.store, **kwargs)

    def self_kind_audit(self, **kwargs: Any) -> Any:
        from ara_memory.self_audit import audit_self_kinds

        return audit_self_kinds(self.store, **kwargs)

    def milestone_check(self, **kwargs: Any) -> Any:
        from ara_memory.milestone import run_milestone_check

        return run_milestone_check(self, **kwargs)

    def goal_roadmap(self, **kwargs: Any) -> Any:
        from ara_memory.goal_roadmap import build_goal_roadmap

        return build_goal_roadmap(self, **kwargs)

    def graph_activation_readiness(self, **kwargs: Any) -> Any:
        from ara_memory.graph_readiness import run_graph_activation_readiness

        return run_graph_activation_readiness(self, **kwargs)

    def global_spreading_sandbox(self, **kwargs: Any) -> Any:
        from ara_memory.global_spreading import run_global_spreading_sandbox

        return run_global_spreading_sandbox(self, **kwargs)

    def privacy_pre_push(self, **kwargs: Any) -> Any:
        from ara_memory.privacy_push import run_privacy_pre_push_gate

        return run_privacy_pre_push_gate(**kwargs)

    def doctor(
        self,
        *,
        scope: str = "global",
        recall_query: str = "current memory state",
        recall_budget: int = 1600,
        hot_budget: int = 1200,
        include_global: bool = True,
    ) -> DoctorReport:
        scope = _canonical_scope(scope) or "global"
        return MemoryDoctor(self.store).run(
            scope=scope,
            recall_query=recall_query,
            recall_budget=recall_budget,
            hot_budget=hot_budget,
            include_global=include_global,
        )

    def health(self, **kwargs: Any) -> Any:
        from ara_memory.health import run_health_check

        return run_health_check(self, **kwargs)

    def backup(
        self,
        *,
        output: Path | None = None,
        include_archive: bool = True,
        archive_mode: str | None = None,
    ) -> BackupResult:
        return create_backup(
            self.store,
            output=output,
            include_archive=include_archive,
            archive_mode=archive_mode,
        )

    def verify_backup(self, path: Path, *, trust_root: Path | None = None) -> dict[str, Any]:
        return verify_backup(path, trust_root=trust_root or self.store.root)

    def backup_stewardship(self, **kwargs: Any) -> BackupStewardshipReport:
        return run_backup_stewardship(self.store, **kwargs)

    def restore_backup(
        self,
        path: Path,
        target_root: Path,
        *,
        force: bool = False,
        trust_root: Path | None = None,
    ) -> dict[str, Any]:
        return restore_backup(path, target_root, force=force, trust_root=trust_root or self.store.root)

    def restore_drill(
        self,
        path: Path,
        *,
        scope: str = "global",
        recall_query: str | None = None,
        recall_budget: int = 1200,
        trust_root: Path | None = None,
    ) -> dict[str, Any]:
        scope = _canonical_scope(scope) or "global"
        return drill_restore_backup(
            path,
            scope=scope,
            recall_query=recall_query,
            recall_budget=recall_budget,
            trust_root=trust_root or self.store.root,
        )

    def maintenance(self, *, vacuum: bool = True) -> MaintenanceReport:
        return run_maintenance(self.store, vacuum=vacuum)

    def archive_encrypt(self, *, apply: bool = False, limit: int | None = None) -> dict[str, Any]:
        return migrate_archive_objects(self.store.root, apply=apply, limit=limit)

    def retention(self, *, scope: str | None = None, cold_limit: int = 20) -> RetentionReport:
        scope = _canonical_scope(scope)
        return RetentionAnalyzer(self.store).run(scope=scope, cold_limit=cold_limit)

    def cold_stewardship(self, **kwargs: Any) -> Any:
        from ara_memory.cold_stewardship import ColdStewardshipAnalyzer

        return ColdStewardshipAnalyzer(self.store).run(**kwargs)

    def cold_map(self, **kwargs: Any) -> Any:
        from ara_memory.cold_map import ColdMemoryMapper

        kwargs = dict(kwargs)
        kwargs["scope"] = _canonical_scope(kwargs.get("scope"))
        return ColdMemoryMapper(self.store).run(**kwargs)

    def provenance_compaction(self, **kwargs: Any) -> Any:
        from ara_memory.provenance_compaction import ProvenanceCompactor

        kwargs = dict(kwargs)
        kwargs["scope"] = _canonical_scope(kwargs.get("scope"))
        return ProvenanceCompactor(self.store).run(**kwargs)

    def lifecycle(self, **kwargs: Any) -> MemoryLifecycleReport:
        from ara_memory.memory_lifecycle import MemoryLifecycleAnalyzer

        return MemoryLifecycleAnalyzer(self.store).run(**kwargs)

    def retention_cycle(self, **kwargs: Any) -> RetentionCycleReport:
        return RetentionCycleRunner(self.store).run(**kwargs)

    def candidate_pressure(self, *, scope: str | None = None, limit: int = 20) -> Any:
        from ara_memory.candidate_pressure import CandidatePressureAnalyzer

        scope = _canonical_scope(scope)
        return CandidatePressureAnalyzer(self.store).run(scope=scope, limit=limit)

    def candidate_summary(self, **kwargs: Any) -> Any:
        from ara_memory.candidate_summary import CandidateSummaryConsolidator

        return CandidateSummaryConsolidator(self.store).run(**kwargs)

    def conflict_adjudicate(self, **kwargs: Any) -> Any:
        from ara_memory.conflict_adjudication import ConflictAdjudicator

        return ConflictAdjudicator(self.store).run(**kwargs)

    def episode_summary(self, **kwargs: Any) -> Any:
        from ara_memory.episode_summary import EpisodeSummaryConsolidator

        return EpisodeSummaryConsolidator(self.store).run(**kwargs)

    def cold_export(
        self,
        *,
        output: Path | None = None,
        scope: str | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
        include_events: bool = True,
        order: str = "newest",
    ) -> ColdExportResult:
        status_values = [MemoryStatus(status) for status in statuses] if statuses else None
        scope = _canonical_scope(scope)
        return export_cold_capsules(
            self.store,
            output=output,
            scope=scope,
            statuses=status_values,
            limit=limit,
            include_events=include_events,
            order=order,
        )

    def verify_cold_export(self, path: Path, *, trust_root: Path | None = None) -> dict[str, Any]:
        return verify_cold_export(path, trust_root=trust_root or self.store.root)

    def prune_plan(
        self,
        *,
        scope: str | None = None,
        limit: int | None = None,
        export_path: Path | None = None,
        recall_queries: list[str] | None = None,
        recall_budget: int = 1200,
        include_global: bool = True,
    ) -> PrunePlanReport:
        scope = _canonical_scope(scope)
        return PrunePlanner(self.store).run(
            scope=scope,
            limit=limit,
            export_path=export_path,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
        )

    def shadow_prune(
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
        scope = _canonical_scope(scope)
        return ShadowPruner(self.store).run(
            backup_path=backup_path,
            export_path=export_path,
            scope=scope,
            limit=limit,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
            doctor_query=doctor_query,
        )

    def prepare_live_prune(
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
        scope = _canonical_scope(scope)
        return LivePruneController(self.store).prepare(
            backup_path=backup_path,
            export_path=export_path,
            scope=scope,
            limit=limit,
            recall_queries=recall_queries,
            recall_budget=recall_budget,
            include_global=include_global,
            doctor_query=doctor_query,
            ttl_minutes=ttl_minutes,
        )

    def live_prune(self, *, approval_token: str, confirmation: str) -> LivePruneReport:
        return LivePruneController(self.store).run(approval_token=approval_token, confirmation=confirmation)

    def irreversible_operations(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return LivePruneController(self.store).list_operations(limit=limit)

    def _advisor(self) -> MemoryAdvisor:
        if self.advisor is None:
            self.advisor = build_memory_advisor(self.store)
        return self.advisor


def _canonical_scope(scope: str | None) -> str | None:
    if scope is None:
        return None
    return privacy_safe_label(str(scope), label="scope")
