from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RoadmapItem:
    name: str
    status: str
    evidence: str
    next_action: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "next_action": self.next_action,
        }


@dataclass(slots=True)
class GoalRoadmap:
    scope: str
    status: str
    items: list[RoadmapItem]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "status": self.status,
            "items": [item.as_dict() for item in self.items],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [f"# Ara Goal Roadmap: {self.scope}", f"status: {self.status}", "## Requirements"]
        for item in self.items:
            lines.append(f"- [{item.status}] {item.name}: {item.evidence}")
            if item.status != "pass" and item.next_action:
                lines.append(f"  next: {item.next_action}")
        lines.append("## Recommendations")
        lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def build_goal_roadmap(
    memory: Any,
    *,
    scope: str = "global",
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
    repair_hot: bool = False,
) -> GoalRoadmap:
    health = memory.health(
        scope=scope,
        query="goal roadmap memory health",
        regression_cases=regression_cases,
        regression_baseline=regression_baseline,
    )
    purpose = memory.purpose_check(scope=scope, repair_hot=repair_hot)
    identity = memory.identity_check(scope=scope, repair_hot=repair_hot)
    failure_audit = memory.failure_kind_audit(scope=scope, statuses=["candidate", "stable"], dry_run=True)
    self_audit = memory.self_kind_audit(scope=scope, statuses=["candidate", "stable"], dry_run=True)
    milestone = memory.milestone_check(
        scope=scope,
        regression_cases=regression_cases,
        regression_baseline=regression_baseline,
        repair_hot=repair_hot,
    )
    graph_readiness = memory.graph_activation_readiness(scope=scope)
    global_spreading = memory.global_spreading_sandbox(scope=scope)
    privacy_pre_push = memory.privacy_pre_push(repo=Path.cwd())
    reconsolidation = memory.reconsolidation_frame(
        "next natural memory architecture work: purpose identity decision failure forgetting boundary",
        scope=scope,
        budgets=[800, 1600],
        working_budget=900,
        recall_budget=1600,
    )
    reconsolidation_review = memory.review_reconsolidation(scope=scope, limit=10)
    reconsolidation_queue = memory.reconsolidation_review_queue(scope=scope, status="open", limit=10)
    reconsolidation_exceptions = _reconsolidation_exception_summary(memory, scope=scope)
    cold_stewardship = memory.cold_stewardship(scope=scope, group_limit=3, examples_per_group=0)
    cold_map = memory.cold_map(
        scope=scope,
        query="cold memory evidence retention pruning",
        group_limit=3,
        examples_per_group=0,
        budget=700,
    )
    lifecycle = memory.lifecycle(scope=scope, limit=1000, examples_per_tier=0)
    recall_policy = memory.recall_policy(
        "next natural memory architecture work purpose-aware recall controller",
        scope=scope,
        budgets=[800, 1600, 2500],
        include_hot=True,
        cold_budget=700,
        cold_group_limit=3,
    )
    recall_policy_eval = memory.evaluate_recall_policy(scope=scope, include_global=False, min_evaluated=3)
    long_run_stress = memory.long_run_stress(
        scope=scope,
        iterations=2,
        budget=1200,
        include_global=False,
        include_hot=True,
        min_unique_capsules=2,
    )
    long_run_stress_trend = memory.long_run_stress_trend(scope=scope, include_global=False, min_samples=3)
    agency = memory.agency_review(
        prompt="continue building Ara Memory OS natural memory with self-directed judgment",
        scope=scope,
        proposed_action="choose the next implementation step from purpose, identity, recall policy, and working memory",
        budgets=[800, 1600],
        include_hot=True,
        include_health=False,
        record=False,
    )
    worker_schedule = memory.verify_worker_schedule(
        output=memory.store.root / "scripts" / "install-worker-task.ps1",
        scope=scope,
    )
    context = memory.recall_context(
        "current Ara memory architecture, purpose, identity, and next work",
        scope=scope,
        budgets=[800, 1600, 2500],
        include_hot=True,
    )
    stats = memory.stats()

    items = [
        RoadmapItem(
            "local-first durable store",
            "pass" if stats.get("events", 0) > 0 and stats.get("capsules", 0) > 0 else "fail",
            f"events={stats.get('events', 0)}, capsules={stats.get('capsules', 0)}, storage_bytes={stats.get('storage_bytes', 0)}",
            "Initialize and capture memory events." if stats.get("events", 0) == 0 else "",
        ),
        RoadmapItem(
            "bounded recall instead of raw reread",
            "pass" if context.diagnostics["estimated_tokens_after"] <= 1600 else "fail",
            f"budget={context.plan.recommended_budget}, tokens={context.diagnostics['estimated_tokens_after']}, capsules={context.diagnostics['capsules_selected']}",
            "Tune recall-plan budgets or ranking until a small context pack is selected.",
        ),
        RoadmapItem(
            "purpose continuity",
            "pass" if purpose.passed else "fail",
            f"stable_goals={purpose.stable_goals}, hot={purpose.hot_has_goals}, recall={purpose.recall_has_goals}",
            "Capture/promote a stable goal and rebuild hot memory.",
        ),
        RoadmapItem(
            "identity continuity",
            "pass" if identity.passed else "fail",
            f"stable_self={identity.stable_self}, hot={identity.hot_has_identity}, recall={identity.recall_has_identity}",
            "Capture/promote a stable self memory and rebuild hot memory.",
        ),
        RoadmapItem(
            "semantic hygiene",
            "pass" if failure_audit.changed == 0 and self_audit.changed == 0 else "fail",
            f"false_failures={failure_audit.changed}, false_self={self_audit.changed}",
            "Run failure-kind-audit and self-kind-audit with review before applying reclassifications.",
        ),
        RoadmapItem(
            "operational health",
            "pass" if health.passed else "fail",
            f"{health.status} score={health.score}/100",
            "Run health with JSON output and fix failing signals.",
        ),
        RoadmapItem(
            "milestone readiness",
            "pass" if milestone.passed else "fail",
            f"{milestone.status}, checks={len(milestone.checks)}",
            "Run milestone-check and resolve failing gates.",
        ),
        RoadmapItem(
            "graph activation readiness",
            "pass" if graph_readiness.passed else "watch",
            _graph_readiness_evidence(graph_readiness),
            "Consolidate temporal edges that connect lexical seeds to source capsules, then rerun graph activation readiness.",
        ),
        RoadmapItem(
            "global spreading sandbox",
            global_spreading.status,
            _global_spreading_evidence(global_spreading),
            "Tune global fanout, hub suppression, risk filtering, or representative global evidence before broader spreading.",
        ),
        RoadmapItem(
            "privacy pre-push gate",
            "pass" if privacy_pre_push.passed else "fail",
            _privacy_pre_push_evidence(privacy_pre_push),
            "Remove tracked private memory files or secret-like content before pushing the public repository.",
        ),
        RoadmapItem(
            "reconsolidation frame",
            "fail" if reconsolidation.status == "fail" else "watch" if reconsolidation.status == "watch" else "pass",
            _reconsolidation_evidence(reconsolidation),
            "Build a read-only frame that keeps purpose, identity, decisions, failures, and forgetting boundaries visible before applying memory rewrites.",
        ),
        RoadmapItem(
            "reconsolidation apply safety",
            "pass" if reconsolidation_review.passed and reconsolidation_review.pass_count > 0 else "watch",
            _reconsolidation_review_evidence(reconsolidation_review),
            "Run reconsolidation-prepare, reconsolidation-apply, and reconsolidation-review so candidate-only frame snapshots have passing witnesses.",
        ),
        RoadmapItem(
            "reconsolidation rollback review queue",
            _reconsolidation_queue_status(reconsolidation_queue),
            _reconsolidation_queue_evidence(reconsolidation_queue),
            "Run reconsolidation-review --record-queue, inspect open blocker rows, and resolve failed witnesses before stronger live mutation.",
        ),
        RoadmapItem(
            "reconsolidation exception witness gate",
            "pass" if reconsolidation_exceptions["table_ready"] else "fail",
            _reconsolidation_exception_evidence(reconsolidation_exceptions),
            "Create the reconsolidation exception witness table and wire review/rollback gates to exact digest transitions.",
        ),
        RoadmapItem(
            "cold-memory stewardship",
            "watch" if cold_stewardship.status == "watch" else "pass",
            _cold_evidence(health, cold_stewardship),
            "Run cold-stewardship to inspect cold groups and keep retention-cycle evidence current before live pruning.",
        ),
        RoadmapItem(
            "distant-memory navigation",
            _cold_map_status(cold_stewardship, cold_map),
            _cold_map_evidence(cold_map),
            "Run cold-map with a narrower query so distant memory can be inspected without rendering cold bodies.",
        ),
        RoadmapItem(
            "purpose-aware lifecycle policy",
            "fail" if lifecycle.status == "fail" else "watch" if lifecycle.status == "watch" else "pass",
            _lifecycle_evidence(lifecycle),
            "Run lifecycle and promote/review/summarize memories until hot memory is core-only and query recall stays bounded.",
        ),
        RoadmapItem(
            "purpose-aware recall controller",
            "fail" if recall_policy.status == "fail" else "watch" if recall_policy.status == "watch" else "pass",
            _recall_policy_evidence(recall_policy),
            "Run recall-policy and fix intent routing, hot-memory visibility, or cold-map evidence before relying on autonomous recall.",
        ),
        RoadmapItem(
            "recall-policy feedback loop",
            "fail" if recall_policy_eval.status == "fail" else "watch" if recall_policy_eval.status == "watch" else "pass",
            _recall_policy_eval_evidence(recall_policy_eval),
            "Record recall-policy-impact after real turns so recall routing can be audited without reward hacking.",
        ),
        RoadmapItem(
            "long-run stress gate",
            _long_run_stress_status(long_run_stress, long_run_stress_trend),
            _long_run_stress_evidence(long_run_stress, long_run_stress_trend),
            "Review token growth, leakage, critic failures, and harmful impact ratios before tuning recall thresholds.",
        ),
        RoadmapItem(
            "self-directed deliberation",
            "pass" if agency.action_allowed else "watch",
            _agency_evidence(agency),
            "Run agency-review and repair purpose, identity, recall policy, or working-memory evidence before claiming autonomous judgment.",
        ),
        RoadmapItem(
            "scheduled worker script readiness",
            "pass" if worker_schedule.passed else "watch",
            _worker_schedule_evidence(worker_schedule),
            "Run worker-schedule, verify the generated scripts, then separately install and inspect the scheduled task.",
        ),
    ]
    for item in items:
        if item.status == "pass":
            item.next_action = ""
    status = _status(items)
    return GoalRoadmap(scope=scope, status=status, items=items, recommendations=_recommend(items))


def _cold_evidence(health: Any, cold_stewardship: Any) -> str:
    cold = next((signal for signal in health.signals if signal.name == "cold_ratio"), None)
    retention = next((signal for signal in health.signals if signal.name == "retention_cycle"), None)
    parts = []
    if cold:
        parts.append(cold.detail)
    parts.append(
        "cold stewardship "
        f"protected_source_events={cold_stewardship.totals['protected_source_events']}, "
        f"prunable_source_events={cold_stewardship.totals['prunable_source_events']}, "
        f"evidence={cold_stewardship.totals.get('evidence_capsules', 0)}, "
        f"archive={cold_stewardship.totals.get('archive_capsules', 0)}, "
        f"reject={cold_stewardship.totals.get('reject_capsules', 0)}"
    )
    if getattr(cold_stewardship, "active_pins", None):
        top_pin = cold_stewardship.active_pins[0]
        parts.append(
            "top active pin "
            f"{top_pin.status}/{top_pin.kind}/{top_pin.pattern.rstrip(':')} "
            f"source_events={top_pin.source_events}"
        )
    if retention:
        parts.append(retention.detail)
    return "; ".join(parts) if parts else "no cold pressure signal"


def _graph_readiness_evidence(graph_readiness: Any) -> str:
    diagnostics = graph_readiness.diagnostics
    return (
        f"{graph_readiness.status}, used={diagnostics['spreading_activation_used']}, "
        f"edges={diagnostics['graph_activation_edges']}, "
        f"boosted={diagnostics['spreading_activation_boosted_count']}, "
        f"supplemented={diagnostics['spreading_activation_supplemented_count']}, "
        f"visible={diagnostics['visible_capsules']}, "
        f"tokens={diagnostics['estimated_tokens']}, budget={diagnostics['best_budget']}"
    )


def _global_spreading_evidence(global_spreading: Any) -> str:
    diagnostics = global_spreading.diagnostics
    return (
        f"{global_spreading.status}, global={diagnostics['include_global']}, "
        f"quality={diagnostics['quality_score']}, "
        f"edges={diagnostics['graph_activation_edges']}, "
        f"supplemented={diagnostics['spreading_activation_supplemented_count']}, "
        f"depth={diagnostics['max_observed_depth']}, "
        f"visible={diagnostics['visible_capsules']}, "
        f"risk_filtered={diagnostics['capsules_filtered_by_risk']}, "
        f"budget={diagnostics['best_budget']}"
    )


def _privacy_pre_push_evidence(report: Any) -> str:
    failures = sum(1 for finding in report.findings if finding.severity == "fail")
    warnings = sum(1 for finding in report.findings if finding.severity == "warning")
    return (
        f"{report.status}, scanned={report.scanned_files}, "
        f"failures={failures}, warnings={warnings}"
    )


def _reconsolidation_evidence(report: Any) -> str:
    diagnostics = report.diagnostics
    return (
        f"{report.status}, intent={diagnostics['intent']}, "
        f"frames={len(report.frames)}, working_items={diagnostics['working_items']}, "
        f"core={diagnostics['core_capsules']}, guarded={diagnostics['guarded_capsules']}, "
        f"false_failures={diagnostics['false_failure_candidates']}, "
        f"false_self={diagnostics['false_self_candidates']}"
    )


def _reconsolidation_review_evidence(report: Any) -> str:
    return (
        f"reviewed={report.reviewed}, pass={report.pass_count}, "
        f"watch={report.watch_count}, fail={report.fail_count}"
    )


def _reconsolidation_queue_status(report: Any) -> str:
    if not report.open_items:
        return "pass"
    if any(str(item.get("action", "")).startswith("block-strong-reconsolidation") for item in report.open_items):
        return "fail"
    return "watch"


def _reconsolidation_queue_evidence(report: Any) -> str:
    fail_count = sum(1 for item in report.open_items if item.get("review_status") == "fail")
    watch_count = sum(1 for item in report.open_items if item.get("review_status") == "watch")
    blockers = sum(1 for item in report.open_items if str(item.get("action", "")).startswith("block-strong-reconsolidation"))
    return f"open={len(report.open_items)}, blockers={blockers}, fail={fail_count}, watch={watch_count}"


def _reconsolidation_exception_summary(memory: Any, *, scope: str) -> dict[str, Any]:
    with memory.store.session() as conn:
        table = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'reconsolidation_exception_witnesses'
            """
        ).fetchone()
        if table is None:
            return {"table_ready": False, "active": 0, "scope": scope}
        active = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM reconsolidation_exception_witnesses
            WHERE scope = ? AND status = 'active'
            """,
            (scope,),
        ).fetchone()
    return {"table_ready": True, "active": int(active["count"] if active else 0), "scope": scope}


def _reconsolidation_exception_evidence(summary: dict[str, Any]) -> str:
    return (
        f"table_ready={summary['table_ready']}, "
        f"active={summary['active']}, scope={summary['scope']}"
    )


def _cold_map_status(cold_stewardship: Any, cold_map: Any) -> str:
    if cold_stewardship.totals.get("cold_capsules", 0) == 0:
        return "pass"
    if cold_map.totals.get("matched_capsules", 0) > 0 and cold_map.totals.get("within_budget", False):
        return "pass"
    return "watch"


def _cold_map_evidence(cold_map: Any) -> str:
    totals = cold_map.totals
    return (
        f"{cold_map.status}, matched={totals['matched_capsules']}, groups={totals['groups']}, "
        f"map_tokens={totals['estimated_map_tokens']}/{totals['budget_tokens']}, "
        f"raw_tokens={totals['matched_raw_tokens']}, "
        f"reduction={totals['token_reduction_ratio']:.1f}x"
    )


def _worker_schedule_evidence(worker_schedule: Any) -> str:
    details = worker_schedule.details
    if worker_schedule.passed:
        return (
            f"script={worker_schedule.path}, interval={details.get('interval_minutes')}m, "
            f"task={details.get('task_name')}, long_run_stress={details.get('long_run_stress', False)}"
        )
    if worker_schedule.issues:
        return "; ".join(worker_schedule.issues[:3])
    return f"script={worker_schedule.path}"


def _lifecycle_evidence(lifecycle: Any) -> str:
    totals = lifecycle.totals
    policy = lifecycle.token_policy
    return (
        f"{lifecycle.status}, core={totals['core_capsules']}, working={totals['working_capsules']}, "
        f"guarded={totals['guarded_capsules']}, archive={totals['archive_capsules']}, "
        f"core_tokens={policy['core_tokens']}/{policy['target_hot_tokens']}, "
        f"raw_to_core_reduction={policy['raw_to_core_reduction']:.1f}x"
    )


def _recall_policy_evidence(recall_policy: Any) -> str:
    plan = recall_policy.recall_plan
    tokens = recall_policy.token_policy
    return (
        f"{recall_policy.status}, intent={recall_policy.intent}, "
        f"strategy={recall_policy.strategy}, "
        f"budget={plan.recommended_budget}, tokens={plan.estimated_tokens}, "
        f"visible={plan.diagnostics.get('visible_capsules', 0)}, "
        f"cold_avoided={tokens.get('avoided_cold_raw_tokens', 0)}"
    )


def _recall_policy_eval_evidence(recall_policy_eval: Any) -> str:
    totals = recall_policy_eval.totals
    return (
        f"{recall_policy_eval.status}, impacts={totals['impacts']}, "
        f"evaluated={totals['evaluated']}, helpful={totals['helpful']}, "
        f"harmful={totals['harmful']}, unknown={totals['unknown']}"
    )


def _long_run_stress_status(report: Any, trend: Any) -> str:
    if report.status == "fail" or trend.status == "fail":
        return "fail"
    if report.status == "watch" or trend.status == "watch":
        return "watch"
    return "pass"


def _long_run_stress_evidence(report: Any, trend: Any) -> str:
    diagnostics = report.diagnostics
    trend_totals = trend.totals
    return (
        f"{report.status}, score={report.score}/100, runs={diagnostics['runs']}, "
        f"token_growth={diagnostics['token_growth']:.3f}, "
        f"unique_capsules={diagnostics['unique_capsules']}, "
        f"leaks={diagnostics['forbidden_leaks']}, "
        f"critic_failures={diagnostics['critic_failures']}, "
        f"harmful_impact_ratio={diagnostics['harmful_impact_ratio']:.3f}; "
        f"trend={trend.status}, samples={trend_totals['runs']}, "
        f"score_delta={trend.score_delta}, token_growth_delta={trend.token_growth_delta}"
    )


def _agency_evidence(agency: Any) -> str:
    policy = agency.evidence.get("recall_policy", {})
    working = agency.evidence.get("working_memory", {})
    return (
        f"stance={agency.stance}, reasons={len(agency.reasons)}, "
        f"action_allowed={agency.action_allowed}, "
        f"intent={policy.get('intent')}, actions={len(policy.get('actions', []))}, "
        f"working_items={working.get('items', 0)}, control_tokens={agency.diagnostics.get('control_tokens', 0)}"
    )


def _status(items: list[RoadmapItem]) -> str:
    if any(item.status == "fail" for item in items):
        return "fail"
    if any(item.status == "watch" for item in items):
        return "watch"
    return "pass"


def _recommend(items: list[RoadmapItem]) -> list[str]:
    active = [item for item in items if item.status != "pass" and item.next_action]
    if not active:
        return ["Goal roadmap gates are green; continue with deeper capability work and verified backups."]
    return [f"{item.name}: {item.next_action}" for item in active]
