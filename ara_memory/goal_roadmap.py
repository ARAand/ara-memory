from __future__ import annotations

from dataclasses import dataclass
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
    cold_stewardship = memory.cold_stewardship(scope=scope, group_limit=3, examples_per_group=0)
    lifecycle = memory.lifecycle(scope=scope, limit=1000, examples_per_tier=0)
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
            "cold-memory stewardship",
            "watch" if cold_stewardship.status == "watch" else "pass",
            _cold_evidence(health, cold_stewardship),
            "Run cold-stewardship to inspect cold groups and keep retention-cycle evidence current before live pruning.",
        ),
        RoadmapItem(
            "purpose-aware lifecycle policy",
            "fail" if lifecycle.status == "fail" else "watch" if lifecycle.status == "watch" else "pass",
            _lifecycle_evidence(lifecycle),
            "Run lifecycle and promote/review/summarize memories until hot memory is core-only and query recall stays bounded.",
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
        f"prunable_source_events={cold_stewardship.totals['prunable_source_events']}"
    )
    if retention:
        parts.append(retention.detail)
    return "; ".join(parts) if parts else "no cold pressure signal"


def _worker_schedule_evidence(worker_schedule: Any) -> str:
    details = worker_schedule.details
    if worker_schedule.passed:
        return (
            f"script={worker_schedule.path}, interval={details.get('interval_minutes')}m, "
            f"task={details.get('task_name')}"
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
