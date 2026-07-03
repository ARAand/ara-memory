from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.compressors import compact_text, extract_keywords


@dataclass(slots=True)
class ProjectedCapsuleFeedback:
    capsule_id: str
    total: int
    evaluated: int
    helpful: int
    harmful: int
    unknown: int
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "total": self.total,
            "evaluated": self.evaluated,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "unknown": self.unknown,
            "verdict": self.verdict,
        }


@dataclass(slots=True)
class RecallQualityReport:
    query: str
    scope: str
    status: str
    policy_summary: dict[str, Any]
    critic_summary: dict[str, Any]
    critic_impact_summary: dict[str, Any]
    working_summary: dict[str, Any]
    policy_eval_summary: dict[str, Any]
    memory_impact_summary: dict[str, Any]
    stress_trend_summary: dict[str, Any]
    projected_feedback: list[ProjectedCapsuleFeedback]
    diagnostics: dict[str, Any]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "status": self.status,
            "policy": self.policy_summary,
            "critic": self.critic_summary,
            "critic_impact": self.critic_impact_summary,
            "working_memory": self.working_summary,
            "policy_eval": self.policy_eval_summary,
            "memory_impact": self.memory_impact_summary,
            "long_run_stress_trend": self.stress_trend_summary,
            "projected_feedback": [item.as_dict() for item in self.projected_feedback],
            "diagnostics": self.diagnostics,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Recall Quality Gate: {self.scope}",
            f"status: {self.status}",
            f"query: {compact_text(self.query, limit=260)}",
            "## Policy",
            (
                "- "
                f"intent={self.policy_summary['intent']}, "
                f"status={self.policy_summary['status']}, "
                f"strategy={self.policy_summary['strategy']}"
            ),
            "## Recall Critic",
            (
                "- "
                f"status={self.critic_summary['status']}, "
                f"score={self.critic_summary['score']}, "
                f"visible={self.critic_summary['diagnostics'].get('visible_capsules', 0)}, "
                f"terms={self.critic_summary['diagnostics'].get('query_terms_visible_count', 0)}/"
                f"{self.critic_summary['diagnostics'].get('query_term_count', 0)}"
            ),
            (
                "- impact_eval: "
                f"status={self.critic_impact_summary['status']}, "
                f"evaluated={self.critic_impact_summary['totals'].get('evaluated', 0)}, "
                f"harmful={self.critic_impact_summary['totals'].get('harmful', 0)}"
            ),
            "## Long-Run Stress Trend",
            (
                "- "
                f"status={self.stress_trend_summary['status']}, "
                f"runs={self.stress_trend_summary['totals'].get('runs', 0)}, "
                f"latest_score={self.stress_trend_summary.get('latest', {}).get('score', 'n/a')}"
            ),
            "## Working Memory",
            (
                "- "
                f"projected={len(self.working_summary['projected_capsule_ids'])}, "
                f"items={self.working_summary['items']}, "
                f"tokens={self.working_summary['estimated_tokens']}"
            ),
            "## Projected Capsule Feedback",
        ]
        if not self.projected_feedback:
            lines.append("- No projected capsule has reviewed impact feedback yet.")
        for item in self.projected_feedback[:8]:
            lines.append(
                "- "
                f"{item.capsule_id}: verdict={item.verdict}, evaluated={item.evaluated}, "
                f"helpful={item.helpful}, harmful={item.harmful}, unknown={item.unknown}"
            )
        lines.append("## Recommendations")
        lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class RecallQualityImpactPlan:
    query: str
    scope: str
    status: str
    helped: bool | None
    outcome: str
    dry_run: bool
    quality_summary: dict[str, Any]
    critic_impact: dict[str, Any]
    working_memory_impact: dict[str, Any]
    events: list[dict[str, Any]]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "status": self.status,
            "passed": self.passed,
            "helped": self.helped,
            "outcome": self.outcome,
            "dry_run": self.dry_run,
            "quality": self.quality_summary,
            "critic_impact": self.critic_impact,
            "working_memory_impact": self.working_memory_impact,
            "events": self.events,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Recall Quality Impact Plan: {self.scope}",
            f"status: {self.status}",
            f"dry_run: {self.dry_run}",
            f"helped: {self.helped if self.helped is not None else 'unknown'}",
            f"query: {compact_text(self.query, limit=260)}",
            "## Planned Records",
            (
                "- recall-critic-impact: "
                f"critic_status={self.critic_impact['critic_status']}, "
                f"decisions={', '.join(self.critic_impact['decisions']) or 'none'}"
            ),
            (
                "- working-memory-impact: "
                f"capsules={len(self.working_memory_impact['capsule_ids'])}"
            ),
            "## Recommendations",
        ]
        lines.extend(f"- {item}" for item in self.recommendations)
        if self.events:
            lines.append("## Events")
            lines.extend(f"- {event['source']}: {event['event_id']}" for event in self.events)
        return "\n".join(lines)


def build_recall_quality_gate(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budgets: list[int] | None = None,
    include_global: bool = True,
    include_hot: bool = True,
    working_budget: int = 900,
    recall_budget: int = 1600,
    limit: int = 500,
    min_evaluated: int = 3,
    stress_min_samples: int = 3,
) -> RecallQualityReport:
    policy = memory.recall_policy(
        query,
        scope=scope,
        budgets=budgets or [800, 1600, 2500],
        include_global=include_global,
        include_hot=include_hot,
    )
    working = memory.working_memory(
        prompt=query,
        scope=scope,
        budget=working_budget,
        recall_budget=recall_budget,
        include_global=include_global,
        include_hot=include_hot,
    )
    policy_eval = memory.evaluate_recall_policy(
        scope=scope,
        include_global=include_global,
        limit=limit,
        min_evaluated=min_evaluated,
    )
    impact_eval = memory.evaluate_memory_impact(
        scope=scope,
        include_global=include_global,
        limit=limit,
        min_evaluated=min_evaluated,
    )
    critic_impact_eval = memory.evaluate_recall_critic_impact(
        scope=scope,
        include_global=include_global,
        limit=limit,
        min_evaluated=min_evaluated,
    )
    stress_trend = memory.long_run_stress_trend(
        scope=scope,
        include_global=include_global,
        limit=limit,
        min_samples=stress_min_samples,
    )
    projected_ids = working.influential_capsule_ids
    projected_feedback = _projected_feedback(projected_ids, impact_eval.capsules)
    critic = _critic_summary(policy)
    stress_summary = _stress_trend_summary(stress_trend)
    status = _quality_status(
        policy,
        critic,
        critic_impact_eval,
        policy_eval,
        impact_eval,
        stress_summary,
        projected_feedback,
    )
    return RecallQualityReport(
        query=query,
        scope=scope,
        status=status,
        policy_summary=_policy_summary(policy),
        critic_summary=critic,
        critic_impact_summary=_eval_summary(critic_impact_eval),
        working_summary=_working_summary(working),
        policy_eval_summary=_eval_summary(policy_eval),
        memory_impact_summary=_eval_summary(impact_eval),
        stress_trend_summary=stress_summary,
        projected_feedback=projected_feedback,
        diagnostics={
            "query_terms": extract_keywords(query, limit=16),
            "include_global": include_global,
            "include_hot": include_hot,
            "min_evaluated": min_evaluated,
            "stress_min_samples": stress_min_samples,
            "limit": limit,
            "quality_basis": (
                "recall-policy + recall-critic + working-memory + reviewed impact feedback "
                "+ long-run stress trend"
            ),
            "automatic_policy_mutation": False,
            "critic_status": critic["status"],
            "critic_score": critic["score"],
            "critic_impact_status": critic_impact_eval.status,
            "long_run_stress_trend_status": stress_summary["status"],
        },
        recommendations=_quality_recommendations(
            policy,
            critic,
            critic_impact_eval,
            policy_eval,
            impact_eval,
            stress_summary,
            projected_feedback,
            projected_ids=projected_ids,
        ),
    )


def plan_recall_quality_impact(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    outcome: str,
    helped: bool | None = None,
    apply: bool = False,
    budgets: list[int] | None = None,
    include_global: bool = True,
    include_hot: bool = True,
    working_budget: int = 900,
    recall_budget: int = 1600,
    limit: int = 500,
    min_evaluated: int = 3,
    stress_min_samples: int = 3,
) -> RecallQualityImpactPlan:
    report = build_recall_quality_gate(
        memory,
        query,
        scope=scope,
        budgets=budgets,
        include_global=include_global,
        include_hot=include_hot,
        working_budget=working_budget,
        recall_budget=recall_budget,
        limit=limit,
        min_evaluated=min_evaluated,
        stress_min_samples=stress_min_samples,
    )
    critic = _critic_impact_payload(report, outcome=outcome, helped=helped)
    working = _working_memory_impact_payload(report, outcome=outcome, helped=helped)
    events: list[dict[str, Any]] = []
    if apply:
        critic_event = memory.record_recall_critic_impact(
            scope=scope,
            query=query,
            critic_status=critic["critic_status"],
            decisions=critic["decisions"],
            outcome=outcome,
            helped=helped,
            source="recall-quality-critic-impact",
        )
        events.append({"event_id": critic_event.id, "source": critic_event.source})
        if working["capsule_ids"]:
            working_event = memory.record_memory_impact(
                scope=scope,
                cue=query,
                capsule_ids=working["capsule_ids"],
                outcome=outcome,
                helped=helped,
                source="recall-quality-working-memory-impact",
            )
            events.append({"event_id": working_event.id, "source": working_event.source})
    return RecallQualityImpactPlan(
        query=query,
        scope=scope,
        status="recorded" if apply else "planned",
        helped=helped,
        outcome=outcome,
        dry_run=not apply,
        quality_summary={
            "status": report.status,
            "critic_status": report.critic_summary["status"],
            "critic_score": report.critic_summary["score"],
            "policy_status": report.policy_summary["status"],
            "working_projected_capsules": len(report.working_summary["projected_capsule_ids"]),
            "stress_trend_status": report.stress_trend_summary["status"],
        },
        critic_impact=critic,
        working_memory_impact=working,
        events=events,
        recommendations=_impact_plan_recommendations(report, apply=apply, helped=helped),
    )


def _policy_summary(policy: Any) -> dict[str, Any]:
    return {
        "status": policy.status,
        "intent": policy.intent,
        "strategy": policy.strategy,
        "recommended_budget": policy.recall_plan.recommended_budget,
        "actions": [action.as_dict() for action in policy.actions],
        "risks": list(policy.diagnostics.get("risks", [])),
    }


def _critic_impact_payload(report: RecallQualityReport, *, outcome: str, helped: bool | None) -> dict[str, Any]:
    decisions = [
        f"quality={report.status}",
        f"critic_score={report.critic_summary['score']}",
        f"visible_capsules={report.critic_summary['diagnostics'].get('visible_capsules', 0)}",
        f"query_terms={report.critic_summary['diagnostics'].get('query_terms_visible_count', 0)}/"
        f"{report.critic_summary['diagnostics'].get('query_term_count', 0)}",
    ]
    return {
        "critic_status": report.critic_summary["status"],
        "decisions": decisions,
        "outcome": outcome,
        "helped": helped,
    }


def _working_memory_impact_payload(report: RecallQualityReport, *, outcome: str, helped: bool | None) -> dict[str, Any]:
    return {
        "cue": report.query,
        "capsule_ids": list(report.working_summary["projected_capsule_ids"]),
        "outcome": outcome,
        "helped": helped,
    }


def _impact_plan_recommendations(report: RecallQualityReport, *, apply: bool, helped: bool | None) -> list[str]:
    recommendations = [
        "Use this as reviewed outcome evidence only after the current turn outcome is known.",
        "This records evidence; it does not tune thresholds or promote memories automatically.",
    ]
    if helped is None:
        recommendations.append("Helped is unknown, so this will increase coverage but not evaluated helpful/harmful counts.")
    if not report.working_summary["projected_capsule_ids"]:
        recommendations.append("No projected working-memory capsules were present; only critic impact can be recorded.")
    if not apply:
        recommendations.append("Dry-run only; rerun with --apply after reviewing outcome and helped flag.")
    return recommendations


def _critic_summary(policy: Any) -> dict[str, Any]:
    critic = dict(policy.recall_plan.diagnostics.get("critic", {}))
    diagnostics = dict(critic.get("diagnostics", {}))
    if not critic:
        critic = {
            "status": str(policy.recall_plan.diagnostics.get("critic_status", "unknown")),
            "score": int(policy.recall_plan.diagnostics.get("critic_score", 0)),
            "reasons": list(policy.recall_plan.diagnostics.get("critic_reasons", [])),
            "recommendations": list(policy.recall_plan.diagnostics.get("critic_recommendations", [])),
            "diagnostics": diagnostics,
        }
    critic["status"] = str(critic.get("status", "unknown"))
    critic["score"] = int(critic.get("score", 0))
    critic["reasons"] = list(critic.get("reasons", []))
    critic["recommendations"] = list(critic.get("recommendations", []))
    critic["diagnostics"] = diagnostics
    return critic


def _working_summary(working: Any) -> dict[str, Any]:
    return {
        "items": len(working.items),
        "projected_capsule_ids": working.influential_capsule_ids,
        "sections": dict(working.diagnostics.get("sections", {})),
        "estimated_tokens": int(working.diagnostics.get("estimated_tokens", 0)),
        "low_evidence_fallback_suppressed": bool(
            working.diagnostics.get("low_evidence_fallback_suppressed", False)
        ),
    }


def _eval_summary(report: Any) -> dict[str, Any]:
    return {
        "status": report.status,
        "totals": dict(report.totals),
        "recommendations": list(report.recommendations),
    }


def _stress_trend_summary(report: Any) -> dict[str, Any]:
    return {
        "status": report.status,
        "totals": dict(report.totals),
        "latest": dict(report.latest or {}),
        "score_delta": report.score_delta,
        "token_growth_delta": report.token_growth_delta,
        "harmful_ratio_delta": report.harmful_ratio_delta,
        "recommendations": list(report.recommendations),
    }


def _projected_feedback(projected_ids: list[str], groups: list[Any]) -> list[ProjectedCapsuleFeedback]:
    by_id = {group.key: group for group in groups}
    feedback: list[ProjectedCapsuleFeedback] = []
    for capsule_id in projected_ids:
        group = by_id.get(capsule_id)
        if group is None:
            continue
        verdict = "unknown"
        if group.evaluated > 0 and group.harmful > group.helpful:
            verdict = "harmful-history"
        elif group.evaluated > 0 and group.helpful > group.harmful:
            verdict = "helpful-history"
        elif group.evaluated > 0:
            verdict = "mixed-history"
        feedback.append(
            ProjectedCapsuleFeedback(
                capsule_id=capsule_id,
                total=group.total,
                evaluated=group.evaluated,
                helpful=group.helpful,
                harmful=group.harmful,
                unknown=group.unknown,
                verdict=verdict,
            )
        )
    return feedback


def _quality_status(
    policy: Any,
    critic: dict[str, Any],
    critic_impact_eval: Any,
    policy_eval: Any,
    impact_eval: Any,
    stress_trend: dict[str, Any],
    projected: list[ProjectedCapsuleFeedback],
) -> str:
    if critic.get("status") == "fail":
        return "fail"
    if policy.status == "fail":
        return "fail"
    if any(item.verdict == "harmful-history" for item in projected):
        return "fail"
    if stress_trend["status"] == "fail":
        return "fail"
    if policy_eval.status == "fail" or impact_eval.status == "fail":
        return "fail"
    if critic_impact_eval.status == "fail":
        return "watch"
    if (
        policy.status == "watch"
        or critic.get("status") == "watch"
        or policy_eval.status == "watch"
        or impact_eval.status == "watch"
        or stress_trend["status"] == "watch"
        or not projected
        or any(item.verdict in {"mixed-history", "unknown"} for item in projected)
    ):
        return "watch"
    return "pass"


def _quality_recommendations(
    policy: Any,
    critic: dict[str, Any],
    critic_impact_eval: Any,
    policy_eval: Any,
    impact_eval: Any,
    stress_trend: dict[str, Any],
    projected: list[ProjectedCapsuleFeedback],
    *,
    projected_ids: list[str],
) -> list[str]:
    recommendations: list[str] = []
    harmful = [item for item in projected if item.verdict == "harmful-history"]
    for item in harmful[:3]:
        recommendations.append(
            f"Do not trust projected capsule {item.capsule_id} without review; harmful={item.harmful}, helpful={item.helpful}."
        )
    if critic.get("status") == "fail":
        recommendations.append("Do not trust recalled memory context; recall critic failed the visible-evidence gate.")
        recommendations.extend(str(item) for item in critic.get("recommendations", [])[:2])
    elif critic.get("status") == "watch":
        recommendations.append("Treat recalled memory as cues only; recall critic marked the pack watch.")
        recommendations.extend(str(item) for item in critic.get("recommendations", [])[:2])
    if critic_impact_eval.status == "fail":
        recommendations.extend(critic_impact_eval.recommendations[:2])
    if policy.status != "pass":
        recommendations.append(f"Resolve recall-policy status {policy.status} before treating the route as natural memory.")
    if policy_eval.status != "pass":
        recommendations.extend(policy_eval.recommendations[:2])
    if impact_eval.status != "pass":
        recommendations.extend(impact_eval.recommendations[:2])
    if stress_trend["status"] != "pass":
        recommendations.extend(stress_trend["recommendations"][:2])
        recommendations.append(
            f"Resolve long-run stress trend status {stress_trend['status']} before treating recall quality as stable."
        )
    if projected_ids and not projected:
        recommendations.append(
            "Record working-memory-impact for projected capsule ids after this turn; current cue has no reviewed capsule feedback."
        )
    if not projected_ids:
        recommendations.append("No working-memory capsule projected for this cue; use current evidence and record outcome.")
    if not recommendations:
        recommendations.append("Recall quality gate passed; still record reviewed impact after the turn.")
    return list(dict.fromkeys(recommendations))
