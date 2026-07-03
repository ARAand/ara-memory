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
    working_summary: dict[str, Any]
    policy_eval_summary: dict[str, Any]
    memory_impact_summary: dict[str, Any]
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
            "working_memory": self.working_summary,
            "policy_eval": self.policy_eval_summary,
            "memory_impact": self.memory_impact_summary,
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
    projected_ids = working.influential_capsule_ids
    projected_feedback = _projected_feedback(projected_ids, impact_eval.capsules)
    critic = _critic_summary(policy)
    status = _quality_status(policy, critic, policy_eval, impact_eval, projected_feedback)
    return RecallQualityReport(
        query=query,
        scope=scope,
        status=status,
        policy_summary=_policy_summary(policy),
        critic_summary=critic,
        working_summary=_working_summary(working),
        policy_eval_summary=_eval_summary(policy_eval),
        memory_impact_summary=_eval_summary(impact_eval),
        projected_feedback=projected_feedback,
        diagnostics={
            "query_terms": extract_keywords(query, limit=16),
            "include_global": include_global,
            "include_hot": include_hot,
            "min_evaluated": min_evaluated,
            "limit": limit,
            "quality_basis": "recall-policy + recall-critic + working-memory + reviewed impact feedback",
            "automatic_policy_mutation": False,
            "critic_status": critic["status"],
            "critic_score": critic["score"],
        },
        recommendations=_quality_recommendations(
            policy,
            critic,
            policy_eval,
            impact_eval,
            projected_feedback,
            projected_ids=projected_ids,
        ),
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
    policy_eval: Any,
    impact_eval: Any,
    projected: list[ProjectedCapsuleFeedback],
) -> str:
    if critic.get("status") == "fail":
        return "fail"
    if policy.status == "fail":
        return "fail"
    if any(item.verdict == "harmful-history" for item in projected):
        return "fail"
    if policy_eval.status == "fail" or impact_eval.status == "fail":
        return "fail"
    if (
        policy.status == "watch"
        or critic.get("status") == "watch"
        or policy_eval.status == "watch"
        or impact_eval.status == "watch"
        or not projected
        or any(item.verdict in {"mixed-history", "unknown"} for item in projected)
    ):
        return "watch"
    return "pass"


def _quality_recommendations(
    policy: Any,
    critic: dict[str, Any],
    policy_eval: Any,
    impact_eval: Any,
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
    if policy.status != "pass":
        recommendations.append(f"Resolve recall-policy status {policy.status} before treating the route as natural memory.")
    if policy_eval.status != "pass":
        recommendations.extend(policy_eval.recommendations[:2])
    if impact_eval.status != "pass":
        recommendations.extend(impact_eval.recommendations[:2])
    if projected_ids and not projected:
        recommendations.append(
            "Record working-memory-impact for projected capsule ids after this turn; current cue has no reviewed capsule feedback."
        )
    if not projected_ids:
        recommendations.append("No working-memory capsule projected for this cue; use current evidence and record outcome.")
    if not recommendations:
        recommendations.append("Recall quality gate passed; still record reviewed impact after the turn.")
    return list(dict.fromkeys(recommendations))
