from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RecallCriticReport:
    query: str
    scope: str
    status: str
    score: int
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "status": self.status,
            "score": self.score,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
            "diagnostics": self.diagnostics,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Recall Critic: {self.scope}",
            f"query: {self.query}",
            f"status: {self.status}",
            f"score: {self.score}",
            "## Reasons",
        ]
        lines.extend(f"- {item}" for item in (self.reasons or ["No risks detected."]))
        lines.append("## Recommendations")
        lines.extend(f"- {item}" for item in (self.recommendations or ["Use the recall pack normally."]))
        return "\n".join(lines)


def build_recall_critic(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budget: int = 1600,
    include_global: bool = True,
    include_hot: bool = True,
) -> RecallCriticReport:
    result = memory.recall_result(
        query,
        scope=scope,
        budget=budget,
        include_global=include_global,
        include_hot=include_hot,
    )
    return build_recall_critic_from_diagnostics(
        query,
        scope=scope,
        diagnostics=result.diagnostics,
        budget=budget,
    )


def build_recall_critic_from_diagnostics(
    query: str,
    *,
    scope: str,
    diagnostics: dict[str, Any],
    budget: int | None = None,
) -> RecallCriticReport:
    summary = _summary(diagnostics, budget=budget)
    reasons: list[str] = []
    recommendations: list[str] = []
    score = 100
    fail = False
    watch = False

    if summary["estimated_tokens"] > summary["budget_tokens"]:
        fail = True
        score -= 55
        reasons.append("recall pack exceeds the configured token budget")
        recommendations.append("rerun recall with a smaller pack or use current evidence only")

    if summary["low_evidence_fallback_suppressed"]:
        fail = True
        score -= 60
        reasons.append("salience fallback was suppressed because no direct evidence matched")
        recommendations.append("do not trust this memory pack; gather current evidence or reformulate the query")
    elif summary["visible_capsules"] == 0 and summary["query_term_count"] > 0:
        fail = True
        score -= 45
        reasons.append("no rendered capsule evidence is visible for the query")
        recommendations.append("use current evidence first, then run recall-plan with a wider budget if needed")

    if summary["fallback_used"]:
        watch = True
        score -= 18
        reasons.append("recall used salience fallback rather than direct lexical evidence")
        recommendations.append("treat recalled context as orientation, not authority")

    if summary["capsules_filtered_by_risk"] > 0:
        watch = True
        score -= min(20, summary["capsules_filtered_by_risk"] * 4)
        reasons.append("some candidate capsules were filtered by risk policy")
        recommendations.append("inspect risk-filtered memory only through explicit review tools")

    if summary["query_term_count"] > 0:
        coverage = summary["query_terms_visible_count"] / summary["query_term_count"]
        if summary["visible_capsules"] > 0 and coverage < 0.34:
            watch = True
            score -= 14
            reasons.append("visible evidence covers less than one third of query terms")
            recommendations.append("ask a narrower recall query before trusting the pack")

    if summary["sections_truncated"] > 6 and summary["query_terms_visible_count"] == 0:
        watch = True
        score -= 8
        reasons.append("many sections were compressed while query terms were not visible")
        recommendations.append("increase budget only after recall-plan shows direct evidence")

    score = max(0, min(100, score))
    status = "fail" if fail else "watch" if watch else "pass"
    if not recommendations and status == "pass":
        recommendations.append("memory context is usable within this budget")
    return RecallCriticReport(
        query=query,
        scope=scope,
        status=status,
        score=score,
        reasons=reasons,
        recommendations=_dedupe(recommendations),
        diagnostics=summary,
    )


def critic_payload_from_probe(
    query: str,
    *,
    scope: str,
    probe: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = {
        "budget_tokens": int(probe.get("budget", 0)),
        "estimated_tokens_after": int(probe.get("estimated_tokens", 0)),
        "capsules_rendered_after_budget": int(probe.get("visible_capsules", 0)),
        "capsules_selected": int(probe.get("selected_capsules", 0)),
        "capsules_filtered_by_risk": int(probe.get("capsules_filtered_by_risk", 0)),
        "query_term_count": len(list(probe.get("query_terms", []))),
        "query_terms_visible_count": int(probe.get("query_terms_visible_count", 0)),
        "fallback_used": bool(probe.get("fallback_used", False)),
        "low_evidence_fallback_suppressed": bool(
            probe.get("low_evidence_fallback_suppressed", False)
        ),
        "salience_supplement_used": bool(probe.get("salience_supplement_used", False)),
    }
    return build_recall_critic_from_diagnostics(
        query,
        scope=scope,
        diagnostics=diagnostics,
        budget=int(probe.get("budget", 0)),
    ).as_dict()


def _summary(diagnostics: dict[str, Any], *, budget: int | None) -> dict[str, Any]:
    budget_tokens = int(budget if budget is not None else diagnostics.get("budget_tokens", 0))
    if budget_tokens <= 0:
        budget_tokens = int(diagnostics.get("budget_tokens", 0))
    return {
        "budget_tokens": budget_tokens,
        "estimated_tokens": int(diagnostics.get("estimated_tokens_after", 0)),
        "selected_capsules": int(diagnostics.get("capsules_selected", 0)),
        "visible_capsules": int(
            diagnostics.get(
                "capsules_rendered_after_budget",
                diagnostics.get("visible_capsules", 0),
            )
        ),
        "capsules_filtered_by_risk": int(diagnostics.get("capsules_filtered_by_risk", 0)),
        "query_term_count": int(diagnostics.get("query_term_count", 0)),
        "query_terms_visible_count": int(diagnostics.get("query_terms_visible_count", 0)),
        "sections_truncated": int(diagnostics.get("sections_truncated", 0)),
        "fallback_used": bool(diagnostics.get("fallback_used", False)),
        "low_evidence_fallback_suppressed": bool(
            diagnostics.get("low_evidence_fallback_suppressed", False)
        ),
        "salience_supplement_used": bool(diagnostics.get("salience_supplement_used", False)),
        "visible_capsule_ids": list(diagnostics.get("visible_capsule_ids", [])),
        "selected_capsule_ids": list(diagnostics.get("selected_capsule_ids", [])),
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
