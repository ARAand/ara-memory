from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ara_memory.compressors import compact_text
from ara_memory.models import Event


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


@dataclass(slots=True)
class RecallCriticImpactGroup:
    key: str
    total: int
    helpful: int
    harmful: int
    unknown: int

    @property
    def evaluated(self) -> int:
        return self.helpful + self.harmful

    @property
    def helpful_rate(self) -> float | None:
        if self.evaluated == 0:
            return None
        return self.helpful / self.evaluated

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "total": self.total,
            "evaluated": self.evaluated,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "unknown": self.unknown,
            "helpful_rate": self.helpful_rate,
        }


@dataclass(slots=True)
class RecallCriticImpactEvalReport:
    scope: str
    status: str
    totals: dict[str, Any]
    statuses: list[RecallCriticImpactGroup]
    decisions: list[RecallCriticImpactGroup]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "status": self.status,
            "totals": self.totals,
            "statuses": [item.as_dict() for item in self.statuses],
            "decisions": [item.as_dict() for item in self.decisions],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Recall Critic Impact Eval: {self.scope}",
            f"status: {self.status}",
            "totals: "
            f"impacts={self.totals['impacts']}, evaluated={self.totals['evaluated']}, "
            f"helpful={self.totals['helpful']}, harmful={self.totals['harmful']}, "
            f"unknown={self.totals['unknown']}",
            "## By Critic Status",
        ]
        lines.extend(_impact_group_lines(self.statuses))
        lines.append("## By Decision")
        lines.extend(_impact_group_lines(self.decisions))
        lines.append("## Recommendations")
        lines.extend(f"- {item}" for item in self.recommendations)
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


def record_recall_critic_impact(
    memory: Any,
    *,
    scope: str,
    query: str,
    critic_status: str,
    decisions: list[str],
    outcome: str,
    helped: bool | None = None,
    source: str = "recall-critic-impact",
) -> Event:
    unique_decisions = list(dict.fromkeys(str(item).strip() for item in decisions if str(item).strip()))
    payload = {
        "query": query,
        "critic_status": str(critic_status).strip() or "unknown",
        "decisions": unique_decisions,
        "outcome": outcome,
        "helped": helped,
    }
    text = (
        "Recall critic impact:\n"
        f"Query: {compact_text(query, limit=260)}\n"
        f"Critic status: {payload['critic_status']}\n"
        f"Decisions: {', '.join(unique_decisions) or 'none'}\n"
        f"Outcome: {compact_text(outcome, limit=500)}\n"
        f"Helped: {helped if helped is not None else 'unknown'}"
    )
    event = memory.retain(
        kind="note",
        text=text,
        source=source,
        scope=scope,
        metadata={"recall_critic_impact": payload},
    )
    memory.store.record_recall_critic_impact(event)
    return event


def evaluate_recall_critic_impact(
    memory: Any,
    *,
    scope: str = "global",
    include_global: bool = True,
    limit: int = 500,
    min_evaluated: int = 3,
) -> RecallCriticImpactEvalReport:
    rows = memory.store.list_recall_critic_impacts(
        scope=scope,
        include_global=include_global,
        limit=limit,
    )
    totals = _impact_totals(rows)
    statuses = _impact_groups(rows, key="critic_status")
    decisions = _impact_groups(rows, key="decision")
    status = _impact_status(totals, min_evaluated=min_evaluated)
    return RecallCriticImpactEvalReport(
        scope=scope,
        status=status,
        totals=totals,
        statuses=statuses,
        decisions=decisions,
        recommendations=_impact_recommendations(totals, decisions, min_evaluated=min_evaluated),
    )


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


def _impact_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    helpful = sum(1 for row in rows if row.get("helped") == 1)
    harmful = sum(1 for row in rows if row.get("helped") == 0)
    unknown = sum(1 for row in rows if row.get("helped") is None)
    return {
        "impacts": len(rows),
        "helpful": helpful,
        "harmful": harmful,
        "unknown": unknown,
        "evaluated": helpful + harmful,
    }


def _impact_groups(rows: list[dict[str, Any]], *, key: str) -> list[RecallCriticImpactGroup]:
    grouped: dict[str, Counter[str]] = {}
    for row in rows:
        group_key = str(row.get(key) or "unknown")
        counts = grouped.setdefault(group_key, Counter())
        helped = row.get("helped")
        if helped == 1:
            counts["helpful"] += 1
        elif helped == 0:
            counts["harmful"] += 1
        else:
            counts["unknown"] += 1
        counts["total"] += 1
    groups = [
        RecallCriticImpactGroup(
            key=key,
            total=int(counts["total"]),
            helpful=int(counts["helpful"]),
            harmful=int(counts["harmful"]),
            unknown=int(counts["unknown"]),
        )
        for key, counts in grouped.items()
    ]
    groups.sort(key=lambda item: (item.harmful, item.helpful, item.total, item.key), reverse=True)
    return groups


def _impact_status(totals: dict[str, Any], *, min_evaluated: int) -> str:
    if totals["impacts"] == 0:
        return "watch"
    if totals["evaluated"] < min_evaluated:
        return "watch"
    if totals["harmful"] > totals["helpful"]:
        return "fail"
    if totals["unknown"] > totals["evaluated"]:
        return "watch"
    return "pass"


def _impact_recommendations(
    totals: dict[str, Any],
    decisions: list[RecallCriticImpactGroup],
    *,
    min_evaluated: int,
) -> list[str]:
    if totals["impacts"] == 0:
        return ["Record recall-critic-impact after reviewed turns so critic thresholds become auditable."]
    recommendations: list[str] = []
    if totals["evaluated"] < min_evaluated:
        recommendations.append(
            f"Need {min_evaluated - totals['evaluated']} more reviewed recall-critic outcomes before tuning thresholds."
        )
    harmful = [item for item in decisions if item.harmful > item.helpful and item.evaluated > 0]
    for item in harmful[:3]:
        recommendations.append(
            f"Review critic decision '{item.key}'; harmful={item.harmful}, helpful={item.helpful}."
        )
    if not recommendations:
        recommendations.append("Recall critic impact is bounded threshold evidence only; keep reviewing outcomes.")
    return recommendations


def _impact_group_lines(groups: list[RecallCriticImpactGroup]) -> list[str]:
    if not groups:
        return ["- None."]
    return [
        "- "
        f"{item.key}: total={item.total}, evaluated={item.evaluated}, "
        f"helpful={item.helpful}, harmful={item.harmful}, unknown={item.unknown}, "
        f"helpful_rate={item.helpful_rate if item.helpful_rate is not None else 'n/a'}"
        for item in groups[:10]
    ]
