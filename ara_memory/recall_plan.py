from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ara_memory.costs import estimate_api_cost


@dataclass(slots=True)
class RecallPlan:
    query: str
    scope: str
    recommended_budget: int
    include_hot: bool
    include_global: bool
    selected_capsules: int
    estimated_tokens: int
    estimated_tokens_without_hot: int
    api_cost: dict[str, float | int]
    alternatives: list[dict[str, Any]]
    rationale: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "recommended_budget": self.recommended_budget,
            "include_hot": self.include_hot,
            "include_global": self.include_global,
            "selected_capsules": self.selected_capsules,
            "estimated_tokens": self.estimated_tokens,
            "estimated_tokens_without_hot": self.estimated_tokens_without_hot,
            "api_cost": self.api_cost,
            "alternatives": self.alternatives,
            "rationale": self.rationale,
            "diagnostics": self.diagnostics,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Recall Plan: {self.scope}",
            f"query: {self.query}",
            f"recommended_budget: {self.recommended_budget}",
            f"include_hot: {self.include_hot}",
            f"include_global: {self.include_global}",
            f"estimated_tokens: {self.estimated_tokens}",
            f"selected_capsules: {self.selected_capsules}",
            "## Rationale",
        ]
        lines.extend(f"- {item}" for item in self.rationale)
        lines.append("## Alternatives")
        for item in self.alternatives:
            lines.append(
                "- "
                f"budget={item['budget']}, hot={item['include_hot']}, global={item['include_global']}, "
                f"tokens={item['estimated_tokens']}, capsules={item['selected_capsules']}, "
                f"visible={item['visible_capsules']}, quality={item['quality_score']}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class RecallContext:
    plan: RecallPlan
    pack: str
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "pack": self.pack,
            "diagnostics": self.diagnostics,
        }

    def to_text(self, *, include_plan: bool = True) -> str:
        if not include_plan:
            return self.pack
        return self.plan.to_text() + "\n\n---\n\n" + self.pack


def build_recall_plan(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budgets: list[int] | None = None,
    include_global: bool = True,
    include_hot: bool = True,
    output_tokens: int = 0,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
) -> RecallPlan:
    budgets = sorted(set(budgets or [800, 1600, 2500]))
    if not budgets:
        raise ValueError("At least one recall budget is required.")
    alternatives = []
    for budget in budgets:
        result = memory.recall_result(
            query,
            scope=scope,
            budget=budget,
            include_global=include_global,
            include_hot=include_hot,
        )
        diagnostics = result.diagnostics
        alternatives.append(
            {
                "budget": budget,
                "include_hot": include_hot,
                "include_global": include_global,
                "estimated_tokens": int(diagnostics["estimated_tokens_after"]),
                "selected_capsules": int(diagnostics["capsules_selected"]),
                "rendered_capsules": int(diagnostics.get("capsules_rendered_before_budget", 0)),
                "visible_capsules": int(diagnostics.get("capsules_rendered_after_budget", 0)),
                "graph_edges": int(diagnostics["graph_edges_considered"]),
                "graph_activation_edges": int(diagnostics.get("graph_activation_edges_considered", 0)),
                "spreading_activation_used": bool(diagnostics.get("spreading_activation_used", False)),
                "spreading_activation_boosted_count": int(
                    diagnostics.get("spreading_activation_boosted_count", 0)
                ),
                "spreading_activation_supplemented_count": int(
                    diagnostics.get("spreading_activation_supplemented_count", 0)
                ),
                "query_term_count": int(diagnostics.get("query_term_count", 0)),
                "query_terms_visible_count": int(diagnostics.get("query_terms_visible_count", 0)),
                "visible_section_count": int(diagnostics.get("visible_section_count", 0)),
                "sections_truncated": int(diagnostics.get("sections_truncated", 0)),
                "fallback_used": bool(diagnostics.get("fallback_used", False)),
                "low_evidence_fallback_suppressed": bool(
                    diagnostics.get("low_evidence_fallback_suppressed", False)
                ),
                "salience_supplement_used": bool(diagnostics.get("salience_supplement_used", False)),
                "relevance_score_avg": float(diagnostics.get("relevance_score_avg", 0.0)),
                "selected_capsule_ids": list(diagnostics["selected_capsule_ids"]),
                "visible_capsule_ids": list(diagnostics.get("visible_capsule_ids", [])),
            }
        )
        alternatives[-1]["quality_score"] = _alternative_quality(alternatives[-1])
    selected = _select_alternative(alternatives)
    without_hot = memory.recall_result(
        query,
        scope=scope,
        budget=int(selected["budget"]),
        include_global=include_global,
        include_hot=False,
    )
    cost = estimate_api_cost(
        input_tokens=int(selected["estimated_tokens"]),
        output_tokens=output_tokens,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    rationale = _rationale(
        alternatives,
        selected,
        without_hot_tokens=int(without_hot.diagnostics["estimated_tokens_after"]),
        include_hot=include_hot,
        include_global=include_global,
    )
    return RecallPlan(
        query=query,
        scope=scope,
        recommended_budget=int(selected["budget"]),
        include_hot=include_hot,
        include_global=include_global,
        selected_capsules=int(selected["selected_capsules"]),
        estimated_tokens=int(selected["estimated_tokens"]),
        estimated_tokens_without_hot=int(without_hot.diagnostics["estimated_tokens_after"]),
        api_cost={
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "input_cost_usd": cost.input_cost_usd,
            "output_cost_usd": cost.output_cost_usd,
            "total_cost_usd": cost.total_cost_usd,
        },
        alternatives=alternatives,
        rationale=rationale,
        diagnostics=_selected_diagnostics(selected),
    )


def build_recall_context(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budgets: list[int] | None = None,
    include_global: bool = True,
    include_hot: bool = True,
    output_tokens: int = 0,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
) -> RecallContext:
    plan = build_recall_plan(
        memory,
        query,
        scope=scope,
        budgets=budgets,
        include_global=include_global,
        include_hot=include_hot,
        output_tokens=output_tokens,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    result = memory.recall_result(
        query,
        scope=scope,
        budget=plan.recommended_budget,
        include_global=plan.include_global,
        include_hot=plan.include_hot,
    )
    return RecallContext(plan=plan, pack=result.pack, diagnostics=result.diagnostics)


def _select_alternative(alternatives: list[dict[str, Any]]) -> dict[str, Any]:
    non_empty = [item for item in alternatives if int(item["visible_capsules"]) > 0]
    if not non_empty:
        return alternatives[0]
    for item in non_empty:
        if float(item["quality_score"]) >= 60.0:
            return item
    return max(non_empty, key=lambda item: (float(item["quality_score"]), -int(item["budget"])))


def _rationale(
    alternatives: list[dict[str, Any]],
    selected: dict[str, Any],
    *,
    without_hot_tokens: int,
    include_hot: bool,
    include_global: bool,
) -> list[str]:
    items = [
        "Choose the smallest tested budget with enough visible evidence, not just selected capsules.",
        "Use the selected recall pack instead of reading the raw event ledger.",
    ]
    items.append(
        f"Selected pack quality={selected['quality_score']}, "
        f"visible_capsules={selected['visible_capsules']}, "
        f"visible_query_terms={selected['query_terms_visible_count']}/{selected['query_term_count']}."
    )
    if selected.get("low_evidence_fallback_suppressed"):
        items.append("Selected pack had no direct evidence; salience fallback bodies were suppressed.")
    elif selected.get("fallback_used"):
        items.append("Selected pack used salience fallback; treat it as lower-confidence context.")
    elif selected.get("salience_supplement_used"):
        items.append("Selected pack mixed FTS hits with salience supplements to preserve useful context.")
    if selected.get("spreading_activation_used"):
        items.append(
            "Graph activation contributed locally: "
            f"edges={selected.get('graph_activation_edges', 0)}, "
            f"boosted={selected.get('spreading_activation_boosted_count', 0)}, "
            f"supplemented={selected.get('spreading_activation_supplemented_count', 0)}."
        )
    if include_hot:
        delta = int(selected["estimated_tokens"]) - without_hot_tokens
        items.append(f"Hot memory adds about {max(0, delta)} tokens at this budget.")
    if include_global:
        items.append("Global memory is included for shared identity, goals, and cross-project policy.")
    if alternatives and int(selected["budget"]) < int(alternatives[-1]["budget"]):
        items.append("A larger fallback budget is available if the first answer lacks evidence.")
    return items


def _alternative_quality(item: dict[str, Any]) -> int:
    query_terms = max(1, int(item.get("query_term_count", 0)))
    visible_terms = int(item.get("query_terms_visible_count", 0))
    term_coverage = visible_terms / query_terms
    visible_capsules = int(item.get("visible_capsules", 0))
    section_count = int(item.get("visible_section_count", 0))
    avg_relevance = float(item.get("relevance_score_avg", 0.0))
    if bool(item.get("low_evidence_fallback_suppressed", False)):
        return 0
    score = 0.0
    score += min(30.0, visible_capsules * 7.5)
    score += min(40.0, term_coverage * 40.0)
    score += min(15.0, section_count * 3.0)
    score += min(15.0, max(0.0, avg_relevance) * 1.8)
    if bool(item.get("fallback_used", False)):
        score -= 18.0
    elif bool(item.get("salience_supplement_used", False)):
        score -= 4.0
    if int(item.get("sections_truncated", 0)) > 6 and term_coverage < 0.5:
        score -= 8.0
    return max(0, min(100, int(round(score))))


def _selected_diagnostics(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "quality_score": int(selected.get("quality_score", 0)),
        "visible_capsules": int(selected.get("visible_capsules", 0)),
        "rendered_capsules": int(selected.get("rendered_capsules", 0)),
        "selected_capsules": int(selected.get("selected_capsules", 0)),
        "query_term_count": int(selected.get("query_term_count", 0)),
        "query_terms_visible_count": int(selected.get("query_terms_visible_count", 0)),
        "visible_section_count": int(selected.get("visible_section_count", 0)),
        "sections_truncated": int(selected.get("sections_truncated", 0)),
        "graph_activation_edges": int(selected.get("graph_activation_edges", 0)),
        "spreading_activation_used": bool(selected.get("spreading_activation_used", False)),
        "spreading_activation_boosted_count": int(selected.get("spreading_activation_boosted_count", 0)),
        "spreading_activation_supplemented_count": int(
            selected.get("spreading_activation_supplemented_count", 0)
        ),
        "fallback_used": bool(selected.get("fallback_used", False)),
        "low_evidence_fallback_suppressed": bool(selected.get("low_evidence_fallback_suppressed", False)),
        "salience_supplement_used": bool(selected.get("salience_supplement_used", False)),
        "visible_capsule_ids": list(selected.get("visible_capsule_ids", [])),
    }
