from __future__ import annotations

from dataclasses import dataclass
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
                f"tokens={item['estimated_tokens']}, capsules={item['selected_capsules']}"
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
                "graph_edges": int(diagnostics["graph_edges_considered"]),
                "selected_capsule_ids": list(diagnostics["selected_capsule_ids"]),
            }
        )
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
    non_empty = [item for item in alternatives if int(item["selected_capsules"]) > 0]
    if not non_empty:
        return alternatives[0]
    for item in non_empty:
        if int(item["selected_capsules"]) >= 3:
            return item
    return non_empty[-1]


def _rationale(
    alternatives: list[dict[str, Any]],
    selected: dict[str, Any],
    *,
    without_hot_tokens: int,
    include_hot: bool,
    include_global: bool,
) -> list[str]:
    items = [
        "Choose the smallest tested budget that retrieves a useful capsule set.",
        "Use the selected recall pack instead of reading the raw event ledger.",
    ]
    if include_hot:
        delta = int(selected["estimated_tokens"]) - without_hot_tokens
        items.append(f"Hot memory adds about {max(0, delta)} tokens at this budget.")
    if include_global:
        items.append("Global memory is included for shared identity, goals, and cross-project policy.")
    if alternatives and int(selected["budget"]) < int(alternatives[-1]["budget"]):
        items.append("A larger fallback budget is available if the first answer lacks evidence.")
    return items
