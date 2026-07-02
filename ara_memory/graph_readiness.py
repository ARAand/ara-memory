from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_GRAPH_READINESS_QUERY = "next natural memory architecture graph activation temporal edge recall"


@dataclass(slots=True)
class GraphActivationReadiness:
    scope: str
    query: str
    passed: bool
    status: str
    diagnostics: dict[str, Any]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "passed": self.passed,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        diagnostics = self.diagnostics
        lines = [
            f"# Ara Graph Activation Readiness: {self.scope}",
            f"status: {self.status}",
            f"query: {self.query}",
            "## Evidence",
            (
                "- "
                f"used={diagnostics['spreading_activation_used']}, "
                f"edges={diagnostics['graph_activation_edges']}, "
                f"boosted={diagnostics['spreading_activation_boosted_count']}, "
                f"supplemented={diagnostics['spreading_activation_supplemented_count']}, "
                f"visible={diagnostics['visible_capsules']}, "
                f"tokens={diagnostics['estimated_tokens']}, "
                f"budget={diagnostics['best_budget']}"
            ),
            "## Recommendations",
        ]
        lines.extend(f"- {item}" for item in self.recommendations)
        if not self.recommendations:
            lines.append("- Graph activation has live bounded recall evidence.")
        return "\n".join(lines)


def run_graph_activation_readiness(
    memory: Any,
    *,
    scope: str = "global",
    query: str = DEFAULT_GRAPH_READINESS_QUERY,
    budgets: list[int] | None = None,
    include_global: bool = True,
) -> GraphActivationReadiness:
    plan = memory.recall_plan(
        query,
        scope=scope,
        budgets=budgets or [800, 1600],
        include_hot=False,
        include_global=include_global,
    )
    alternatives = plan.as_dict()["alternatives"]
    best = max(
        alternatives,
        key=lambda item: (
            bool(item.get("spreading_activation_used", False)),
            int(item.get("spreading_activation_boosted_count", 0)),
            int(item.get("spreading_activation_supplemented_count", 0)),
            int(item.get("graph_activation_edges", 0)),
            int(item.get("visible_capsules", 0)),
        ),
    )
    diagnostics = {
        "recommended_budget": int(plan.recommended_budget),
        "best_budget": int(best.get("budget", 0)),
        "estimated_tokens": int(best.get("estimated_tokens", 0)),
        "selected_capsules": int(best.get("selected_capsules", 0)),
        "visible_capsules": int(best.get("visible_capsules", 0)),
        "quality_score": int(best.get("quality_score", 0)),
        "graph_activation_edges": int(best.get("graph_activation_edges", 0)),
        "spreading_activation_used": bool(best.get("spreading_activation_used", False)),
        "spreading_activation_boosted_count": int(best.get("spreading_activation_boosted_count", 0)),
        "spreading_activation_supplemented_count": int(best.get("spreading_activation_supplemented_count", 0)),
    }
    passed = (
        diagnostics["spreading_activation_used"]
        and diagnostics["graph_activation_edges"] > 0
        and diagnostics["spreading_activation_boosted_count"] > 0
        and diagnostics["visible_capsules"] > 0
    )
    recommendations: list[str] = []
    if not passed:
        recommendations.append(
            "Capture or consolidate temporal edges that connect lexical recall seeds to source capsules, "
            "then rerun graph activation readiness."
        )
    return GraphActivationReadiness(
        scope=scope,
        query=query,
        passed=passed,
        status="pass" if passed else "watch",
        diagnostics=diagnostics,
        recommendations=recommendations,
    )
