from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_GRAPH_READINESS_QUERIES = [
    "next natural memory architecture graph activation temporal edge recall",
    "how should Ara run backup restore before destructive memory cleanup",
]


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
            f"probes_evaluated: {diagnostics['probes_evaluated']}",
            "## Evidence",
            (
                "- "
                f"used={diagnostics['spreading_activation_used']}, "
                f"edges={diagnostics['graph_activation_edges']}, "
                f"depths={diagnostics['spreading_activation_depth_counts']}, "
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
    query: str | None = None,
    budgets: list[int] | None = None,
    include_global: bool = True,
) -> GraphActivationReadiness:
    probe_queries = [query] if query else DEFAULT_GRAPH_READINESS_QUERIES
    evaluated = [_evaluate_probe(memory, probe, scope=scope, budgets=budgets, include_global=include_global) for probe in probe_queries]
    best_probe = max(evaluated, key=lambda item: _probe_rank(item["best"]))
    best = best_probe["best"]
    diagnostics = {
        "recommended_budget": int(best_probe["recommended_budget"]),
        "best_budget": int(best.get("budget", 0)),
        "estimated_tokens": int(best.get("estimated_tokens", 0)),
        "selected_capsules": int(best.get("selected_capsules", 0)),
        "visible_capsules": int(best.get("visible_capsules", 0)),
        "quality_score": int(best.get("quality_score", 0)),
        "graph_activation_edges": int(best.get("graph_activation_edges", 0)),
        "spreading_activation_used": bool(best.get("spreading_activation_used", False)),
        "spreading_activation_multi_hop_used": bool(best.get("spreading_activation_multi_hop_used", False)),
        "spreading_activation_depth_counts": dict(best.get("spreading_activation_depth_counts", {})),
        "spreading_activation_boosted_count": int(best.get("spreading_activation_boosted_count", 0)),
        "spreading_activation_supplemented_count": int(best.get("spreading_activation_supplemented_count", 0)),
        "probes_evaluated": len(evaluated),
        "probe_results": [
            {
                "query": item["query"],
                "best_budget": int(item["best"].get("budget", 0)),
                "graph_activation_edges": int(item["best"].get("graph_activation_edges", 0)),
                "spreading_activation_used": bool(item["best"].get("spreading_activation_used", False)),
                "spreading_activation_multi_hop_used": bool(
                    item["best"].get("spreading_activation_multi_hop_used", False)
                ),
                "spreading_activation_depth_counts": dict(
                    item["best"].get("spreading_activation_depth_counts", {})
                ),
                "spreading_activation_boosted_count": int(item["best"].get("spreading_activation_boosted_count", 0)),
                "spreading_activation_supplemented_count": int(
                    item["best"].get("spreading_activation_supplemented_count", 0)
                ),
                "visible_capsules": int(item["best"].get("visible_capsules", 0)),
            }
            for item in evaluated
        ],
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
        query=str(best_probe["query"]),
        passed=passed,
        status="pass" if passed else "watch",
        diagnostics=diagnostics,
        recommendations=recommendations,
    )


def _evaluate_probe(
    memory: Any,
    query: str,
    *,
    scope: str,
    budgets: list[int] | None,
    include_global: bool,
) -> dict[str, Any]:
    plan = memory.recall_plan(
        query,
        scope=scope,
        budgets=budgets or [800, 1600],
        include_hot=False,
        include_global=include_global,
    )
    alternatives = plan.as_dict()["alternatives"]
    best = max(alternatives, key=_probe_rank)
    return {
        "query": query,
        "recommended_budget": int(plan.recommended_budget),
        "best": best,
    }


def _probe_rank(item: dict[str, Any]) -> tuple[bool, int, int, int, int, int]:
    return (
        bool(item.get("spreading_activation_used", False)),
        int(item.get("spreading_activation_boosted_count", 0)),
        int(item.get("spreading_activation_supplemented_count", 0)),
        int(item.get("graph_activation_edges", 0)),
        int(item.get("visible_capsules", 0)),
        int(item.get("quality_score", 0)),
    )
