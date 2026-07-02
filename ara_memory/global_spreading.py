from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_GLOBAL_SPREADING_QUERIES = [
    "Ara identity purpose natural memory global recall policy",
    "cross project memory safety privacy graph spreading recall",
    "long running goal efficient memory store reconsolidation",
]


@dataclass(slots=True)
class GlobalSpreadingSandbox:
    scope: str
    status: str
    passed: bool
    diagnostics: dict[str, Any]
    issues: list[str]
    warnings: list[str]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "status": self.status,
            "passed": self.passed,
            "diagnostics": self.diagnostics,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        diagnostics = self.diagnostics
        lines = [
            f"# Ara Global Spreading Sandbox: {self.scope}",
            f"status: {self.status}",
            f"probes_evaluated: {diagnostics['probes_evaluated']}",
            "## Evidence",
            (
                "- "
                f"global={diagnostics['include_global']}, "
                f"best_query={diagnostics['best_query']}, "
                f"budget={diagnostics['best_budget']}, "
                f"tokens={diagnostics['estimated_tokens']}, "
                f"quality={diagnostics['quality_score']}, "
                f"visible={diagnostics['visible_capsules']}, "
                f"edges={diagnostics['graph_activation_edges']}, "
                f"relation_edges={diagnostics['relation_activation_edges']}, "
                f"depths={diagnostics['spreading_activation_depth_counts']}, "
                f"boosted={diagnostics['spreading_activation_boosted_count']}, "
                f"supplemented={diagnostics['spreading_activation_supplemented_count']}, "
                f"filtered_by_risk={diagnostics['capsules_filtered_by_risk']}"
            ),
            "## Probe Results",
        ]
        for item in diagnostics["probe_results"]:
            lines.append(
                "- "
                f"{item['query']}: budget={item['best_budget']}, "
                f"quality={item['quality_score']}, visible={item['visible_capsules']}, "
                f"edges={item['graph_activation_edges']}, "
                f"supplemented={item['spreading_activation_supplemented_count']}, "
                f"depths={item['spreading_activation_depth_counts']}"
            )
        if self.issues:
            lines.append("## Issues")
            lines.extend(f"- {item}" for item in self.issues)
        if self.warnings:
            lines.append("## Warnings")
            lines.extend(f"- {item}" for item in self.warnings)
        lines.append("## Recommendations")
        lines.extend(f"- {item}" for item in self.recommendations)
        if not self.recommendations:
            lines.append("- Global spreading stayed bounded and evidence-backed.")
        return "\n".join(lines)


def run_global_spreading_sandbox(
    memory: Any,
    *,
    scope: str = "global",
    queries: list[str] | None = None,
    budgets: list[int] | None = None,
    max_edges: int = 120,
    max_supplemented: int = 8,
    max_depth: int = 2,
    min_quality: int = 35,
) -> GlobalSpreadingSandbox:
    probe_queries = [query for query in (queries or DEFAULT_GLOBAL_SPREADING_QUERIES) if query.strip()]
    if not probe_queries:
        raise ValueError("At least one global spreading sandbox query is required.")
    evaluated = [
        _evaluate_probe(memory, query, scope=scope, budgets=budgets)
        for query in probe_queries
    ]
    best_probe = max(evaluated, key=lambda item: _probe_rank(item["best"]))
    best = best_probe["best"]
    max_observed_depth = max((_max_depth(item["best"]) for item in evaluated), default=0)
    diagnostics = {
        "include_global": True,
        "recommended_budget": int(best_probe["recommended_budget"]),
        "best_query": str(best_probe["query"]),
        "best_budget": int(best.get("budget", 0)),
        "estimated_tokens": int(best.get("estimated_tokens", 0)),
        "selected_capsules": int(best.get("selected_capsules", 0)),
        "visible_capsules": int(best.get("visible_capsules", 0)),
        "quality_score": int(best.get("quality_score", 0)),
        "graph_activation_edges": int(best.get("graph_activation_edges", 0)),
        "relation_activation_edges": int(best.get("relation_activation_edges", 0)),
        "spreading_activation_used": bool(best.get("spreading_activation_used", False)),
        "spreading_activation_multi_hop_used": bool(best.get("spreading_activation_multi_hop_used", False)),
        "spreading_activation_depth_counts": dict(best.get("spreading_activation_depth_counts", {})),
        "spreading_activation_boosted_count": int(best.get("spreading_activation_boosted_count", 0)),
        "spreading_activation_supplemented_count": int(best.get("spreading_activation_supplemented_count", 0)),
        "capsules_filtered_by_risk": int(best.get("capsules_filtered_by_risk", 0)),
        "low_evidence_fallback_suppressed": bool(best.get("low_evidence_fallback_suppressed", False)),
        "fallback_used": bool(best.get("fallback_used", False)),
        "max_observed_depth": max_observed_depth,
        "limits": {
            "max_edges": max_edges,
            "max_supplemented": max_supplemented,
            "max_depth": max_depth,
            "min_quality": min_quality,
        },
        "probes_evaluated": len(evaluated),
        "probe_results": [_probe_summary(item) for item in evaluated],
    }
    issues = _hard_issues(diagnostics)
    warnings = _warnings(diagnostics)
    status = "fail" if issues else "watch" if warnings else "pass"
    return GlobalSpreadingSandbox(
        scope=scope,
        status=status,
        passed=status == "pass",
        diagnostics=diagnostics,
        issues=issues,
        warnings=warnings,
        recommendations=_recommend(issues, warnings),
    )


def _evaluate_probe(memory: Any, query: str, *, scope: str, budgets: list[int] | None) -> dict[str, Any]:
    plan = memory.recall_plan(
        query,
        scope=scope,
        budgets=budgets or [800, 1600],
        include_hot=False,
        include_global=True,
    )
    alternatives = plan.as_dict()["alternatives"]
    best = max(alternatives, key=_probe_rank)
    return {
        "query": query,
        "recommended_budget": int(plan.recommended_budget),
        "best": best,
    }


def _probe_rank(item: dict[str, Any]) -> tuple[bool, int, int, int, int, int, int]:
    return (
        bool(item.get("spreading_activation_used", False)),
        int(item.get("visible_capsules", 0)),
        int(item.get("quality_score", 0)),
        int(item.get("spreading_activation_boosted_count", 0)),
        -int(item.get("spreading_activation_supplemented_count", 0)),
        -int(item.get("graph_activation_edges", 0)),
        -int(item.get("budget", 0)),
    )


def _probe_summary(item: dict[str, Any]) -> dict[str, Any]:
    best = item["best"]
    return {
        "query": item["query"],
        "best_budget": int(best.get("budget", 0)),
        "quality_score": int(best.get("quality_score", 0)),
        "visible_capsules": int(best.get("visible_capsules", 0)),
        "graph_activation_edges": int(best.get("graph_activation_edges", 0)),
        "relation_activation_edges": int(best.get("relation_activation_edges", 0)),
        "spreading_activation_used": bool(best.get("spreading_activation_used", False)),
        "spreading_activation_depth_counts": dict(best.get("spreading_activation_depth_counts", {})),
        "spreading_activation_boosted_count": int(best.get("spreading_activation_boosted_count", 0)),
        "spreading_activation_supplemented_count": int(best.get("spreading_activation_supplemented_count", 0)),
        "capsules_filtered_by_risk": int(best.get("capsules_filtered_by_risk", 0)),
        "low_evidence_fallback_suppressed": bool(best.get("low_evidence_fallback_suppressed", False)),
        "fallback_used": bool(best.get("fallback_used", False)),
    }


def _max_depth(item: dict[str, Any]) -> int:
    depths = item.get("spreading_activation_depth_counts", {})
    if not isinstance(depths, dict) or not depths:
        return 0
    observed = []
    for key, value in depths.items():
        try:
            if int(value) > 0:
                observed.append(int(key))
        except (TypeError, ValueError):
            continue
    return max(observed, default=0)


def _hard_issues(diagnostics: dict[str, Any]) -> list[str]:
    limits = diagnostics["limits"]
    issues: list[str] = []
    if diagnostics["graph_activation_edges"] > limits["max_edges"]:
        issues.append(
            f"global graph fanout exceeded limit: {diagnostics['graph_activation_edges']} > {limits['max_edges']}"
        )
    if diagnostics["spreading_activation_supplemented_count"] > limits["max_supplemented"]:
        issues.append(
            "global graph supplementation exceeded limit: "
            f"{diagnostics['spreading_activation_supplemented_count']} > {limits['max_supplemented']}"
        )
    if diagnostics["max_observed_depth"] > limits["max_depth"]:
        issues.append(
            f"global spreading exceeded depth limit: {diagnostics['max_observed_depth']} > {limits['max_depth']}"
        )
    return issues


def _warnings(diagnostics: dict[str, Any]) -> list[str]:
    limits = diagnostics["limits"]
    warnings: list[str] = []
    if not diagnostics["spreading_activation_used"]:
        warnings.append("global recall did not produce live spreading activation evidence yet")
    if diagnostics["visible_capsules"] <= 0:
        warnings.append("global recall produced no visible evidence capsule")
    if diagnostics["low_evidence_fallback_suppressed"]:
        warnings.append("global recall fell back to no-evidence suppression")
    if diagnostics["quality_score"] < limits["min_quality"]:
        warnings.append(
            f"global recall quality is below the sandbox target: {diagnostics['quality_score']} < {limits['min_quality']}"
        )
    if diagnostics["capsules_filtered_by_risk"] > 0:
        warnings.append(
            "global recall candidate risk filter activated; inspect whether private or instruction-like memories are being considered"
        )
    if diagnostics["fallback_used"]:
        warnings.append("global recall used salience fallback; treat the selected evidence as lower confidence")
    return warnings


def _recommend(issues: list[str], warnings: list[str]) -> list[str]:
    items: list[str] = []
    if any("fanout" in issue or "supplementation" in issue for issue in issues):
        items.append("Tighten global graph term filtering, hub suppression, or per-depth edge limits before enabling broader spreading.")
    if any("depth" in issue for issue in issues):
        items.append("Keep global spreading at bounded two-hop until deeper paths have explicit privacy and relevance witnesses.")
    if any("no visible" in issue or "no-evidence" in issue for issue in issues):
        items.append("Add representative global purpose/identity/relation evidence before trusting global recall expansion.")
    if warnings:
        items.append("Review sandbox warnings before using global memories as always-on context.")
    return items
