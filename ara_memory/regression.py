from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara_memory.core import AraMemory


@dataclass(slots=True)
class RecallRegressionCase:
    name: str
    query: str
    scope: str = "global"
    expected_terms: list[str] = field(default_factory=list)
    expected_any_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    budget: int = 1200
    include_global: bool = True
    include_hot: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RecallRegressionCase":
        return cls(
            name=str(payload["name"]),
            query=str(payload["query"]),
            scope=str(payload.get("scope", "global")),
            expected_terms=[str(term) for term in payload.get("expected_terms", [])],
            expected_any_terms=[str(term) for term in payload.get("expected_any_terms", [])],
            forbidden_terms=[str(term) for term in payload.get("forbidden_terms", [])],
            budget=int(payload.get("budget", 1200)),
            include_global=bool(payload.get("include_global", True)),
            include_hot=bool(payload.get("include_hot", False)),
        )


@dataclass(slots=True)
class RecallRegressionCaseResult:
    name: str
    passed: bool
    details: dict[str, Any]


@dataclass(slots=True)
class RecallRegressionReport:
    results: list[RecallRegressionCaseResult]
    baseline_comparison: list[RecallRegressionCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results) and all(
            result.passed for result in self.baseline_comparison
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "cases": [
                {"name": result.name, "passed": result.passed, "details": result.details}
                for result in self.results
            ],
            "baseline_comparison": [
                {"name": result.name, "passed": result.passed, "details": result.details}
                for result in self.baseline_comparison
            ],
        }


def load_recall_regression_cases(path: Path) -> list[RecallRegressionCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Recall regression manifest must contain a non-empty 'cases' list.")
    cases = []
    for item in raw_cases:
        if not isinstance(item, dict):
            raise ValueError("Each recall regression case must be a JSON object.")
        cases.append(RecallRegressionCase.from_dict(item))
    return cases


def load_recall_regression_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Recall regression baseline must be a JSON object.")
    return payload


def run_recall_regression(
    memory: AraMemory,
    cases: list[RecallRegressionCase],
    *,
    baseline: dict[str, Any] | None = None,
    max_token_growth: float = 0.25,
    min_overlap: float = 0.35,
) -> RecallRegressionReport:
    results = [_run_case(memory, case) for case in cases]
    comparisons = _compare_to_baseline(
        current=results,
        baseline=baseline,
        max_token_growth=max_token_growth,
        min_overlap=min_overlap,
    )
    return RecallRegressionReport(results=results, baseline_comparison=comparisons)


def write_recall_regression_baseline(report: RecallRegressionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _run_case(memory: AraMemory, case: RecallRegressionCase) -> RecallRegressionCaseResult:
    result = memory.recall_result(
        case.query,
        scope=case.scope,
        budget=case.budget,
        include_global=case.include_global,
        include_hot=case.include_hot,
    )
    evidence_text = _regression_evidence_text(result.pack).lower()
    full_text = result.pack.lower()
    expected_hits = {term: term.lower() in evidence_text for term in case.expected_terms}
    expected_any_hits = {term: term.lower() in evidence_text for term in case.expected_any_terms}
    forbidden_hits = {term: term.lower() in full_text for term in case.forbidden_terms}
    tokens = int(result.diagnostics["estimated_tokens_after"])
    selected_ids = [str(item) for item in result.diagnostics.get("selected_capsule_ids", [])]
    visible_ids = [str(item) for item in result.diagnostics.get("visible_capsule_ids", [])]
    passed = (
        all(expected_hits.values())
        and (not expected_any_hits or any(expected_any_hits.values()))
        and not any(forbidden_hits.values())
        and tokens <= case.budget
        and len(visible_ids) > 0
    )
    return RecallRegressionCaseResult(
        name=case.name,
        passed=passed,
        details={
            "query": case.query,
            "scope": case.scope,
            "budget": case.budget,
            "estimated_tokens": tokens,
            "expected_hits": expected_hits,
            "expected_any_hits": expected_any_hits,
            "forbidden_hits": forbidden_hits,
            "capsules_selected": len(selected_ids),
            "capsules_visible": len(visible_ids),
            "selected_capsule_ids": selected_ids,
            "visible_capsule_ids": visible_ids,
            "include_global": case.include_global,
            "include_hot": case.include_hot,
        },
    )


def _compare_to_baseline(
    *,
    current: list[RecallRegressionCaseResult],
    baseline: dict[str, Any] | None,
    max_token_growth: float,
    min_overlap: float,
) -> list[RecallRegressionCaseResult]:
    if baseline is None:
        return []
    previous = {
        str(item.get("name")): item
        for item in baseline.get("cases", [])
        if isinstance(item, dict) and item.get("name") is not None
    }
    comparisons: list[RecallRegressionCaseResult] = []
    for result in current:
        prior = previous.get(result.name)
        if not prior:
            comparisons.append(
                RecallRegressionCaseResult(
                    name=result.name,
                    passed=False,
                    details={"reason": "missing_baseline_case"},
                )
            )
            continue
        prior_details = prior.get("details", {}) if isinstance(prior.get("details"), dict) else {}
        prior_passed = bool(prior.get("passed", False))
        prior_tokens = int(prior_details.get("estimated_tokens", 0) or 0)
        current_tokens = int(result.details.get("estimated_tokens", 0) or 0)
        token_limit = max(prior_tokens + 100, int(prior_tokens * (1.0 + max_token_growth)))
        prior_ids, current_ids, overlap_basis = _baseline_overlap_ids(prior_details, result.details)
        overlap = _overlap_ratio(prior_ids, current_ids)
        failures = []
        if prior_passed and not result.passed:
            failures.append("previously_passed_now_failed")
        if prior_tokens > 0 and current_tokens > token_limit:
            failures.append("token_growth_exceeded")
        if prior_ids and current_ids and overlap < min_overlap:
            if overlap_basis == "visible_capsule_ids":
                failures.append("visible_capsule_overlap_below_threshold")
            else:
                failures.append("selected_capsule_overlap_below_threshold")
        comparisons.append(
            RecallRegressionCaseResult(
                name=result.name,
                passed=not failures,
                details={
                    "failures": failures,
                    "baseline_tokens": prior_tokens,
                    "current_tokens": current_tokens,
                    "token_limit": token_limit,
                    "selected_overlap": overlap,
                    "evidence_overlap": overlap,
                    "overlap_basis": overlap_basis,
                    "min_overlap": min_overlap,
                    "max_token_growth": max_token_growth,
                },
            )
        )
    return comparisons


def _regression_evidence_text(pack: str) -> str:
    excluded_sections = {
        "Memory Safety Boundary",
        "Hot Memory",
        "Matched Tags",
        "Temporal Graph Hints",
    }
    lines: list[str] = []
    current_section: str | None = None
    for line in pack.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            continue
        if stripped.startswith("#") or stripped.startswith(("Query:", "Scope:")):
            continue
        if current_section in excluded_sections:
            continue
        if stripped in {"- None found.", "- None."}:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _baseline_overlap_ids(
    prior_details: dict[str, Any],
    current_details: dict[str, Any],
) -> tuple[set[str], set[str], str]:
    prior_visible = [str(item) for item in prior_details.get("visible_capsule_ids", [])]
    current_visible = [str(item) for item in current_details.get("visible_capsule_ids", [])]
    if prior_visible and current_visible:
        return set(prior_visible), set(current_visible), "visible_capsule_ids"
    return (
        {str(item) for item in prior_details.get("selected_capsule_ids", [])},
        {str(item) for item in current_details.get("selected_capsule_ids", [])},
        "selected_capsule_ids",
    )


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
