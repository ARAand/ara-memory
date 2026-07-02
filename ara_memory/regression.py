from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara_memory.core import AraMemory
from ara_memory.storage import row_to_capsule


SOURCE_EVENT_ID_DETAIL_LIMIT = 64


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
        memory=memory,
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
    selected_source_detail = _source_event_detail_for_capsules(memory, selected_ids)
    visible_source_detail = _source_event_detail_for_capsules(memory, visible_ids)
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
            "selected_source_event_ids": selected_source_detail["ids"],
            "selected_source_event_count": selected_source_detail["count"],
            "selected_source_event_digest": selected_source_detail["digest"],
            "selected_source_event_ids_truncated": selected_source_detail["truncated"],
            "visible_source_event_ids": visible_source_detail["ids"],
            "visible_source_event_count": visible_source_detail["count"],
            "visible_source_event_digest": visible_source_detail["digest"],
            "visible_source_event_ids_truncated": visible_source_detail["truncated"],
            "include_global": case.include_global,
            "include_hot": case.include_hot,
        },
    )


def _compare_to_baseline(
    *,
    memory: AraMemory,
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
        overlap_info = _baseline_overlap_info(memory, prior_details, result.details)
        overlap = float(overlap_info["overlap"])
        overlap_basis = str(overlap_info["basis"])
        capsule_overlap = float(overlap_info["capsule_id_overlap"])
        source_overlap = overlap_info["source_event_overlap"]
        source_jaccard = overlap_info["source_event_jaccard"]
        source_guard = bool(overlap_info["source_event_guard"])
        source_validated = bool(overlap_info["source_event_validated"])
        source_can_substitute = (
            source_guard
            and source_validated
            and source_overlap is not None
            and float(source_overlap) >= min_overlap
        )
        failures = []
        if prior_passed and not result.passed:
            failures.append("previously_passed_now_failed")
        if prior_tokens > 0 and current_tokens > token_limit:
            failures.append("token_growth_exceeded")
        if overlap_info["capsule_prior_count"] and overlap_info["capsule_current_count"]:
            if capsule_overlap < min_overlap and not source_can_substitute:
                if overlap_info["capsule_basis"] == "visible_capsule_ids":
                    failures.append("visible_capsule_overlap_below_threshold")
                else:
                    failures.append("selected_capsule_overlap_below_threshold")
        if source_guard:
            if not source_validated:
                failures.append(_source_event_failure_name(str(overlap_info["source_event_basis"]), "incomplete"))
            elif source_overlap is not None and float(source_overlap) < min_overlap:
                failures.append(
                    _source_event_failure_name(str(overlap_info["source_event_basis"]), "below_threshold")
                )
            elif source_overlap is None:
                failures.append(_source_event_failure_name(str(overlap_info["source_event_basis"]), "missing"))
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
                    "capsule_id_overlap": capsule_overlap,
                    "source_event_overlap": source_overlap,
                    "source_event_jaccard": source_jaccard,
                    "source_event_guard": source_guard,
                    "source_event_validated": source_validated,
                    "source_event_digest_match": overlap_info["source_event_digest_match"],
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


def _baseline_overlap_info(
    memory: AraMemory,
    prior_details: dict[str, Any],
    current_details: dict[str, Any],
) -> dict[str, Any]:
    prior_visible = [str(item) for item in prior_details.get("visible_capsule_ids", [])]
    current_visible = [str(item) for item in current_details.get("visible_capsule_ids", [])]
    if prior_visible and current_visible:
        return _overlap_info_for_kind(
            memory,
            prior_details,
            current_details,
            id_field="visible_capsule_ids",
            source_field="visible_source_event_ids",
        )
    return _overlap_info_for_kind(
        memory,
        prior_details,
        current_details,
        id_field="selected_capsule_ids",
        source_field="selected_source_event_ids",
    )


def _overlap_info_for_kind(
    memory: AraMemory,
    prior_details: dict[str, Any],
    current_details: dict[str, Any],
    *,
    id_field: str,
    source_field: str,
) -> dict[str, Any]:
    prior_ids = {str(item) for item in prior_details.get(id_field, [])}
    current_ids = {str(item) for item in current_details.get(id_field, [])}
    id_overlap = _overlap_ratio(prior_ids, current_ids)
    prior_source = _source_event_info_from_details_or_store(
        memory,
        details=prior_details,
        source_field=source_field,
        capsule_ids=prior_ids,
    )
    current_source = _source_event_info_from_details_or_store(
        memory,
        details=current_details,
        source_field=source_field,
        capsule_ids=current_ids,
    )
    prior_source_ids = prior_source["ids"]
    current_source_ids = current_source["ids"]
    source_jaccard = (
        _overlap_ratio(prior_source_ids, current_source_ids)
        if prior_source_ids and current_source_ids
        else None
    )
    source_overlap = (
        _coverage_ratio(prior_source_ids, current_source_ids)
        if prior_source_ids and current_source_ids
        else None
    )
    source_digest_match = None
    if prior_source["digest"] and current_source["digest"]:
        source_digest_match = prior_source["digest"] == current_source["digest"]
        if source_digest_match and not (prior_source["complete"] and current_source["complete"]):
            source_jaccard = 1.0
            source_overlap = 1.0
    source_guard = bool(
        prior_source["ids"]
        or prior_source["count"]
        or prior_source["digest"]
        or source_field in prior_details
    )
    source_validated = bool(
        not source_guard
        or (prior_source["complete"] and current_source["complete"])
        or source_digest_match is True
    )
    source_is_more_specific = source_overlap is not None and source_overlap != id_overlap
    if source_is_more_specific:
        return {
            "prior_ids": prior_source_ids,
            "current_ids": current_source_ids,
            "basis": source_field,
            "overlap": source_overlap,
            "capsule_id_overlap": id_overlap,
            "source_event_overlap": source_overlap,
            "capsule_basis": id_field,
            "source_event_basis": source_field,
            "capsule_prior_count": len(prior_ids),
            "capsule_current_count": len(current_ids),
            "source_event_guard": source_guard,
            "source_event_validated": source_validated,
            "source_event_digest_match": source_digest_match,
            "source_event_jaccard": source_jaccard,
        }
    return {
        "prior_ids": prior_ids,
        "current_ids": current_ids,
        "basis": id_field,
        "overlap": id_overlap,
        "capsule_id_overlap": id_overlap,
        "source_event_overlap": source_overlap,
        "capsule_basis": id_field,
        "source_event_basis": source_field,
        "capsule_prior_count": len(prior_ids),
        "capsule_current_count": len(current_ids),
        "source_event_guard": source_guard,
        "source_event_validated": source_validated,
        "source_event_digest_match": source_digest_match,
        "source_event_jaccard": source_jaccard,
    }


def _source_event_info_from_details_or_store(
    memory: AraMemory,
    *,
    details: dict[str, Any],
    source_field: str,
    capsule_ids: set[str],
) -> dict[str, Any]:
    prefix = source_field.removesuffix("_ids")
    source_ids = {str(item) for item in details.get(source_field, []) if str(item)}
    truncated = bool(details.get(f"{source_field}_truncated", False))
    source_count = _safe_int(details.get(f"{prefix}_count"), default=len(source_ids))
    source_digest = details.get(f"{prefix}_digest")
    if not isinstance(source_digest, str) or not source_digest:
        source_digest = _source_event_ids_digest(source_ids)
    if source_ids and not truncated:
        return {
            "ids": source_ids,
            "count": source_count,
            "digest": source_digest,
            "complete": True,
        }
    store_source_ids = set(_source_event_ids_for_capsules(memory, sorted(capsule_ids)))
    if store_source_ids:
        return {
            "ids": store_source_ids,
            "count": len(store_source_ids),
            "digest": _source_event_ids_digest(store_source_ids),
            "complete": True,
        }
    return {
        "ids": source_ids,
        "count": source_count,
        "digest": source_digest,
        "complete": not truncated,
    }


def _source_event_detail_for_capsules(memory: AraMemory, capsule_ids: list[str]) -> dict[str, Any]:
    source_ids = _source_event_ids_for_capsules(memory, capsule_ids)
    return {
        "ids": source_ids[:SOURCE_EVENT_ID_DETAIL_LIMIT],
        "count": len(source_ids),
        "digest": _source_event_ids_digest(source_ids),
        "truncated": len(source_ids) > SOURCE_EVENT_ID_DETAIL_LIMIT,
    }


def _source_event_ids_for_capsules(memory: AraMemory, capsule_ids: list[str]) -> list[str]:
    source_ids: set[str] = set()
    for capsule_id in capsule_ids:
        row = memory.store.get_capsule(capsule_id)
        if row is None:
            continue
        try:
            capsule = row_to_capsule(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        source_ids.update(str(item) for item in capsule.get("source_event_ids", []) if str(item))
    source_ids.update(memory.store.provenance_witness_source_event_ids_for_capsules(capsule_ids))
    return sorted(source_ids)


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _coverage_ratio(prior: set[str], current: set[str]) -> float:
    if not prior:
        return 1.0
    if not current:
        return 0.0
    return len(prior & current) / len(prior)


def _source_event_ids_digest(source_ids: set[str] | list[str]) -> str | None:
    normalized = sorted(str(item) for item in source_ids if str(item))
    if not normalized:
        return None
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_event_failure_name(source_field: str, reason: str) -> str:
    prefix = "visible" if source_field.startswith("visible_") else "selected"
    if reason == "below_threshold":
        return f"{prefix}_source_event_overlap_below_threshold"
    return f"{prefix}_source_event_overlap_{reason}"


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
