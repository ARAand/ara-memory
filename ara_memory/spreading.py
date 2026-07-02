from __future__ import annotations

from typing import Any

from ara_memory.compressors import compact_text, is_search_term


MAX_SPREADING_BOOST = 1.35
LOW_VALUE_GRAPH_TERMS = {
    "ara",
    "memory",
    "current",
    "latest",
    "recent",
    "what",
    "when",
    "where",
    "which",
    "should",
    "about",
    "thing",
    "things",
    "context",
}


def apply_spreading_activation(
    capsules: list[dict[str, Any]],
    edges: list[Any],
    *,
    terms: list[str],
    seed_ids: set[str],
    supplemented_ids: set[str],
) -> dict[str, Any]:
    query_terms = graph_activation_terms(terms)
    if not capsules or not edges or not query_terms:
        _clear_spreading(capsules)
        return {
            "activation_used": False,
            "edge_count": len(edges),
            "boosted_capsule_ids": [],
            "supplemented_capsule_ids": sorted(supplemented_ids),
        }

    capsules_by_id = {str(cap.get("id") or ""): cap for cap in capsules}
    boosted: list[str] = []
    path_counts: dict[str, int] = {}
    seen_edges: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        source_id = str(_edge_value(edge, "source_capsule_id") or "")
        if source_id not in capsules_by_id:
            continue
        edge_key = (
            source_id,
            str(_edge_value(edge, "subject") or "").lower().strip(),
            str(_edge_value(edge, "predicate") or "").lower().strip(),
            str(_edge_value(edge, "object") or "").lower().strip(),
        )
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        cap = capsules_by_id[source_id]
        confidence = _bounded_float(_edge_value(edge, "confidence"), default=0.5)
        overlap = _edge_overlap(edge, query_terms)
        if overlap <= 0.0:
            continue
        seed_factor = 0.45 if source_id in seed_ids else 1.0
        supplement_factor = 1.10 if source_id in supplemented_ids else 1.0
        delta = confidence * (0.15 + 0.85 * overlap) * seed_factor * supplement_factor
        if delta <= 0.0:
            continue
        previous = float(cap.get("spreading_activation_score") or 0.0)
        cap["spreading_activation_score"] = round(min(MAX_SPREADING_BOOST, previous + delta), 4)
        paths = list(cap.get("spreading_activation_paths") or [])
        if len(paths) < 3:
            paths.append(_edge_path(edge))
        cap["spreading_activation_paths"] = paths
        path_counts[source_id] = path_counts.get(source_id, 0) + 1
        if source_id not in boosted:
            boosted.append(source_id)

    for cap in capsules:
        cap.setdefault("spreading_activation_score", 0.0)
        cap.setdefault("spreading_activation_paths", [])
    return {
        "activation_used": bool(boosted),
        "edge_count": len(edges),
        "boosted_capsule_ids": boosted,
        "supplemented_capsule_ids": sorted(supplemented_ids),
        "path_counts": path_counts,
    }


def _clear_spreading(capsules: list[dict[str, Any]]) -> None:
    for cap in capsules:
        cap["spreading_activation_score"] = 0.0
        cap["spreading_activation_paths"] = []


def graph_activation_terms(terms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = str(term or "").lower().strip()
        if not normalized or normalized in seen:
            continue
        if normalized in LOW_VALUE_GRAPH_TERMS:
            continue
        if len(normalized) < 4 and normalized not in {"risk", "safe"}:
            continue
        if not is_search_term(normalized):
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _edge_overlap(edge: Any, terms: list[str]) -> float:
    text = " ".join(
        str(_edge_value(edge, key) or "").lower()
        for key in ("subject", "predicate", "object")
    )
    if not text:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return min(1.0, hits / max(1, min(len(terms), 3)))


def _edge_path(edge: Any) -> str:
    subject = compact_text(str(_edge_value(edge, "subject") or ""), limit=72)
    predicate = compact_text(str(_edge_value(edge, "predicate") or ""), limit=48)
    object_ = compact_text(str(_edge_value(edge, "object") or ""), limit=72)
    return f"{subject} {predicate} {object_}".strip()


def _edge_value(edge: Any, key: str) -> Any:
    if isinstance(edge, dict):
        return edge.get(key)
    try:
        return edge[key]
    except (IndexError, KeyError, TypeError):
        return None


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
