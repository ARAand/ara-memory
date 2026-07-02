from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

from ara_memory.cold_stewardship import COLD_TIER_POLICIES, _cold_tier, _title_pattern
from ara_memory.compressors import compact_text, estimate_tokens, extract_keywords
from ara_memory.models import MemoryStatus
from ara_memory.retention import COLD_STATUSES
from ara_memory.risk import redact_memory_tags, redact_sensitive_text
from ara_memory.storage import MemoryStore


ACTIVE_STATUSES = {
    MemoryStatus.CANDIDATE.value,
    MemoryStatus.STABLE.value,
}


@dataclass(slots=True)
class ColdMapGroup:
    tier: str
    status: str
    kind: str
    pattern: str
    count: int
    score: float
    matched_terms: list[str]
    top_tags: list[str]
    source_events: int
    protected_source_events: int
    prunable_source_events: int
    source_event_digest: str | None
    oldest_updated_at: str
    newest_updated_at: str
    examples: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "status": self.status,
            "kind": self.kind,
            "pattern": self.pattern,
            "count": self.count,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "top_tags": self.top_tags,
            "source_events": self.source_events,
            "protected_source_events": self.protected_source_events,
            "prunable_source_events": self.prunable_source_events,
            "source_event_digest": self.source_event_digest,
            "oldest_updated_at": self.oldest_updated_at,
            "newest_updated_at": self.newest_updated_at,
            "examples": self.examples,
        }


@dataclass(slots=True)
class ColdMemoryMap:
    scope: str | None
    query: str
    terms: list[str]
    status: str
    totals: dict[str, Any]
    groups: list[ColdMapGroup]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "terms": self.terms,
            "status": self.status,
            "totals": self.totals,
            "groups": [group.as_dict() for group in self.groups],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Cold Memory Map: {scope}", f"status: {self.status}"]
        if self.query:
            lines.append(f"query: {redact_sensitive_text(self.query)}")
        lines.append(
            "totals: "
            f"cold={self.totals['cold_capsules']}, matched={self.totals['matched_capsules']}, "
            f"raw_tokens={self.totals['matched_raw_tokens']}, "
            f"map_tokens={self.totals['estimated_map_tokens']}, "
            f"reduction={self.totals['token_reduction_ratio']:.1f}x"
        )
        lines.append("## Boundary")
        lines.append(
            "- This is a navigation map for distant memory. It does not promote, render, or restore cold capsule bodies."
        )
        lines.append("## Groups")
        if not self.groups:
            lines.append("- None")
        for group in self.groups:
            terms = ", ".join(group.matched_terms) if group.matched_terms else "-"
            tags = ", ".join(group.top_tags) if group.top_tags else "-"
            lines.append(
                f"- [{group.tier}] {group.status}/{group.kind}/{group.pattern}: "
                f"count={group.count}, score={group.score:.2f}, "
                f"sources=count={group.source_events},digest={group.source_event_digest or 'none'}, "
                f"protected={group.protected_source_events}, prunable={group.prunable_source_events}, "
                f"terms={terms}, tags={tags}"
            )
            for example in group.examples:
                lines.append(f"  example: {example['id']} {example['title']} ({example['updated_at']})")
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


class ColdMemoryMapper:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        query: str = "",
        group_limit: int = 10,
        examples_per_group: int = 2,
        budget: int = 900,
    ) -> ColdMemoryMap:
        self.store.init()
        terms = extract_keywords(query, limit=10) if query else []
        active_event_ids = self.store.source_event_ids_for_statuses(ACTIVE_STATUSES, scope=scope)
        rows = [_annotate_row(row, active_event_ids=active_event_ids) for row in _cold_rows(self.store, scope=scope)]
        scored_rows = [
            (score, row)
            for row in rows
            for score in [_row_score(row, terms)]
            if not terms or score > 0.0
        ]
        matched_rows = [row for _, row in scored_rows]
        candidate_groups = _groups(
            scored_rows,
            terms=terms,
            active_event_ids=active_event_ids,
            group_limit=group_limit,
            examples_per_group=examples_per_group,
        )
        groups, map_tokens = _fit_groups_to_budget(
            candidate_groups,
            scope=scope,
            query=query,
            cold_count=len(rows),
            matched_count=len(matched_rows),
            raw_tokens=sum(int(row.get("estimated_tokens") or 0) for row in matched_rows),
            budget=budget,
        )
        totals = _totals(
            rows=rows,
            matched_rows=matched_rows,
            groups=groups,
            budget=budget,
            map_tokens=map_tokens,
        )
        status = "pass" if groups or not terms else "watch"
        return ColdMemoryMap(
            scope=scope,
            query=query,
            terms=terms,
            status=status,
            totals=totals,
            groups=groups,
            recommendations=_recommend(query=query, totals=totals, groups=groups),
        )


def _cold_rows(store: MemoryStore, *, scope: str | None) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in COLD_STATUSES)
    clauses = [f"status IN ({placeholders})"]
    args: list[Any] = [*COLD_STATUSES]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    with store.session() as conn:
        rows = conn.execute(
            f"""
            SELECT id, kind, status, title, body, updated_at, source_event_ids_json, tags_json
            FROM capsules
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at ASC, id ASC
            """,
            args,
        )
        return [_row_to_cold(row) for row in rows]


def _row_to_cold(row: Any) -> dict[str, Any]:
    try:
        source_event_ids = json.loads(row["source_event_ids_json"])
    except (TypeError, json.JSONDecodeError):
        source_event_ids = []
    try:
        tags = json.loads(row["tags_json"])
    except (TypeError, json.JSONDecodeError):
        tags = []
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "title": row["title"],
        "body": row["body"],
        "updated_at": row["updated_at"],
        "source_event_ids": [str(event_id) for event_id in source_event_ids if str(event_id)],
        "tags": [str(tag) for tag in tags if str(tag)],
    }


def _annotate_row(row: dict[str, Any], *, active_event_ids: set[str]) -> dict[str, Any]:
    source_ids = set(row["source_event_ids"])
    annotated = dict(row)
    annotated["source_role"] = "active-linked" if source_ids.intersection(active_event_ids) else "cold-only"
    annotated["tier"] = _cold_tier(
        str(row["status"]),
        has_active_provenance=annotated["source_role"] == "active-linked",
    )
    annotated["estimated_tokens"] = estimate_tokens(f"{row['title']}\n{row.get('body') or ''}")
    return annotated


def _row_score(row: dict[str, Any], terms: list[str]) -> float:
    if not terms:
        return 1.0
    score = 0.0
    title = str(row.get("title") or "").lower()
    body = str(row.get("body") or "").lower()
    tags = " ".join(str(tag).lower() for tag in row.get("tags", []))
    for term in terms:
        stem = _stem(term)
        if not stem:
            continue
        if stem in title:
            score += 2.5
        if stem in tags:
            score += 2.0
        if stem in body:
            score += 1.0
    if score <= 0.0:
        return 0.0
    if row.get("tier") == "evidence":
        score += 0.3
    elif row.get("tier") == "archive":
        score += 0.1
    return score


def _groups(
    scored_rows: list[tuple[float, dict[str, Any]]],
    *,
    terms: list[str],
    active_event_ids: set[str],
    group_limit: int,
    examples_per_group: int,
) -> list[ColdMapGroup]:
    buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for score, row in scored_rows:
        key = (row["tier"], row["status"], row["kind"], _title_pattern(row["title"]))
        bucket = buckets.setdefault(key, {"rows": [], "score": 0.0, "source_event_ids": set()})
        bucket["rows"].append(row)
        bucket["score"] += score
        bucket["source_event_ids"].update(row["source_event_ids"])

    groups: list[ColdMapGroup] = []
    for (tier, status, kind, pattern), bucket in buckets.items():
        rows = bucket["rows"]
        source_event_ids = sorted(bucket["source_event_ids"])
        protected = sorted(set(source_event_ids).intersection(active_event_ids))
        prunable = sorted(set(source_event_ids) - set(protected))
        updated = [row["updated_at"] for row in rows]
        matched_terms = _matched_terms(rows, terms)
        tags = _top_tags(rows)
        examples = [
            {
                "id": row["id"],
                "title": compact_text(redact_sensitive_text(str(row["title"])), limit=140),
                "updated_at": row["updated_at"],
                "tier": row["tier"],
                "source_role": row["source_role"],
            }
            for row in rows[: max(0, examples_per_group)]
        ]
        groups.append(
            ColdMapGroup(
                tier=tier,
                status=status,
                kind=kind,
                pattern=redact_sensitive_text(pattern),
                count=len(rows),
                score=round(float(bucket["score"]), 3),
                matched_terms=matched_terms,
                top_tags=tags,
                source_events=len(source_event_ids),
                protected_source_events=len(protected),
                prunable_source_events=len(prunable),
                source_event_digest=_source_digest(source_event_ids),
                oldest_updated_at=min(updated) if updated else "",
                newest_updated_at=max(updated) if updated else "",
                examples=examples,
            )
        )
    return sorted(
        groups,
        key=lambda group: (
            -group.score,
            -group.count,
            -group.protected_source_events,
            -group.prunable_source_events,
            group.tier,
            group.kind,
            group.pattern,
        ),
    )[: max(0, group_limit)]


def _totals(
    *,
    rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    groups: list[ColdMapGroup],
    budget: int,
    map_tokens: int,
) -> dict[str, Any]:
    raw_tokens = sum(int(row.get("estimated_tokens") or 0) for row in matched_rows)
    return {
        "cold_capsules": len(rows),
        "matched_capsules": len(matched_rows),
        "groups": len(groups),
        "matched_raw_tokens": raw_tokens,
        "estimated_map_tokens": map_tokens,
        "budget_tokens": budget,
        "within_budget": map_tokens <= budget,
        "token_reduction_ratio": round(raw_tokens / max(1, map_tokens), 2) if raw_tokens else 0.0,
        "evidence_capsules": sum(1 for row in rows if row.get("tier") == "evidence"),
        "archive_capsules": sum(1 for row in rows if row.get("tier") == "archive"),
        "reject_capsules": sum(1 for row in rows if row.get("tier") == "reject"),
    }


def _fit_groups_to_budget(
    groups: list[ColdMapGroup],
    *,
    scope: str | None,
    query: str,
    cold_count: int,
    matched_count: int,
    raw_tokens: int,
    budget: int,
) -> tuple[list[ColdMapGroup], int]:
    if budget <= 0:
        return [], _estimate_map_tokens(
            scope=scope,
            query=query,
            groups=[],
            cold_count=cold_count,
            matched_count=matched_count,
            raw_tokens=raw_tokens,
        )
    variants: list[list[ColdMapGroup]] = [
        groups,
        [_copy_group_with_examples(group, 1) for group in groups],
        [_copy_group_with_examples(group, 0) for group in groups],
    ]
    best_groups = variants[-1]
    best_tokens = _estimate_map_tokens(
        scope=scope,
        query=query,
        groups=best_groups,
        cold_count=cold_count,
        matched_count=matched_count,
        raw_tokens=raw_tokens,
    )
    for variant in variants:
        current = list(variant)
        while current:
            estimated = _estimate_map_tokens(
                scope=scope,
                query=query,
                groups=current,
                cold_count=cold_count,
                matched_count=matched_count,
                raw_tokens=raw_tokens,
            )
            if estimated <= budget:
                return current, estimated
            if estimated < best_tokens:
                best_groups, best_tokens = list(current), estimated
            current = current[:-1]
    empty_tokens = _estimate_map_tokens(
        scope=scope,
        query=query,
        groups=[],
        cold_count=cold_count,
        matched_count=matched_count,
        raw_tokens=raw_tokens,
    )
    if empty_tokens <= budget or empty_tokens < best_tokens:
        return [], empty_tokens
    return best_groups, best_tokens


def _copy_group_with_examples(group: ColdMapGroup, limit: int) -> ColdMapGroup:
    return replace(group, examples=list(group.examples[: max(0, limit)]))


def _estimate_map_tokens(
    *,
    scope: str | None,
    query: str,
    groups: list[ColdMapGroup],
    cold_count: int,
    matched_count: int,
    raw_tokens: int,
) -> int:
    scope_text = scope or "all"
    lines = [
        f"# Ara Cold Memory Map: {scope_text}",
        "status: pass",
    ]
    if query:
        lines.append(f"query: {redact_sensitive_text(query)}")
    lines.append(
        "totals: "
        f"cold={cold_count}, matched={matched_count}, raw_tokens={raw_tokens}, "
        "map_tokens=0000, reduction=0.0x"
    )
    lines.extend(
        [
            "## Boundary",
            "- This is a navigation map for distant memory. It does not promote, render, or restore cold capsule bodies.",
            "## Groups",
        ]
    )
    if not groups:
        lines.append("- None")
    for group in groups:
        terms = ", ".join(group.matched_terms) if group.matched_terms else "-"
        tags = ", ".join(group.top_tags) if group.top_tags else "-"
        lines.append(
            f"- [{group.tier}] {group.status}/{group.kind}/{group.pattern}: "
            f"count={group.count}, score={group.score:.2f}, "
            f"sources=count={group.source_events},digest={group.source_event_digest or 'none'}, "
            f"protected={group.protected_source_events}, prunable={group.prunable_source_events}, "
            f"terms={terms}, tags={tags}"
        )
        for example in group.examples:
            lines.append(f"  example: {example['id']} {example['title']} ({example['updated_at']})")
    lines.extend(
        [
            "## Recommendations",
            "- Use this map to choose a narrow cold-export, prune-plan, or manual audit path instead of rendering cold capsule bodies.",
        ]
    )
    return estimate_tokens("\n".join(lines))


def _matched_terms(rows: list[dict[str, Any]], terms: list[str], *, limit: int = 8) -> list[str]:
    text = " ".join(
        f"{row.get('title', '')} {row.get('body', '')} {' '.join(row.get('tags', []))}".lower()
        for row in rows
    )
    candidates = terms or extract_keywords(text, limit=limit * 4)
    out: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        stem = _stem(term)
        if not stem or stem in seen:
            continue
        if stem in text:
            out.append(redact_sensitive_text(term))
            seen.add(stem)
        if len(out) >= limit:
            break
    return out


def _top_tags(rows: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    tags = Counter(
        tag
        for row in rows
        for tag in redact_memory_tags(row.get("tags", []))
        if tag and not str(tag).startswith("artifact:")
    )
    return [str(tag) for tag, _ in tags.most_common(limit)]


def _source_digest(source_event_ids: list[str]) -> str | None:
    unique = sorted(set(source_event_ids))
    if not unique:
        return None
    return sha256("\n".join(unique).encode("utf-8")).hexdigest()[:12]


def _recommend(query: str, totals: dict[str, Any], groups: list[ColdMapGroup]) -> list[str]:
    if query and not groups:
        return [
            "No distant-memory group matched the query; use active recall or inspect current files instead of broad cold rereads."
        ]
    recommendations = [
        "Use this map to choose a narrow cold-export, prune-plan, or manual audit path instead of rendering cold capsule bodies."
    ]
    if totals["matched_raw_tokens"] and totals["estimated_map_tokens"]:
        recommendations.append(
            f"Matched cold bodies are about {totals['matched_raw_tokens']} tokens; the map is about {totals['estimated_map_tokens']} tokens."
        )
    if any(group.tier == "evidence" for group in groups):
        recommendations.append("Evidence-tier groups still share provenance with active memory; preserve source events.")
    if any(group.tier == "archive" for group in groups):
        recommendations.append("Archive-tier groups need signed cold export verification before destructive cleanup.")
    if any(group.tier == "reject" for group in groups):
        recommendations.append("Reject-tier groups are audit evidence; inspect before discarding.")
    if not totals.get("within_budget", True):
        recommendations.append("Map exceeds the requested budget; rerun with a narrower query or lower --group-limit.")
    return recommendations


def _stem(term: str) -> str:
    normalized = str(term).lower().strip()
    if len(normalized) > 5 and normalized.endswith("ing"):
        normalized = normalized[:-3]
    elif len(normalized) > 4 and normalized.endswith(("ed", "es")):
        normalized = normalized[:-2]
    elif len(normalized) > 4 and normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized[:6] if len(normalized) > 6 else normalized
