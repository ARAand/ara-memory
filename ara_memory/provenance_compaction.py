from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ara_memory.models import MemoryStatus
from ara_memory.retention import COLD_STATUSES
from ara_memory.storage import MemoryStore


ACTIVE_STATUSES = {MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value}
CONFIRMATION = "COMPACT PROVENANCE"


@dataclass(slots=True)
class ProvenanceCompactionItem:
    capsule_id: str
    status: str
    kind: str
    title: str
    scope: str
    expected_source_event_ids: list[str]
    source_event_count: int
    proposed_source_event_count: int
    pinned_cold_events: int
    released_source_events: int
    globally_unpinned_cold_events: int
    retained_source_event_ids: list[str]
    released_source_event_ids: list[str]
    applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "status": self.status,
            "kind": self.kind,
            "title": self.title,
            "scope": self.scope,
            "source_event_count": self.source_event_count,
            "proposed_source_event_count": self.proposed_source_event_count,
            "pinned_cold_events": self.pinned_cold_events,
            "released_source_events": self.released_source_events,
            "globally_unpinned_cold_events": self.globally_unpinned_cold_events,
            "retained_source_event_ids": self.retained_source_event_ids,
            "released_source_event_ids": self.released_source_event_ids,
            "applied": self.applied,
        }


@dataclass(slots=True)
class ProvenanceCompactionReport:
    scope: str | None
    dry_run: bool
    passed: bool
    blocked_reason: str
    keep_events: int
    min_pinned_events: int
    summary_only: bool
    totals: dict[str, Any]
    items: list[ProvenanceCompactionItem]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "passed": self.passed,
            "blocked_reason": self.blocked_reason,
            "keep_events": self.keep_events,
            "min_pinned_events": self.min_pinned_events,
            "summary_only": self.summary_only,
            "totals": self.totals,
            "items": [item.as_dict() for item in self.items],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Provenance Compaction: {scope}",
            f"dry_run: {self.dry_run}",
            f"passed: {self.passed}",
        ]
        if self.blocked_reason:
            lines.append(f"blocked: {self.blocked_reason}")
        lines.append(
            "totals: "
            f"candidates={self.totals['candidate_capsules']}, "
            f"release_links={self.totals['released_source_events']}, "
            f"globally_unpinned_cold_events={self.totals['globally_unpinned_cold_events']}, "
            f"applied={self.totals['applied_capsules']}"
        )
        if not self.items:
            lines.append("- No eligible active provenance pins.")
        for item in self.items:
            lines.append(
                f"- {item.capsule_id} [{item.kind}/{item.status}] "
                f"{item.source_event_count}->{item.proposed_source_event_count}; "
                f"pinned_cold={item.pinned_cold_events}, "
                f"release_links={item.released_source_events}, "
                f"globally_unpinned={item.globally_unpinned_cold_events}; "
                f"{item.title}"
            )
        lines.append("## Recommendations")
        for recommendation in self.recommendations:
            lines.append(f"- {recommendation}")
        return "\n".join(lines)


class ProvenanceCompactor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        keep_events: int = 8,
        min_pinned_events: int = 20,
        limit: int = 20,
        summary_only: bool = True,
        dry_run: bool = True,
        confirm: str = "",
    ) -> ProvenanceCompactionReport:
        if keep_events <= 0:
            raise ValueError("keep_events must be positive.")
        if min_pinned_events <= 0:
            raise ValueError("min_pinned_events must be positive.")
        if limit <= 0:
            raise ValueError("limit must be positive.")

        self.store.init()
        cold_event_ids = _cold_source_event_ids(self.store, scope=scope)
        candidate_rows = _active_rows(self.store, scope=scope, summary_only=summary_only)
        active_counts = _active_event_counts(_active_rows(self.store, scope=scope, summary_only=False))
        event_created_at = _event_created_at_lookup(self.store, candidate_rows)
        items = _plan_items(
            candidate_rows,
            cold_event_ids=cold_event_ids,
            active_counts=active_counts,
            event_created_at=event_created_at,
            keep_events=keep_events,
            min_pinned_events=min_pinned_events,
            limit=limit,
        )
        blocked_reason = ""
        passed = True
        if not dry_run and confirm != CONFIRMATION:
            blocked_reason = f'apply requires --confirm "{CONFIRMATION}"'
            passed = False
        elif not dry_run:
            applied = self.store.update_capsule_source_events_batch(
                [
                    {
                        "capsule_id": item.capsule_id,
                        "source_event_ids": item.retained_source_event_ids,
                        "expected_source_event_ids": item.expected_source_event_ids,
                        "expected_statuses": ACTIVE_STATUSES,
                        "reason": (
                            f"compacted active provenance links from {item.source_event_count} "
                            f"to {item.proposed_source_event_count}; released={item.released_source_events}"
                        ),
                    }
                    for item in items
                ],
                actor="provenance-compaction",
                action="compact-provenance",
            )
            if applied:
                for item in items:
                    item.applied = True
            elif items:
                passed = False
                blocked_reason = "one or more candidate capsules changed before provenance compaction"

        totals = _totals(items, candidate_rows=candidate_rows, dry_run=dry_run, passed=passed)
        return ProvenanceCompactionReport(
            scope=scope,
            dry_run=dry_run,
            passed=passed,
            blocked_reason=blocked_reason,
            keep_events=keep_events,
            min_pinned_events=min_pinned_events,
            summary_only=summary_only,
            totals=totals,
            items=items,
            recommendations=_recommend(items, dry_run=dry_run, passed=passed, blocked_reason=blocked_reason),
        )


def _cold_source_event_ids(store: MemoryStore, *, scope: str | None) -> set[str]:
    placeholders = ",".join("?" for _ in COLD_STATUSES)
    clauses = [f"status IN ({placeholders})"]
    args: list[Any] = [*COLD_STATUSES]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    with store.session() as conn:
        rows = conn.execute(
            f"""
            SELECT source_event_ids_json
            FROM capsules
            WHERE {' AND '.join(clauses)}
            """,
            args,
        )
        ids: set[str] = set()
        for row in rows:
            ids.update(json.loads(row["source_event_ids_json"]))
        return ids


def _active_rows(store: MemoryStore, *, scope: str | None, summary_only: bool) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    clauses = [f"status IN ({placeholders})", "source_event_ids_json != '[]'"]
    args: list[Any] = [*sorted(ACTIVE_STATUSES)]
    if summary_only:
        clauses.append("kind = ?")
        args.append("summary")
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    with store.session() as conn:
        rows = conn.execute(
            f"""
            SELECT id, kind, status, title, scope, updated_at, source_event_ids_json
            FROM capsules
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, id ASC
            """,
            args,
        )
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "status": row["status"],
                "title": row["title"],
                "scope": row["scope"],
                "updated_at": row["updated_at"],
                "source_event_ids": json.loads(row["source_event_ids_json"]),
            }
            for row in rows
        ]


def _active_event_counts(active_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in active_rows:
        for event_id in set(row["source_event_ids"]):
            counts[event_id] = counts.get(event_id, 0) + 1
    return counts


def _plan_items(
    active_rows: list[dict[str, Any]],
    *,
    cold_event_ids: set[str],
    active_counts: dict[str, int],
    event_created_at: dict[str, str],
    keep_events: int,
    min_pinned_events: int,
    limit: int,
) -> list[ProvenanceCompactionItem]:
    planned: list[dict[str, Any]] = []
    for row in active_rows:
        source_ids = list(dict.fromkeys(row["source_event_ids"]))
        pinned = sorted(set(source_ids).intersection(cold_event_ids))
        if len(pinned) < min_pinned_events or len(source_ids) <= keep_events:
            continue
        retained = _retained_source_ids(
            source_ids,
            cold_event_ids=cold_event_ids,
            event_created_at=event_created_at,
            keep_events=keep_events,
        )
        if len(retained) >= len(source_ids):
            continue
        released = [event_id for event_id in source_ids if event_id not in set(retained)]
        planned.append(
            {
                "row": row,
                "source_ids": source_ids,
                "pinned": pinned,
                "retained": retained,
                "released": released,
            }
        )
    planned = sorted(
        planned,
        key=lambda item: (
            -len([event_id for event_id in item["released"] if event_id in cold_event_ids]),
            -len(item["released"]),
            -len(item["pinned"]),
            item["row"]["title"],
        ),
    )[:limit]

    items: list[ProvenanceCompactionItem] = []
    after_counts = dict(active_counts)
    for planned_item in planned:
        row = planned_item["row"]
        source_ids = planned_item["source_ids"]
        pinned = planned_item["pinned"]
        retained = planned_item["retained"]
        released = planned_item["released"]
        globally_unpinned = []
        for event_id in released:
            if event_id not in cold_event_ids:
                continue
            if after_counts.get(event_id, 0) <= 1:
                globally_unpinned.append(event_id)
            after_counts[event_id] = max(0, after_counts.get(event_id, 0) - 1)
        items.append(
            ProvenanceCompactionItem(
                capsule_id=row["id"],
                status=row["status"],
                kind=row["kind"],
                title=row["title"],
                scope=row["scope"],
                expected_source_event_ids=source_ids,
                source_event_count=len(source_ids),
                proposed_source_event_count=len(retained),
                pinned_cold_events=len(pinned),
                released_source_events=len(released),
                globally_unpinned_cold_events=len(globally_unpinned),
                retained_source_event_ids=retained,
                released_source_event_ids=released,
            )
        )
    return sorted(
        items,
        key=lambda item: (-item.globally_unpinned_cold_events, -item.released_source_events, item.title),
    )


def _retained_source_ids(
    source_ids: list[str],
    *,
    cold_event_ids: set[str],
    event_created_at: dict[str, str],
    keep_events: int,
) -> list[str]:
    ordered = _event_time_order(source_ids, event_created_at)
    non_cold = [event_id for event_id in ordered if event_id not in cold_event_ids]
    retained: list[str] = []
    if non_cold:
        retained.extend(_even_sample(non_cold, min(len(non_cold), max(1, keep_events // 2))))
    remaining_budget = keep_events - len(retained)
    if remaining_budget > 0:
        retained.extend(_even_sample([event_id for event_id in ordered if event_id not in set(retained)], remaining_budget))
    return [event_id for event_id in source_ids if event_id in set(retained)]


def _event_created_at_lookup(store: MemoryStore, active_rows: list[dict[str, Any]]) -> dict[str, str]:
    source_ids: list[str] = []
    for row in active_rows:
        source_ids.extend(row["source_event_ids"])
    rows = store.get_events(source_ids)
    return {row["id"]: row["created_at"] for row in rows}


def _event_time_order(source_ids: list[str], event_created_at: dict[str, str]) -> list[str]:
    return sorted(
        list(dict.fromkeys(source_ids)),
        key=lambda event_id: (
            event_created_at.get(event_id, ""),
            event_id,
        ),
    )


def _even_sample(items: list[str], count: int) -> list[str]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[-1]]
    indexes = {round(index * (len(items) - 1) / (count - 1)) for index in range(count)}
    return [items[index] for index in sorted(indexes)]


def _totals(
    items: list[ProvenanceCompactionItem],
    *,
    candidate_rows: list[dict[str, Any]],
    dry_run: bool,
    passed: bool,
) -> dict[str, Any]:
    return {
        "candidate_capsules": len(candidate_rows),
        "eligible_capsules": len(items),
        "source_events_before": sum(item.source_event_count for item in items),
        "source_events_after": sum(item.proposed_source_event_count for item in items),
        "released_source_events": sum(item.released_source_events for item in items),
        "globally_unpinned_cold_events": sum(item.globally_unpinned_cold_events for item in items),
        "applied_capsules": sum(1 for item in items if item.applied),
        "dry_run": dry_run,
        "passed": passed,
    }


def _recommend(
    items: list[ProvenanceCompactionItem],
    *,
    dry_run: bool,
    passed: bool,
    blocked_reason: str,
) -> list[str]:
    if blocked_reason:
        return [blocked_reason]
    if not items:
        return ["No eligible active provenance pins exceed the current threshold."]
    total_release = sum(item.globally_unpinned_cold_events for item in items)
    if dry_run:
        return [
            f"Review {len(items)} planned provenance compactions before applying.",
            f"Applying the current plan would unpin {total_release} cold source events from active capsules.",
            f'To apply, rerun with --apply --confirm "{CONFIRMATION}" after backup/restore evidence is current.',
        ]
    if passed:
        return [
            f"Applied {sum(1 for item in items if item.applied)} provenance compactions.",
            "Run cold-stewardship, health, and recall-regression before any retention-cycle or prune decision.",
        ]
    return ["Compaction did not fully apply; inspect memory_actions and rerun cold-stewardship."]
