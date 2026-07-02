from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ara_memory.models import MemoryStatus
from ara_memory.retention import COLD_STATUSES
from ara_memory.regression import RecallRegressionCase, run_recall_regression
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
    retention_strategy: str
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
            "retention_strategy": self.retention_strategy,
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
    recall_preflight: dict[str, Any] | None
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
            "recall_preflight": self.recall_preflight,
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
        recall_cases: list[RecallRegressionCase] | None = None,
        recall_baseline: dict[str, Any] | None = None,
        recall_max_token_growth: float = 0.25,
        recall_min_overlap: float = 0.35,
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
        baseline_protected_capsule_ids = _baseline_guarded_capsule_ids(recall_baseline)
        active_counts = _active_event_counts(_active_rows(self.store, scope=scope, summary_only=False))
        event_info = _event_info_lookup(self.store, candidate_rows)
        items = _plan_items(
            candidate_rows,
            cold_event_ids=cold_event_ids,
            protected_capsule_ids=baseline_protected_capsule_ids,
            active_counts=active_counts,
            event_info=event_info,
            keep_events=keep_events,
            min_pinned_events=min_pinned_events,
            limit=limit,
        )
        blocked_reason = ""
        passed = True
        recall_preflight = None
        if not dry_run and confirm != CONFIRMATION:
            blocked_reason = f'apply requires --confirm "{CONFIRMATION}"'
            passed = False
        elif recall_cases and items:
            items, recall_preflight = _stabilize_items_with_recall_preflight(
                self.store,
                items,
                cases=recall_cases,
                baseline=recall_baseline,
                max_token_growth=recall_max_token_growth,
                min_overlap=recall_min_overlap,
            )
            if not recall_preflight.get("passed") and not dry_run:
                blocked_reason = "recall regression preflight failed; live provenance was not compacted"
                passed = False
        if not dry_run and passed:
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

        totals = _totals(
            items,
            candidate_rows=candidate_rows,
            baseline_protected_capsule_ids=baseline_protected_capsule_ids,
            dry_run=dry_run,
            passed=passed,
        )
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
            recall_preflight=recall_preflight,
            recommendations=_recommend(
                items,
                dry_run=dry_run,
                passed=passed,
                blocked_reason=blocked_reason,
                recall_preflight=recall_preflight,
                baseline_protected_capsules=int(totals["baseline_protected_capsules"]),
            ),
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
    protected_capsule_ids: set[str],
    active_counts: dict[str, int],
    event_info: dict[str, dict[str, Any]],
    keep_events: int,
    min_pinned_events: int,
    limit: int,
) -> list[ProvenanceCompactionItem]:
    planned: list[dict[str, Any]] = []
    for row in active_rows:
        if row["id"] in protected_capsule_ids:
            continue
        source_ids = list(dict.fromkeys(row["source_event_ids"]))
        pinned = sorted(set(source_ids).intersection(cold_event_ids))
        if len(pinned) < min_pinned_events or len(source_ids) <= keep_events:
            continue
        retained = _retained_source_ids(
            source_ids,
            cold_event_ids=cold_event_ids,
            event_info=event_info,
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
                retention_strategy="quality-time-sample-v1",
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
    event_info: dict[str, dict[str, Any]],
    keep_events: int,
) -> list[str]:
    ordered = _event_time_order(source_ids, event_info)
    non_cold = [event_id for event_id in ordered if event_id not in cold_event_ids]
    retained: list[str] = []
    if non_cold:
        retained.extend(
            _quality_time_sample(
                non_cold,
                min(len(non_cold), max(1, keep_events // 2)),
                event_info=event_info,
            )
        )
    remaining_budget = keep_events - len(retained)
    if remaining_budget > 0:
        retained.extend(
            _quality_time_sample(
                [event_id for event_id in ordered if event_id not in set(retained)],
                remaining_budget,
                event_info=event_info,
            )
        )
    return [event_id for event_id in source_ids if event_id in set(retained)]


def _event_info_lookup(store: MemoryStore, active_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    source_ids: list[str] = []
    for row in active_rows:
        source_ids.extend(row["source_event_ids"])
    rows = store.get_events(source_ids)
    return {
        row["id"]: {
            "kind": row["kind"],
            "source": row["source"],
            "text": row["text"],
            "created_at": row["created_at"],
            "metadata": _loads_json(row["metadata_json"]),
        }
        for row in rows
    }


def _event_time_order(source_ids: list[str], event_info: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        list(dict.fromkeys(source_ids)),
        key=lambda event_id: (
            str(event_info.get(event_id, {}).get("created_at") or ""),
            event_id,
        ),
    )


def _quality_time_sample(items: list[str], count: int, *, event_info: dict[str, dict[str, Any]]) -> list[str]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    quality_target = count if count <= 3 else max(2, count // 2)
    retained = _quality_sample(items, quality_target, event_info=event_info)
    remaining = count - len(retained)
    if remaining > 0:
        retained.extend(_even_sample([item for item in items if item not in set(retained)], remaining))
    retained_set = set(retained)
    return [item for item in items if item in retained_set]


def _quality_sample(items: list[str], count: int, *, event_info: dict[str, dict[str, Any]]) -> list[str]:
    ranked = sorted(
        list(dict.fromkeys(items)),
        key=lambda event_id: (
            -_event_quality_score(event_id, event_info.get(event_id, {})),
            str(event_info.get(event_id, {}).get("created_at") or ""),
            event_id,
        ),
    )
    return ranked[:count]


def _event_quality_score(event_id: str, info: dict[str, Any]) -> float:
    text = str(info.get("text") or "")
    lowered = text.lower()
    kind = str(info.get("kind") or "")
    score = {
        "decision": 5.0,
        "prompt": 4.0,
        "assistant": 3.2,
        "note": 3.0,
        "diff": 2.8,
        "file": 2.4,
        "command": 2.2,
        "image": 1.2,
    }.get(kind, 1.0)
    cue_weights = {
        "decision:": 3.0,
        "verification": 2.4,
        "verified": 2.0,
        "passed": 2.0,
        "failed": 2.0,
        "error": 1.8,
        "regression": 1.8,
        "health": 1.6,
        "milestone": 1.6,
        "backup": 1.4,
        "retention": 1.4,
        "cold": 1.2,
        "prune": 1.2,
        "provenance": 1.2,
        "identity": 1.2,
        "purpose": 1.2,
        "commit": 1.0,
        "pushed": 1.0,
        "test": 1.0,
        "audit": 1.0,
        "risk": 1.0,
    }
    for cue, weight in cue_weights.items():
        if cue in lowered:
            score += weight
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    if metadata:
        score += min(1.0, len(metadata) * 0.1)
    score += min(0.8, len(text) / 2000.0)
    if "ignore previous" in lowered or "system prompt" in lowered:
        score -= 2.0
    return score


def _loads_json(value: Any) -> Any:
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _baseline_guarded_capsule_ids(baseline: dict[str, Any] | None) -> set[str]:
    if not isinstance(baseline, dict):
        return set()
    protected: set[str] = set()
    for item in baseline.get("cases", []):
        if not isinstance(item, dict):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        for key in ("visible_capsule_ids", "selected_capsule_ids"):
            protected.update(str(capsule_id) for capsule_id in details.get(key, []) if str(capsule_id))
    return protected


def _recall_preflight(
    store: MemoryStore,
    items: list[ProvenanceCompactionItem],
    *,
    cases: list[RecallRegressionCase],
    baseline: dict[str, Any] | None,
    max_token_growth: float,
    min_overlap: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        shadow_root = Path(tmp) / "shadow-memory"
        _copy_recall_store(store, shadow_root)
        shadow_store = MemoryStore(shadow_root)
        ok = shadow_store.update_capsule_source_events_batch(
            [
                {
                    "capsule_id": item.capsule_id,
                    "source_event_ids": item.retained_source_event_ids,
                    "expected_source_event_ids": item.expected_source_event_ids,
                    "expected_statuses": ACTIVE_STATUSES,
                    "reason": "recall preflight provenance compaction simulation",
                }
                for item in items
            ],
            actor="provenance-compaction-preflight",
            action="compact-provenance-preflight",
        )
        if not ok:
            return {
                "passed": False,
                "simulation_applied": False,
                "blocked_reason": "shadow source-event update failed",
                "cases": [],
                "baseline_comparison": [],
            }
        from ara_memory.core import AraMemory

        report = run_recall_regression(
            AraMemory(shadow_root),
            cases,
            baseline=baseline,
            max_token_growth=max_token_growth,
            min_overlap=min_overlap,
        )
        payload = report.as_dict()
        return {
            "passed": report.passed,
            "simulation_applied": True,
            "cases": [_compact_recall_case(item) for item in payload.get("cases", [])],
            "baseline_comparison": [_compact_recall_case(item) for item in payload.get("baseline_comparison", [])],
        }


def _stabilize_items_with_recall_preflight(
    store: MemoryStore,
    items: list[ProvenanceCompactionItem],
    *,
    cases: list[RecallRegressionCase],
    baseline: dict[str, Any] | None,
    max_token_growth: float,
    min_overlap: float,
    max_attempts: int = 20,
) -> tuple[list[ProvenanceCompactionItem], dict[str, Any]]:
    active_items = list(items)
    removed_candidate_ids: list[str] = []
    attempts: list[dict[str, Any]] = []
    last_preflight: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        if not active_items:
            last_preflight = {
                "passed": True,
                "simulation_applied": False,
                "cases": [],
                "baseline_comparison": [],
            }
            attempts.append(
                {
                    "attempt": attempt,
                    "passed": True,
                    "candidate_count": 0,
                    "elided_candidate_ids": [],
                }
            )
            break
        last_preflight = _recall_preflight(
            store,
            active_items,
            cases=cases,
            baseline=baseline,
            max_token_growth=max_token_growth,
            min_overlap=min_overlap,
        )
        offender_ids = (
            set()
            if last_preflight.get("passed")
            else _recall_preflight_offender_ids(last_preflight, active_items)
        )
        attempts.append(
            {
                "attempt": attempt,
                "passed": bool(last_preflight.get("passed")),
                "candidate_count": len(active_items),
                "elided_candidate_ids": sorted(offender_ids),
            }
        )
        if last_preflight.get("passed") or not offender_ids:
            break
        if attempt == max_attempts:
            break
        removed_candidate_ids.extend(
            item.capsule_id for item in active_items if item.capsule_id in offender_ids
        )
        active_items = [item for item in active_items if item.capsule_id not in offender_ids]
    if last_preflight is None:
        last_preflight = {
            "passed": True,
            "simulation_applied": False,
            "cases": [],
            "baseline_comparison": [],
        }
    stabilized = dict(last_preflight)
    stabilized["auto_elision"] = {
        "enabled": True,
        "stabilized": bool(stabilized.get("passed")),
        "attempts": attempts,
        "removed_candidate_ids": removed_candidate_ids,
        "removed_candidate_count": len(removed_candidate_ids),
        "final_candidate_count": len(active_items),
    }
    return active_items, stabilized


def _recall_preflight_offender_ids(
    recall_preflight: dict[str, Any],
    items: list[ProvenanceCompactionItem],
) -> set[str]:
    candidate_ids = {item.capsule_id for item in items}
    failed_names = {
        str(item.get("name"))
        for item in recall_preflight.get("baseline_comparison", [])
        if isinstance(item, dict) and not item.get("passed")
    }
    if not failed_names and not recall_preflight.get("baseline_comparison"):
        return set()
    visible_offenders: set[str] = set()
    selected_offenders: set[str] = set()
    for item in recall_preflight.get("cases", []):
        if not isinstance(item, dict) or str(item.get("name")) not in failed_names:
            continue
        visible_offenders.update(
            capsule_id
            for capsule_id in item.get("visible_capsule_ids", [])
            if capsule_id in candidate_ids
        )
        selected_offenders.update(
            capsule_id
            for capsule_id in item.get("selected_capsule_ids", [])
            if capsule_id in candidate_ids
        )
    return visible_offenders or selected_offenders


def _copy_recall_store(store: MemoryStore, shadow_root: Path) -> None:
    shadow_root.mkdir(parents=True, exist_ok=True)
    (shadow_root / "ledger").mkdir(parents=True, exist_ok=True)
    (shadow_root / "archive").mkdir(parents=True, exist_ok=True)
    if store.hot_dir.exists():
        shutil.copytree(store.hot_dir, shadow_root / "hot", dirs_exist_ok=True)
    else:
        (shadow_root / "hot").mkdir(parents=True, exist_ok=True)
    with store.session() as source:
        target = sqlite3.connect(shadow_root / "memory.db")
        try:
            source.backup(target)
        finally:
            target.close()


def _compact_recall_case(item: dict[str, Any]) -> dict[str, Any]:
    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    return {
        "name": item.get("name"),
        "passed": bool(item.get("passed")),
        "failures": details.get("failures", []),
        "estimated_tokens": details.get("estimated_tokens", details.get("current_tokens")),
        "baseline_tokens": details.get("baseline_tokens"),
        "current_tokens": details.get("current_tokens"),
        "source_event_overlap": details.get("source_event_overlap"),
        "capsule_id_overlap": details.get("capsule_id_overlap"),
        "overlap_basis": details.get("overlap_basis"),
        "visible_capsule_ids": details.get("visible_capsule_ids", []),
        "selected_capsule_ids": details.get("selected_capsule_ids", []),
        "visible_source_event_count": details.get("visible_source_event_count"),
        "selected_source_event_count": details.get("selected_source_event_count"),
    }


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
    baseline_protected_capsule_ids: set[str],
    dry_run: bool,
    passed: bool,
) -> dict[str, Any]:
    candidate_ids = {str(row["id"]) for row in candidate_rows}
    return {
        "candidate_capsules": len(candidate_rows),
        "baseline_protected_capsules": len(candidate_ids.intersection(baseline_protected_capsule_ids)),
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
    recall_preflight: dict[str, Any] | None,
    baseline_protected_capsules: int,
) -> list[str]:
    if blocked_reason:
        out = [blocked_reason]
        if recall_preflight:
            out.append("Inspect recall_preflight before retrying provenance compaction.")
        return out
    auto_elision = recall_preflight.get("auto_elision") if isinstance(recall_preflight, dict) else {}
    removed_by_recall = int(auto_elision.get("removed_candidate_count", 0) or 0) if isinstance(auto_elision, dict) else 0
    if not items:
        if removed_by_recall:
            return [
                f"Recall preflight elided {removed_by_recall} unstable compaction candidates; no safe provenance compaction remains."
            ]
        if baseline_protected_capsules:
            return [
                f"No eligible provenance compaction remains after protecting {baseline_protected_capsules} baseline-guarded recall capsules."
            ]
        return ["No eligible active provenance pins exceed the current threshold."]
    total_release = sum(item.globally_unpinned_cold_events for item in items)
    if dry_run:
        out = [
            f"Review {len(items)} planned provenance compactions before applying.",
            f"Applying the current plan would unpin {total_release} cold source events from active capsules.",
            f'To apply, rerun with --apply --confirm "{CONFIRMATION}" after backup/restore evidence is current.',
        ]
        if recall_preflight:
            out.append(f"Recall regression preflight passed={recall_preflight.get('passed')}.")
        if removed_by_recall:
            out.append(f"Recall preflight elided {removed_by_recall} unstable compaction candidates from this plan.")
        return out
    if passed:
        return [
            f"Applied {sum(1 for item in items if item.applied)} provenance compactions.",
            "Run cold-stewardship, health, and recall-regression before any retention-cycle or prune decision.",
        ]
    return ["Compaction did not fully apply; inspect memory_actions and rerun cold-stewardship."]
