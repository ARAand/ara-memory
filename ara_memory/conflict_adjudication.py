from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ara_memory.models import MemoryStatus
from ara_memory.storage import MemoryStore, row_to_capsule


@dataclass(slots=True)
class ConflictAdjudicationGroup:
    key: str
    count: int
    kept_id: str
    duplicate_ids: list[str]
    reason: str
    example_title: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "count": self.count,
            "kept_id": self.kept_id,
            "duplicate_ids": self.duplicate_ids,
            "reason": self.reason,
            "example_title": self.example_title,
        }


@dataclass(slots=True)
class ConflictAdjudicationReport:
    scope: str | None
    dry_run: bool
    conflicts_seen: int
    groups: list[ConflictAdjudicationGroup]
    superseded: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "conflicts_seen": self.conflicts_seen,
            "groups": [group.as_dict() for group in self.groups],
            "superseded": self.superseded,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Conflict Adjudication: {scope}",
            f"dry_run: {self.dry_run}",
            f"conflicts_seen={self.conflicts_seen}, groups={len(self.groups)}, superseded={self.superseded}",
        ]
        if not self.groups:
            lines.append("- No duplicate conflict groups found.")
        for group in self.groups:
            lines.append(
                f"- {group.reason}: count={group.count}, keep={group.kept_id}, supersede={len(group.duplicate_ids)}"
            )
            lines.append(f"  - {group.example_title}")
        return "\n".join(lines)


class ConflictAdjudicator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        limit: int = 500,
        dry_run: bool = True,
    ) -> ConflictAdjudicationReport:
        self.store.init()
        conflicts = self._conflicts(scope=scope, limit=limit)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for cap in conflicts:
            key = _source_signature(cap)
            grouped.setdefault(key, []).append(cap)

        groups: list[ConflictAdjudicationGroup] = []
        superseded = 0
        for key, caps in grouped.items():
            if len(caps) < 2:
                continue
            ordered = sorted(caps, key=lambda cap: (cap["updated_at"], cap["id"]), reverse=True)
            kept = ordered[0]
            duplicates = ordered[1:]
            group = ConflictAdjudicationGroup(
                key=key,
                count=len(ordered),
                kept_id=kept["id"],
                duplicate_ids=[cap["id"] for cap in duplicates],
                reason="duplicate source-event conflict",
                example_title=kept["title"],
            )
            groups.append(group)
            if not dry_run:
                for duplicate in duplicates:
                    if self.store.update_capsule_status(
                        duplicate["id"],
                        MemoryStatus.SUPERSEDED,
                        actor="conflict-adjudicator",
                        reason=f"duplicate conflict superseded by {kept['id']}",
                    ):
                        superseded += 1
            else:
                superseded += len(duplicates)

        groups.sort(key=lambda group: (group.count, group.example_title), reverse=True)
        return ConflictAdjudicationReport(
            scope=scope,
            dry_run=dry_run,
            conflicts_seen=len(conflicts),
            groups=groups,
            superseded=superseded,
        )

    def _conflicts(self, *, scope: str | None, limit: int) -> list[dict[str, Any]]:
        clauses = ["status = ?", "kind = ?"]
        args: list[Any] = [MemoryStatus.CANDIDATE.value, "conflict"]
        if scope:
            clauses.append("scope = ?")
            args.append(scope)
        args.append(limit)
        with self.store.session() as conn:
            return [
                row_to_capsule(row)
                for row in conn.execute(
                    f"""
                    SELECT *
                    FROM capsules
                    WHERE {' AND '.join(clauses)}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    args,
                )
            ]


def _source_signature(cap: dict[str, Any]) -> str:
    source_ids = cap.get("source_event_ids", [])
    if isinstance(source_ids, str):
        try:
            source_ids = json.loads(source_ids)
        except json.JSONDecodeError:
            source_ids = [source_ids]
    return "|".join(sorted(str(item) for item in source_ids))
