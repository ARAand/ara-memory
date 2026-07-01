from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ara_memory.compressors import semantic_consolidation_text
from ara_memory.models import Capsule, CapsuleKind, MemoryStatus
from ara_memory.promotion import can_promote_capsule, promotion_block_reason
from ara_memory.storage import MemoryStore, row_to_capsule


GROUP_PATTERNS = {
    "command": ("Command: ",),
    "file_artifact": ("File artifact: ", "Untracked file "),
    "git_status": ("Git status for ",),
    "session": (),
}
SESSION_EXCLUDED_PREFIXES = ("Command: ", "File artifact: ", "Untracked file ", "Git status")


@dataclass(slots=True)
class EpisodeSummaryGroup:
    pattern: str
    count: int
    summary_title: str
    source_capsule_ids: list[str]
    example_titles: list[str]
    applied_summary_id: str | None = None
    blocked_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "count": self.count,
            "summary_title": self.summary_title,
            "source_capsule_ids": self.source_capsule_ids,
            "example_titles": self.example_titles,
            "applied_summary_id": self.applied_summary_id,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(slots=True)
class EpisodeSummaryReport:
    scope: str | None
    dry_run: bool
    candidates_seen: int
    groups: list[EpisodeSummaryGroup]
    superseded: int
    summaries_created: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "candidates_seen": self.candidates_seen,
            "groups": [group.as_dict() for group in self.groups],
            "superseded": self.superseded,
            "summaries_created": self.summaries_created,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Episode Summary: {scope}",
            f"dry_run: {self.dry_run}",
            f"candidates_seen={self.candidates_seen}, groups={len(self.groups)}, summaries_created={self.summaries_created}, superseded={self.superseded}",
        ]
        for group in self.groups:
            lines.append(f"- {group.pattern}: {group.count} -> {group.summary_title}")
            for title in group.example_titles[:3]:
                lines.append(f"  - {title}")
        return "\n".join(lines)


class EpisodeSummaryConsolidator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        pattern: str = "command",
        min_group_size: int = 5,
        limit: int = 50,
        dry_run: bool = True,
    ) -> EpisodeSummaryReport:
        if pattern not in GROUP_PATTERNS:
            raise ValueError(f"Unknown episode summary pattern: {pattern}")
        self.store.init()
        candidates = self._candidates(scope=scope, pattern=pattern, limit=limit)
        groups = []
        if len(candidates) >= min_group_size:
            groups.append(self._build_group(pattern, candidates))
        summaries_created = 0
        superseded = 0
        if not dry_run:
            for group in groups:
                summary = self._apply_group(group, candidates, scope=scope)
                if summary is None:
                    continue
                group.applied_summary_id = summary.id
                summaries_created += 1
                superseded += group.count
        return EpisodeSummaryReport(
            scope=scope,
            dry_run=dry_run,
            candidates_seen=len(candidates),
            groups=groups,
            summaries_created=summaries_created,
            superseded=superseded,
        )

    def _candidates(self, *, scope: str | None, pattern: str, limit: int) -> list[dict[str, Any]]:
        clauses = ["status = ?", "kind = ?"]
        args: list[Any] = [MemoryStatus.CANDIDATE.value, CapsuleKind.EPISODE.value]
        if scope:
            clauses.append("scope = ?")
            args.append(scope)
        title_clauses = []
        if pattern == "session":
            for prefix in SESSION_EXCLUDED_PREFIXES:
                clauses.append("title NOT LIKE ?")
                args.append(f"{prefix}%")
        else:
            for prefix in GROUP_PATTERNS[pattern]:
                title_clauses.append("title LIKE ?")
                args.append(f"{prefix}%")
            clauses.append(f"({' OR '.join(title_clauses)})")
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

    def _build_group(self, pattern: str, candidates: list[dict[str, Any]]) -> EpisodeSummaryGroup:
        title = _summary_title(pattern, len(candidates))
        return EpisodeSummaryGroup(
            pattern=pattern,
            count=len(candidates),
            summary_title=title,
            source_capsule_ids=[cap["id"] for cap in candidates],
            example_titles=[cap["title"] for cap in candidates[:5]],
        )

    def _apply_group(
        self,
        group: EpisodeSummaryGroup,
        candidates: list[dict[str, Any]],
        *,
        scope: str | None,
    ) -> Capsule | None:
        source_ids: list[str] = []
        tags = {"episode-summary", group.pattern}
        for cap in candidates:
            source_ids.extend(cap["source_event_ids"])
            tags.update(cap["tags"][:8])
        summary = Capsule.create(
            kind=CapsuleKind.SUMMARY,
            title=group.summary_title,
            body=semantic_consolidation_text(candidates, limit=1400),
            scope=scope or candidates[0]["scope"],
            confidence=max(0.7, max(float(cap["confidence"]) for cap in candidates)),
            salience=max(float(cap["salience"]) for cap in candidates),
            source_event_ids=sorted(set(source_ids)),
            tags=sorted(tags),
            status=MemoryStatus.STABLE,
        )
        gate = can_promote_capsule(
            self.store,
            asdict(summary),
            require_provenance=True,
            source_capsule_count=group.count,
        )
        if not gate.allowed:
            group.blocked_reason = promotion_block_reason(gate)
            return None
        self.store.upsert_capsule(summary)
        for cap in candidates:
            self.store.update_capsule_status(
                cap["id"],
                MemoryStatus.SUPERSEDED,
                actor="episode-summary",
                reason=f"consolidated into {summary.id}",
            )
        return summary


def _summary_title(pattern: str, count: int) -> str:
    if pattern == "command":
        return f"Consolidated command episode outcomes ({count} episodes)"
    if pattern == "file_artifact":
        return f"Consolidated file artifact episode evidence ({count} episodes)"
    if pattern == "git_status":
        return f"Consolidated git status episode evidence ({count} episodes)"
    if pattern == "session":
        return f"Consolidated session episode narrative ({count} episodes)"
    return f"Consolidated episode evidence ({count} episodes)"
