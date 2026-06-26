from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.compressors import compact_text
from ara_memory.models import Capsule, CapsuleKind, MemoryStatus
from ara_memory.storage import MemoryStore, row_to_capsule


PATTERNS = {
    "project_file_artifact": {
        "kind": CapsuleKind.PROJECT,
        "title_prefix": "Project memory: File artifact:",
        "body_prefix": "File artifact:",
        "title": "Consolidated project file artifact evidence",
        "tags": ["candidate-summary", "file-artifact", "project"],
    },
    "failure_success_command": {
        "kind": CapsuleKind.FAILURE,
        "title_prefix": "Failure memory: Command:",
        "body_prefix": "Command:",
        "title": "Consolidated successful command evidence misfiled as failures",
        "tags": ["candidate-summary", "command", "success"],
    },
    "procedure_command": {
        "kind": CapsuleKind.PROCEDURE,
        "title_prefix": "Procedure candidate: Command:",
        "body_prefix": "Command:",
        "title": "Consolidated command procedure evidence",
        "tags": ["candidate-summary", "command", "procedure"],
    },
    "procedure_progress_update": {
        "kind": CapsuleKind.PROCEDURE,
        "title_prefix": "Procedure candidate:",
        "body_like_any": [
            "Continue building %",
            "Continue hardening %",
            "Add %",
            "Added %",
            "Implemented %",
            "Changed %",
            "Fixed %",
            "Wired %",
        ],
        "title": "Consolidated project progress updates misfiled as procedures",
        "tags": ["candidate-summary", "procedure", "progress"],
    },
    "project_worktree_evidence": {
        "kind": CapsuleKind.PROJECT,
        "title_prefix": "Project memory:",
        "body_like_any": [
            "Git status for %",
            "Git diff stat for %",
            "Git diff for %",
            "Untracked file manifest:%",
        ],
        "title": "Consolidated worktree evidence candidates",
        "tags": ["candidate-summary", "git", "worktree", "project"],
    },
    "failure_worktree_evidence": {
        "kind": CapsuleKind.FAILURE,
        "title_prefix": "Failure memory: Git ",
        "body_like_any": [
            "Git diff for %",
            "Git status for %",
            "Git diff stat for %",
        ],
        "title": "Consolidated worktree evidence misfiled as failures",
        "tags": ["candidate-summary", "git", "worktree", "failure"],
    },
    "failure_taxonomy_discussion": {
        "kind": CapsuleKind.FAILURE,
        "title_prefix": "Failure memory:",
        "body_like_any": [
            "%failure/procedure%",
            "%failure or procedural memory%",
            "%failure/regression word%",
            "%no longer creates failure%",
            "%failure candidates%",
            "%failure capsules%",
            "%failure taxonomy discussion%",
            "%failure taxonomy discussions%",
            "%misfiled as failures%",
        ],
        "title": "Consolidated failure taxonomy discussions misfiled as failures",
        "tags": ["candidate-summary", "failure", "taxonomy"],
    },
    "failure_operational_update": {
        "kind": CapsuleKind.FAILURE,
        "title_prefix": "Failure memory:",
        "body_like_any": [
            "Implemented %",
            "Added %",
            "Completed verification%",
            "Continue Ara Memory OS%",
            "Continue building Ara Memory OS%",
            "Narrowed %",
            "After capture, recall-regression exposed%",
            "Decision: %noise%",
            "Decision: %candidate%",
            "Command: python -m ara_memory candidate-summary%-> created%",
        ],
        "title": "Consolidated operational updates misfiled as failures",
        "tags": ["candidate-summary", "failure", "operational-update"],
    },
    "decision_memory_policy": {
        "kind": CapsuleKind.DECISION,
        "title_prefix": "Decision:",
        "body_like_any": [
            "%memory%",
            "%recall%",
            "%candidate%",
            "%summary%",
            "%worker%",
            "%pruning%",
            "%retention%",
            "%spool%",
            "%evidence%",
            "%goal%",
            "%purpose%",
        ],
        "title": "Consolidated memory policy decisions",
        "tags": ["candidate-summary", "decision", "memory-policy"],
    },
    "goal_purpose_update": {
        "kind": CapsuleKind.GOAL,
        "title_prefix": "Goal memory:",
        "body_like_any": [
            "%purpose%",
            "%objective%",
            "%goal%",
            "%intent%",
            "%natural memory%",
            "%long-running%",
        ],
        "title": "Consolidated purpose-layer goal evidence",
        "tags": ["candidate-summary", "goal", "purpose"],
    },
    "procedure_memory_policy": {
        "kind": CapsuleKind.PROCEDURE,
        "title_prefix": "Procedure candidate: Decision:",
        "body_like_any": [
            "%memory%",
            "%recall%",
            "%candidate%",
            "%summary%",
            "%worker%",
            "%pruning%",
            "%retention%",
            "%spool%",
            "%evidence%",
            "%goal%",
            "%purpose%",
        ],
        "title": "Consolidated memory policy procedure evidence",
        "tags": ["candidate-summary", "procedure", "memory-policy"],
    },
}


@dataclass(slots=True)
class CandidateSummaryGroup:
    pattern: str
    count: int
    summary_title: str
    source_capsule_ids: list[str]
    example_titles: list[str]
    applied_summary_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "count": self.count,
            "summary_title": self.summary_title,
            "source_capsule_ids": self.source_capsule_ids,
            "example_titles": self.example_titles,
            "applied_summary_id": self.applied_summary_id,
        }


@dataclass(slots=True)
class CandidateSummaryReport:
    scope: str | None
    dry_run: bool
    candidates_seen: int
    groups: list[CandidateSummaryGroup]
    summaries_created: int
    superseded: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "candidates_seen": self.candidates_seen,
            "groups": [group.as_dict() for group in self.groups],
            "summaries_created": self.summaries_created,
            "superseded": self.superseded,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Candidate Summary: {scope}",
            f"dry_run: {self.dry_run}",
            f"candidates_seen={self.candidates_seen}, groups={len(self.groups)}, summaries_created={self.summaries_created}, superseded={self.superseded}",
        ]
        for group in self.groups:
            lines.append(f"- {group.pattern}: {group.count} -> {group.summary_title}")
            for title in group.example_titles[:3]:
                lines.append(f"  - {title}")
        return "\n".join(lines)


class CandidateSummaryConsolidator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        pattern: str = "all",
        min_group_size: int = 3,
        limit: int = 80,
        dry_run: bool = True,
    ) -> CandidateSummaryReport:
        self.store.init()
        pattern_names = list(PATTERNS) if pattern == "all" else [pattern]
        unknown = [name for name in pattern_names if name not in PATTERNS]
        if unknown:
            raise ValueError(f"Unknown candidate summary pattern: {unknown[0]}")

        groups: list[CandidateSummaryGroup] = []
        candidates_seen = 0
        summaries_created = 0
        superseded = 0
        grouped_candidates: list[tuple[CandidateSummaryGroup, list[dict[str, Any]]]] = []
        for name in pattern_names:
            candidates = self._candidates(scope=scope, pattern=name, limit=limit)
            candidates_seen += len(candidates)
            if len(candidates) < min_group_size:
                continue
            group = self._build_group(name, candidates)
            groups.append(group)
            grouped_candidates.append((group, candidates))

        if not dry_run:
            for group, candidates in grouped_candidates:
                summary = self._apply_group(group, candidates, scope=scope)
                group.applied_summary_id = summary.id
                summaries_created += 1
                superseded += group.count

        return CandidateSummaryReport(
            scope=scope,
            dry_run=dry_run,
            candidates_seen=candidates_seen,
            groups=groups,
            summaries_created=summaries_created,
            superseded=superseded,
        )

    def _candidates(self, *, scope: str | None, pattern: str, limit: int) -> list[dict[str, Any]]:
        config = PATTERNS[pattern]
        clauses = ["status = ?", "kind = ?", "title LIKE ?"]
        args: list[Any] = [
            MemoryStatus.CANDIDATE.value,
            config["kind"].value,
            f"{config['title_prefix']}%",
        ]
        body_prefix = config.get("body_prefix")
        if body_prefix:
            clauses.append("body LIKE ?")
            args.append(f"{body_prefix}%")
        body_like_any = config.get("body_like_any", [])
        if body_like_any:
            clauses.append("(" + " OR ".join(["body LIKE ?" for _ in body_like_any]) + ")")
            args.extend(body_like_any)
        if pattern == "failure_success_command":
            clauses.append("(body LIKE ? OR body LIKE ? OR body LIKE ?)")
            args.extend(["%-> pass%", "%Exit code: 0%", "%\nOK%"])
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

    def _build_group(self, pattern: str, candidates: list[dict[str, Any]]) -> CandidateSummaryGroup:
        return CandidateSummaryGroup(
            pattern=pattern,
            count=len(candidates),
            summary_title=f"{PATTERNS[pattern]['title']} ({len(candidates)} candidates)",
            source_capsule_ids=[cap["id"] for cap in candidates],
            example_titles=[cap["title"] for cap in candidates[:5]],
        )

    def _apply_group(
        self,
        group: CandidateSummaryGroup,
        candidates: list[dict[str, Any]],
        *,
        scope: str | None,
    ) -> Capsule:
        source_ids: list[str] = []
        tags = set(PATTERNS[group.pattern]["tags"])
        bodies = []
        for cap in candidates:
            source_ids.extend(cap["source_event_ids"])
            tags.update(cap["tags"][:8])
            bodies.append(f"- {cap['title']}: {cap['body']}")
        summary = Capsule.create(
            kind=CapsuleKind.SUMMARY,
            title=group.summary_title,
            body=compact_text("\n".join(bodies), limit=1400),
            scope=scope or candidates[0]["scope"],
            confidence=max(0.7, max(float(cap["confidence"]) for cap in candidates)),
            salience=max(float(cap["salience"]) for cap in candidates),
            source_event_ids=sorted(set(source_ids)),
            tags=sorted(tags),
            status=MemoryStatus.STABLE,
        )
        self.store.upsert_capsule(summary)
        for cap in candidates:
            self.store.update_capsule_status(
                cap["id"],
                MemoryStatus.SUPERSEDED,
                actor="candidate-summary",
                reason=f"consolidated into {summary.id}",
            )
        return summary
