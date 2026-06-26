from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.models import MemoryStatus
from ara_memory.storage import MemoryStore


@dataclass(slots=True)
class CandidatePressureReport:
    scope: str | None
    totals: dict[str, Any]
    by_kind: list[dict[str, Any]]
    by_title_pattern: list[dict[str, Any]]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "totals": self.totals,
            "by_kind": self.by_kind,
            "by_title_pattern": self.by_title_pattern,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Candidate Pressure: {scope}"]
        lines.append(
            "totals: "
            f"candidate={self.totals['candidate']}, stable={self.totals['stable']}, "
            f"ratio={self.totals['candidate_stable_ratio']:.2f}"
        )
        lines.append("## Candidate By Kind")
        for row in self.by_kind:
            lines.append(f"- {row['kind']}: {row['count']} ({row['share']:.0%})")
        lines.append("## Candidate Title Patterns")
        for row in self.by_title_pattern:
            lines.append(f"- {row['pattern']}: {row['count']} ({row['share']:.0%})")
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


class CandidatePressureAnalyzer:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, *, scope: str | None = None, limit: int = 20) -> CandidatePressureReport:
        self.store.init()
        clauses = []
        args: list[Any] = []
        if scope:
            clauses.append("scope = ?")
            args.append(scope)
        scope_where = f" AND {' AND '.join(clauses)}" if clauses else ""
        with self.store.session() as conn:
            candidate = conn.execute(
                f"SELECT COUNT(*) FROM capsules WHERE status = ?{scope_where}",
                [MemoryStatus.CANDIDATE.value, *args],
            ).fetchone()[0]
            stable = conn.execute(
                f"SELECT COUNT(*) FROM capsules WHERE status = ?{scope_where}",
                [MemoryStatus.STABLE.value, *args],
            ).fetchone()[0]
            by_kind = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT kind, COUNT(*) AS count
                    FROM capsules
                    WHERE status = ?{scope_where}
                    GROUP BY kind
                    ORDER BY count DESC, kind
                    LIMIT ?
                    """,
                    [MemoryStatus.CANDIDATE.value, *args, limit],
                )
            ]
            rows = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT id, kind, title, source_event_ids_json, updated_at
                    FROM capsules
                    WHERE status = ?{scope_where}
                    ORDER BY updated_at DESC
                    LIMIT 2000
                    """,
                    [MemoryStatus.CANDIDATE.value, *args],
                )
            ]

        total = max(1, candidate)
        for row in by_kind:
            row["share"] = row["count"] / total
        by_title_pattern = _title_patterns(rows, total=total, limit=limit)
        totals = {
            "candidate": candidate,
            "stable": stable,
            "candidate_stable_ratio": candidate / max(1, stable),
            "dominant_kind": by_kind[0]["kind"] if by_kind else None,
            "dominant_kind_count": by_kind[0]["count"] if by_kind else 0,
            "dominant_pattern": by_title_pattern[0]["pattern"] if by_title_pattern else None,
            "dominant_pattern_count": by_title_pattern[0]["count"] if by_title_pattern else 0,
        }
        return CandidatePressureReport(
            scope=scope,
            totals=totals,
            by_kind=by_kind,
            by_title_pattern=by_title_pattern,
            recommendations=_recommend(totals, by_kind, by_title_pattern),
        )


def _title_patterns(rows: list[dict[str, Any]], *, total: int, limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        pattern = _title_pattern(str(row["title"]), str(row["kind"]))
        bucket = buckets.setdefault(
            pattern,
            {
                "pattern": pattern,
                "count": 0,
                "kinds": {},
                "examples": [],
            },
        )
        bucket["count"] += 1
        bucket["kinds"][row["kind"]] = bucket["kinds"].get(row["kind"], 0) + 1
        if len(bucket["examples"]) < 3:
            bucket["examples"].append({"id": row["id"], "kind": row["kind"], "title": row["title"]})
    result = []
    for bucket in buckets.values():
        bucket["share"] = bucket["count"] / total
        bucket["kinds"] = dict(sorted(bucket["kinds"].items()))
        result.append(bucket)
    result.sort(key=lambda row: (row["count"], row["pattern"]), reverse=True)
    return result[:limit]


def _title_pattern(title: str, kind: str) -> str:
    lowered = title.lower()
    if lowered.startswith("untracked file "):
        return "untracked file artifact episodes"
    if lowered.startswith("file artifact: "):
        return "file artifact episodes"
    if lowered.startswith("command: "):
        return "command episodes"
    if lowered.startswith("git status for "):
        return "git status episodes"
    if lowered.startswith("potential memory conflict: "):
        return "conflict candidates"
    if lowered.startswith("procedure candidate: "):
        return "procedure candidates"
    if lowered.startswith("decision: "):
        return "decision candidates"
    if kind == "episode":
        return "other episode candidates"
    return f"{kind} candidates"


def _recommend(
    totals: dict[str, Any],
    by_kind: list[dict[str, Any]],
    by_title_pattern: list[dict[str, Any]],
) -> list[str]:
    recommendations = []
    ratio = float(totals["candidate_stable_ratio"])
    if ratio > 2:
        recommendations.append("Candidate/stable ratio is high; prefer consolidation and explicit promotion over broad recall.")
    dominant_kind = by_kind[0] if by_kind else None
    if dominant_kind and dominant_kind["kind"] == "episode" and dominant_kind["share"] > 0.5:
        recommendations.append("Episode candidates dominate; add or run episode-to-summary consolidation before promoting more memories.")
    dominant_pattern = by_title_pattern[0] if by_title_pattern else None
    if dominant_pattern and "artifact" in dominant_pattern["pattern"] and dominant_pattern["share"] > 0.25:
        recommendations.append("Artifact-like candidates dominate; keep latest stable artifact summaries and avoid promoting raw file episodes.")
    if dominant_pattern and dominant_pattern["pattern"] == "command episodes" and dominant_pattern["share"] > 0.2:
        recommendations.append("Command episodes dominate; summarize repeated command outcomes into stable procedure/failure memories.")
    if not recommendations:
        recommendations.append("Candidate pressure is within current operating budgets.")
    return recommendations
