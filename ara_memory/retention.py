from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.models import MemoryStatus
from ara_memory.storage import MemoryStore


COLD_STATUSES = {
    MemoryStatus.SUPERSEDED.value,
    MemoryStatus.REJECTED.value,
    MemoryStatus.QUARANTINED.value,
}
STORAGE_PRESSURE_BYTES = 50_000_000


@dataclass(slots=True)
class RetentionReport:
    scope: str | None
    totals: dict[str, Any]
    by_status: list[dict[str, Any]]
    by_kind_status: list[dict[str, Any]]
    cold_candidates: list[dict[str, Any]]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "totals": self.totals,
            "by_status": self.by_status,
            "by_kind_status": self.by_kind_status,
            "cold_candidates": self.cold_candidates,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Memory Retention: {scope}"]
        lines.append(
            "totals: "
            f"events={self.totals['events']}, capsules={self.totals['capsules']}, "
            f"cold={self.totals['cold_capsules']}, "
            f"live_storage_bytes={self.totals.get('live_storage_bytes', self.totals['storage_bytes'])}, "
            f"storage_bytes={self.totals['storage_bytes']}"
        )
        lines.append("## By Status")
        for row in self.by_status:
            lines.append(f"- {row['status']}: {row['count']}")
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        if self.cold_candidates:
            lines.append("## Cold Candidates")
            for row in self.cold_candidates[:10]:
                lines.append(f"- {row['id']} [{row['kind']}/{row['status']}] {row['title']}")
        return "\n".join(lines)


class RetentionAnalyzer:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(self, *, scope: str | None = None, cold_limit: int = 20) -> RetentionReport:
        self.store.init()
        with self.store.session() as conn:
            scope_clause = "WHERE scope = ?" if scope else ""
            scope_args = [scope] if scope else []
            events = conn.execute(f"SELECT COUNT(*) FROM events {scope_clause}", scope_args).fetchone()[0]
            capsules = conn.execute(f"SELECT COUNT(*) FROM capsules {scope_clause}", scope_args).fetchone()[0]
            cold_placeholders = ",".join("?" for _ in COLD_STATUSES)
            cold_args = [*COLD_STATUSES]
            if scope:
                cold_args.append(scope)
                cold_where = f"WHERE status IN ({cold_placeholders}) AND scope = ?"
            else:
                cold_where = f"WHERE status IN ({cold_placeholders})"
            cold_capsules = conn.execute(f"SELECT COUNT(*) FROM capsules {cold_where}", cold_args).fetchone()[0]

            by_status = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT status, COUNT(*) AS count
                    FROM capsules
                    {scope_clause}
                    GROUP BY status
                    ORDER BY count DESC, status
                    """,
                    scope_args,
                )
            ]
            by_kind_status = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT kind, status, COUNT(*) AS count
                    FROM capsules
                    {scope_clause}
                    GROUP BY kind, status
                    ORDER BY count DESC, kind, status
                    """,
                    scope_args,
                )
            ]
            cold_candidates = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT id, kind, status, title, updated_at
                    FROM capsules
                    {cold_where}
                    ORDER BY updated_at ASC
                    LIMIT ?
                    """,
                    [*cold_args, cold_limit],
                )
            ]

        storage = self.store.storage_breakdown()
        totals: dict[str, Any] = {
            "events": events,
            "capsules": capsules,
            "cold_capsules": cold_capsules,
        }
        totals.update(storage)
        return RetentionReport(
            scope=scope,
            totals=totals,
            by_status=by_status,
            by_kind_status=by_kind_status,
            cold_candidates=cold_candidates,
            recommendations=_recommend(totals, by_status),
        )


def _recommend(totals: dict[str, Any], by_status: list[dict[str, Any]]) -> list[str]:
    recommendations = []
    status_counts = {row["status"]: row["count"] for row in by_status}
    cold = totals["cold_capsules"]
    capsules = max(1, totals["capsules"])
    live_storage_bytes = totals.get("live_storage_bytes", totals["storage_bytes"])
    backup_or_evidence_bytes = totals.get("backup_bytes", 0) + totals.get("evidence_storage_bytes", 0)
    if cold:
        recommendations.append(
            f"{cold} cold capsules ({cold / capsules:.0%}) are superseded/rejected/quarantined; keep them for provenance unless storage pressure requires pruning."
        )
    if status_counts.get(MemoryStatus.CANDIDATE.value, 0) > status_counts.get(MemoryStatus.STABLE.value, 0) * 2:
        recommendations.append("Candidate capsules dominate stable memory; run sleep/review before relying on recall quality.")
    if live_storage_bytes > STORAGE_PRESSURE_BYTES:
        recommendations.append(
            "Live memory storage exceeds 50MB; run maintenance, create a verified backup, and pass retention-cycle before considering destructive pruning."
        )
    elif totals["storage_bytes"] > STORAGE_PRESSURE_BYTES and backup_or_evidence_bytes:
        recommendations.append(
            "Total memory folder exceeds 50MB while live memory is below threshold; rotate verified backups/export evidence before touching live memory."
        )
    if not recommendations:
        recommendations.append("Retention pressure is low; prefer maintenance and backup over destructive pruning.")
    return recommendations
