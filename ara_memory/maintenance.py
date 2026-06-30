from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.storage import MemoryStore


@dataclass(slots=True)
class MaintenanceReport:
    before: dict[str, int]
    after: dict[str, int]
    sqlite_integrity: str
    operations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "sqlite_integrity": self.sqlite_integrity,
            "operations": self.operations,
            "bytes_reclaimed": max(0, self.before["storage_bytes"] - self.after["storage_bytes"]),
            "live_bytes_reclaimed": max(
                0,
                self.before.get("live_storage_bytes", self.before["storage_bytes"])
                - self.after.get("live_storage_bytes", self.after["storage_bytes"]),
            ),
        }


def run_maintenance(store: MemoryStore, *, vacuum: bool = True) -> MaintenanceReport:
    store.init()
    before = _stats_with_bytes(store)
    operations: list[str] = []
    with store.session() as conn:
        conn.execute("INSERT INTO events_fts(events_fts) VALUES ('optimize')")
        conn.execute("INSERT INTO capsules_fts(capsules_fts) VALUES ('optimize')")
        operations.append("fts_optimize")

    conn = store.connect()
    try:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        operations.append(f"wal_checkpoint={tuple(checkpoint) if checkpoint else 'unknown'}")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()

    if vacuum:
        conn = store.connect()
        try:
            conn.isolation_level = None
            conn.execute("VACUUM")
        finally:
            conn.close()
        operations.append("vacuum")

    after = _stats_with_bytes(store)
    return MaintenanceReport(before=before, after=after, sqlite_integrity=integrity, operations=operations)


def _stats_with_bytes(store: MemoryStore) -> dict[str, int]:
    stats = store.stats()
    stats.update(store.storage_breakdown())
    return stats
