from __future__ import annotations

import json
import re
import sqlite3
from hashlib import sha256
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from ara_memory.models import Capsule, Event, EventKind, MemoryStatus, utc_now


SCHEMA_VERSION = 3
SAFE_SCOPE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memory_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL,
  scope TEXT NOT NULL,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_fingerprints (
  fingerprint TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
  id UNINDEXED,
  text,
  kind,
  scope,
  source
);

CREATE TABLE IF NOT EXISTS capsules (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  scope TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence REAL NOT NULL,
  salience REAL NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source_event_ids_json TEXT NOT NULL,
  tags_json TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS capsules_fts USING fts5(
  id UNINDEXED,
  title,
  body,
  kind,
  scope,
  tags
);

CREATE TABLE IF NOT EXISTS temporal_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  scope TEXT NOT NULL,
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  source_capsule_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_events_scope_time ON events(scope, created_at);
CREATE INDEX IF NOT EXISTS idx_capsules_scope_kind ON capsules(scope, kind, status);
CREATE INDEX IF NOT EXISTS idx_edges_subject ON temporal_edges(scope, subject, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_object ON temporal_edges(scope, object, predicate);

CREATE TABLE IF NOT EXISTS memory_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  action TEXT NOT NULL,
  capsule_id TEXT,
  scope TEXT NOT NULL,
  reason TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_actions_capsule ON memory_actions(capsule_id, created_at);

CREATE TABLE IF NOT EXISTS consolidation_runs (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  candidates_seen INTEGER NOT NULL,
  promoted INTEGER NOT NULL,
  rejected INTEGER NOT NULL,
  merged INTEGER NOT NULL,
  conflicts INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prune_approvals (
  id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  scope TEXT,
  backup_path TEXT NOT NULL,
  cold_export_path TEXT NOT NULL,
  recall_queries_json TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  shadow_json TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_prune_approvals_status ON prune_approvals(status, expires_at);

CREATE TABLE IF NOT EXISTS irreversible_operations (
  id TEXT PRIMARY KEY,
  operation TEXT NOT NULL,
  scope TEXT,
  approval_id TEXT,
  status TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  FOREIGN KEY(approval_id) REFERENCES prune_approvals(id)
);

CREATE INDEX IF NOT EXISTS idx_irreversible_operations_scope ON irreversible_operations(scope, created_at);

CREATE TABLE IF NOT EXISTS memory_quality_scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  capsule_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  status TEXT NOT NULL,
  quality_score REAL NOT NULL,
  decay_score REAL NOT NULL,
  action TEXT NOT NULL,
  reasons_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_quality_capsule_time ON memory_quality_scores(capsule_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quality_scope_action ON memory_quality_scores(scope, action, created_at);

CREATE TABLE IF NOT EXISTS memory_review_queue (
  id TEXT PRIMARY KEY,
  capsule_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  action TEXT NOT NULL,
  priority REAL NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_review_queue_status ON memory_review_queue(scope, status, priority);
"""


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ledger_dir = root / "ledger"
        self.archive_dir = root / "archive"
        self.hot_dir = root / "hot"
        self.db_path = root / "memory.db"

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.hot_dir.mkdir(parents=True, exist_ok=True)
        with self.session() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                """
                INSERT INTO memory_meta(key, value, updated_at)
                VALUES ('schema_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value=excluded.value,
                  updated_at=excluded.updated_at
                """,
                (str(SCHEMA_VERSION), utc_now()),
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def append_event(self, event: Event) -> Event:
        self.init()
        fingerprint = _event_fingerprint(event)
        ledger_path = self.ledger_dir / "events.jsonl"

        with self.session() as conn:
            existing = conn.execute(
                """
                SELECT e.*
                FROM event_fingerprints f
                JOIN events e ON e.id = f.event_id
                WHERE f.fingerprint = ?
                """,
                (fingerprint,),
            ).fetchone()
            if existing is not None:
                return row_to_event(existing)

            with ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")

            conn.execute(
                """
                INSERT INTO events(id, kind, text, source, scope, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.kind.value,
                    event.text,
                    event.source,
                    event.scope,
                    event.created_at,
                    json.dumps(event.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.execute(
                "INSERT INTO events_fts(id, text, kind, scope, source) VALUES (?, ?, ?, ?, ?)",
                (event.id, event.text, event.kind.value, event.scope, event.source),
            )
            conn.execute(
                "INSERT INTO event_fingerprints(fingerprint, event_id, created_at) VALUES (?, ?, ?)",
                (fingerprint, event.id, utc_now()),
            )
            return event

    def upsert_capsule(self, capsule: Capsule) -> None:
        self.init()
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO capsules(
                  id, kind, title, body, scope, status, confidence, salience,
                  created_at, updated_at, source_event_ids_json, tags_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  title=excluded.title,
                  body=excluded.body,
                  status=excluded.status,
                  confidence=excluded.confidence,
                  salience=excluded.salience,
                  updated_at=excluded.updated_at,
                  source_event_ids_json=excluded.source_event_ids_json,
                  tags_json=excluded.tags_json
                """,
                (
                    capsule.id,
                    capsule.kind.value,
                    capsule.title,
                    capsule.body,
                    capsule.scope,
                    capsule.status.value,
                    capsule.confidence,
                    capsule.salience,
                    capsule.created_at,
                    capsule.updated_at,
                    json.dumps(capsule.source_event_ids, ensure_ascii=False),
                    json.dumps(capsule.tags, ensure_ascii=False),
                ),
            )
            conn.execute("DELETE FROM capsules_fts WHERE id = ?", (capsule.id,))
            conn.execute(
                "INSERT INTO capsules_fts(id, title, body, kind, scope, tags) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    capsule.id,
                    capsule.title,
                    capsule.body,
                    capsule.kind.value,
                    capsule.scope,
                    " ".join(capsule.tags),
                ),
            )

    def list_unconsolidated_events(self, limit: int = 100) -> list[sqlite3.Row]:
        self.init()
        with self.session() as conn:
            return list(
                conn.execute(
                    """
                    SELECT e.*
                    FROM events e
                    LEFT JOIN capsules c
                      ON c.source_event_ids_json LIKE '%' || e.id || '%'
                    WHERE c.id IS NULL
                    ORDER BY e.created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
            )

    def search_capsules(
        self,
        query: str,
        *,
        scope: str,
        limit: int,
        include_global: bool = True,
    ) -> list[sqlite3.Row]:
        self.init()
        fts_query = _fts_query(query)
        scope_filter = "(c.scope = ? OR c.scope = 'global')" if include_global else "c.scope = ?"
        with self.session() as conn:
            if fts_query:
                rows = list(
                    conn.execute(
                        f"""
                        SELECT c.*, bm25(capsules_fts) AS bm25_score
                        FROM capsules_fts
                        JOIN capsules c ON c.id = capsules_fts.id
                        WHERE capsules_fts MATCH ?
                          AND c.status IN ('candidate', 'stable')
                          AND {scope_filter}
                        ORDER BY bm25_score ASC, c.salience DESC, c.updated_at DESC
                        LIMIT ?
                        """,
                        (fts_query, scope, limit),
                    )
                )
                if rows:
                    return rows
            return list(
                conn.execute(
                    f"""
                    SELECT *, 0.0 AS bm25_score
                    FROM capsules
                    WHERE status IN ('candidate', 'stable')
                      AND {"(scope = ? OR scope = 'global')" if include_global else "scope = ?"}
                    ORDER BY salience DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (scope, limit),
                )
            )

    def add_edge(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        scope: str,
        source_capsule_id: str | None,
        confidence: float,
    ) -> None:
        self.init()
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO temporal_edges(
                  subject, predicate, object, scope, valid_from, source_capsule_id,
                  confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject,
                    predicate,
                    object_,
                    scope,
                    utc_now(),
                    source_capsule_id,
                    confidence,
                    utc_now(),
                ),
            )

    def graph_neighbors(
        self,
        terms: Iterable[str],
        *,
        scope: str,
        limit: int = 30,
        include_global: bool = True,
    ) -> list[sqlite3.Row]:
        self.init()
        terms = [t for t in terms if len(t) >= 3]
        if not terms:
            return []
        clauses = " OR ".join(["subject LIKE ? OR object LIKE ?" for _ in terms])
        args: list[Any] = []
        for term in terms:
            args.extend([f"%{term}%", f"%{term}%"])
        args.extend([scope, limit])
        scope_filter = "(scope = ? OR scope = 'global')" if include_global else "scope = ?"
        with self.session() as conn:
            return list(
                conn.execute(
                    f"""
                    SELECT *
                    FROM temporal_edges
                    WHERE ({clauses})
                      AND {scope_filter}
                      AND valid_to IS NULL
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?
                    """,
                    args,
                )
            )

    def stats(self) -> dict[str, int]:
        self.init()
        with self.session() as conn:
            return {
                "schema_version": int(
                    conn.execute(
                        "SELECT value FROM memory_meta WHERE key = 'schema_version'"
                    ).fetchone()[0]
                ),
                "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "capsules": conn.execute("SELECT COUNT(*) FROM capsules").fetchone()[0],
                "stable_capsules": conn.execute(
                    "SELECT COUNT(*) FROM capsules WHERE status = ?", (MemoryStatus.STABLE.value,)
                ).fetchone()[0],
                "edges": conn.execute("SELECT COUNT(*) FROM temporal_edges").fetchone()[0],
            }

    def schema_version(self) -> int:
        self.init()
        with self.session() as conn:
            row = conn.execute("SELECT value FROM memory_meta WHERE key = 'schema_version'").fetchone()
            return int(row[0]) if row else 0

    def storage_bytes(self) -> int:
        self.init()
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def update_capsule_status(
        self,
        capsule_id: str,
        status: MemoryStatus,
        *,
        actor: str = "manual",
        reason: str = "",
    ) -> bool:
        self.init()
        with self.session() as conn:
            row = conn.execute("SELECT scope FROM capsules WHERE id = ?", (capsule_id,)).fetchone()
            if row is None:
                return False
            cur = conn.execute(
                "UPDATE capsules SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, utc_now(), capsule_id),
            )
            conn.execute(
                """
                INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (status.value, capsule_id, row["scope"], reason, actor, utc_now()),
            )
            _invalidate_hot_scope(self.hot_dir, row["scope"])
            return cur.rowcount > 0

    def get_capsule(self, capsule_id: str) -> sqlite3.Row | None:
        self.init()
        with self.session() as conn:
            return conn.execute("SELECT * FROM capsules WHERE id = ?", (capsule_id,)).fetchone()

    def get_events(self, event_ids: list[str]) -> list[sqlite3.Row]:
        self.init()
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        with self.session() as conn:
            return list(
                conn.execute(
                    f"SELECT * FROM events WHERE id IN ({placeholders})",
                    event_ids,
                )
            )

    def audit_rows(self) -> dict[str, list[sqlite3.Row]]:
        self.init()
        with self.session() as conn:
            instruction_like = list(
                conn.execute(
                    """
                    SELECT *
                    FROM capsules
                    WHERE lower(body) LIKE '%ignore previous%'
                       OR lower(body) LIKE '%system prompt%'
                       OR lower(body) LIKE '%developer message%'
                       OR lower(body) LIKE '%permanent instruction%'
                       OR lower(body) LIKE '%always obey%'
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """
                )
            )
            low_confidence_stable = list(
                conn.execute(
                    """
                    SELECT *
                    FROM capsules
                    WHERE status = 'stable' AND confidence < 0.65
                    ORDER BY confidence ASC
                    LIMIT 50
                    """
                )
            )
            missing_provenance = list(
                conn.execute(
                    """
                    SELECT *
                    FROM capsules
                    WHERE source_event_ids_json = '[]'
                    LIMIT 50
                    """
                )
            )
            return {
                "instruction_like": instruction_like,
                "low_confidence_stable": low_confidence_stable,
                "missing_provenance": missing_provenance,
            }
    def list_capsules(
        self,
        *,
        scope: str | None = None,
        status: MemoryStatus | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        self.init()
        clauses = []
        args: list[Any] = []
        if scope:
            clauses.append("scope = ?")
            args.append(scope)
        if status:
            clauses.append("status = ?")
            args.append(status.value)
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)
        with self.session() as conn:
            return list(
                conn.execute(
                    f"""
                    SELECT *
                    FROM capsules
                    {where}
                    ORDER BY salience DESC, updated_at DESC
                    LIMIT ?
                    """,
                    args,
                )
            )

    def list_actions(self, *, limit: int = 50) -> list[sqlite3.Row]:
        self.init()
        with self.session() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM memory_actions
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            )

    def record_consolidation_run(
        self,
        *,
        run_id: str,
        scope: str,
        candidates_seen: int,
        promoted: int,
        rejected: int,
        merged: int,
        conflicts: int,
        notes: str,
    ) -> None:
        self.init()
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO consolidation_runs(
                  id, scope, candidates_seen, promoted, rejected, merged, conflicts, created_at, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, scope, candidates_seen, promoted, rejected, merged, conflicts, utc_now(), notes),
            )

    def list_consolidation_runs(self, *, limit: int = 20) -> list[sqlite3.Row]:
        self.init()
        with self.session() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM consolidation_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            )


def row_to_capsule(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["source_event_ids"] = json.loads(data.pop("source_event_ids_json"))
    data["tags"] = json.loads(data.pop("tags_json"))
    return data


def row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        kind=EventKind(row["kind"]),
        text=row["text"],
        source=row["source"],
        scope=row["scope"],
        created_at=row["created_at"],
        metadata=json.loads(row["metadata_json"]),
    )


def _event_fingerprint(event: Event) -> str:
    payload = {
        "kind": event.kind.value,
        "text": event.text,
        "source": event.source,
        "scope": event.scope,
        "metadata": event.metadata,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _fts_query(query: str) -> str:
    tokens = []
    for token in query.replace('"', " ").replace("'", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum() or ch == "_")
        if len(cleaned) >= 2:
            tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens[:12])


def _invalidate_hot_scope(hot_dir: Path, scope: str) -> None:
    if not hot_dir.exists():
        return
    if scope == "global":
        targets = list(hot_dir.glob("*.md"))
    else:
        safe = SAFE_SCOPE_RE.sub("_", scope).strip("._") or "global"
        targets = [hot_dir / f"{safe}.md"]
    for path in targets:
        try:
            path.unlink()
        except FileNotFoundError:
            continue

