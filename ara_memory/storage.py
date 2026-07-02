from __future__ import annotations

import json
import re
import sqlite3
from hashlib import sha256
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from ara_memory.compressors import extract_keywords, is_search_term
from ara_memory.models import Capsule, Event, EventKind, MemoryStatus, new_id, utc_now
from ara_memory.projection import search_projection


SCHEMA_VERSION = 20
ACTIVE_FTS_STATUSES = {MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value}
SAFE_SCOPE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SQLITE_IN_CHUNK_SIZE = 500
STORAGE_CATEGORY_KEYS = (
    "db_bytes",
    "ledger_bytes",
    "hot_bytes",
    "spool_bytes",
    "archive_object_bytes",
    "cold_export_bytes",
    "retention_cycle_bytes",
    "archive_other_bytes",
    "backup_bytes",
    "support_bytes",
)
LIVE_STORAGE_KEYS = {
    "db_bytes",
    "ledger_bytes",
    "hot_bytes",
    "spool_bytes",
    "archive_object_bytes",
}
EVIDENCE_STORAGE_KEYS = {
    "cold_export_bytes",
    "retention_cycle_bytes",
    "archive_other_bytes",
}


class _SourceEventUpdateConflict(Exception):
    pass


def _status_value(status: str | MemoryStatus) -> str:
    return status.value if isinstance(status, MemoryStatus) else str(status)


def _unique_source_event_ids(source_event_ids: Iterable[Any] | None) -> list[str]:
    return list(dict.fromkeys(str(event_id) for event_id in source_event_ids or [] if str(event_id)))


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

CREATE TABLE IF NOT EXISTS capsule_source_events (
  capsule_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  PRIMARY KEY(capsule_id, event_id),
  FOREIGN KEY(capsule_id) REFERENCES capsules(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS provenance_witnesses (
  id TEXT PRIMARY KEY,
  capsule_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  action TEXT NOT NULL,
  actor TEXT NOT NULL,
  reason TEXT NOT NULL,
  original_source_event_ids_json TEXT NOT NULL,
  retained_source_event_ids_json TEXT NOT NULL,
  original_source_event_count INTEGER NOT NULL,
  retained_source_event_count INTEGER NOT NULL,
  original_source_event_digest TEXT NOT NULL,
  retained_source_event_digest TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(capsule_id) REFERENCES capsules(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS capsule_redaction_witnesses (
  id TEXT PRIMARY KEY,
  capsule_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  action TEXT NOT NULL,
  actor TEXT NOT NULL,
  reason TEXT NOT NULL,
  review_queue_id TEXT,
  original_title_digest TEXT NOT NULL,
  redacted_title_digest TEXT NOT NULL,
  original_body_digest TEXT NOT NULL,
  redacted_body_digest TEXT NOT NULL,
  original_tags_digest TEXT NOT NULL,
  redacted_tags_digest TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(capsule_id) REFERENCES capsules(id) ON DELETE CASCADE
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

CREATE TABLE IF NOT EXISTS relation_nodes (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  normalized_label TEXT NOT NULL,
  scope TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(scope, normalized_label)
);

CREATE TABLE IF NOT EXISTS relation_edges (
  id TEXT PRIMARY KEY,
  subject_node_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  predicate_norm TEXT NOT NULL,
  object_node_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  source_capsule_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence_count INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(subject_node_id) REFERENCES relation_nodes(id),
  FOREIGN KEY(object_node_id) REFERENCES relation_nodes(id),
  FOREIGN KEY(source_capsule_id) REFERENCES capsules(id),
  UNIQUE(subject_node_id, predicate_norm, object_node_id, scope, source_capsule_id)
);

CREATE INDEX IF NOT EXISTS idx_events_scope_time ON events(scope, created_at);
CREATE INDEX IF NOT EXISTS idx_capsules_scope_kind ON capsules(scope, kind, status);
CREATE INDEX IF NOT EXISTS idx_capsules_scope_status ON capsules(scope, status);
CREATE INDEX IF NOT EXISTS idx_capsules_status_updated ON capsules(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_capsules_scope_status_updated ON capsules(scope, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_capsule_source_events_event ON capsule_source_events(event_id, capsule_id);
CREATE INDEX IF NOT EXISTS idx_provenance_witnesses_capsule ON provenance_witnesses(capsule_id, created_at);
CREATE INDEX IF NOT EXISTS idx_capsule_redaction_witnesses_capsule ON capsule_redaction_witnesses(capsule_id, created_at);
CREATE INDEX IF NOT EXISTS idx_edges_subject ON temporal_edges(scope, subject, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_object ON temporal_edges(scope, object, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_source_capsule_active ON temporal_edges(scope, source_capsule_id, valid_to, confidence, created_at);
CREATE INDEX IF NOT EXISTS idx_relation_nodes_scope_label ON relation_nodes(scope, normalized_label);
CREATE INDEX IF NOT EXISTS idx_relation_edges_scope_subject ON relation_edges(scope, subject_node_id, predicate_norm);
CREATE INDEX IF NOT EXISTS idx_relation_edges_scope_object ON relation_edges(scope, object_node_id, predicate_norm);
CREATE INDEX IF NOT EXISTS idx_relation_edges_source ON relation_edges(scope, source_capsule_id, confidence, updated_at);

CREATE TABLE IF NOT EXISTS relation_merge_approvals (
  id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL,
  candidates_json TEXT NOT NULL,
  relation_fingerprint TEXT NOT NULL,
  rollback_witness_json TEXT NOT NULL,
  candidate_limit INTEGER NOT NULL DEFAULT 20,
  threshold REAL NOT NULL,
  node_limit INTEGER NOT NULL,
  include_global INTEGER NOT NULL DEFAULT 0,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_relation_merge_approvals_status ON relation_merge_approvals(scope, status, expires_at);

CREATE TABLE IF NOT EXISTS relation_merge_witnesses (
  id TEXT PRIMARY KEY,
  approval_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  canonical_node_id TEXT NOT NULL,
  candidate_node_id TEXT NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(approval_id) REFERENCES relation_merge_approvals(id)
);

CREATE INDEX IF NOT EXISTS idx_relation_merge_witnesses_approval ON relation_merge_witnesses(approval_id, created_at);

CREATE TABLE IF NOT EXISTS relation_merge_review_queue (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  approval_id TEXT,
  witness_id TEXT,
  action TEXT NOT NULL,
  priority REAL NOT NULL,
  reason TEXT NOT NULL,
  review_status TEXT NOT NULL,
  review_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY(approval_id) REFERENCES relation_merge_approvals(id),
  FOREIGN KEY(witness_id) REFERENCES relation_merge_witnesses(id)
);

CREATE INDEX IF NOT EXISTS idx_relation_merge_review_queue_status ON relation_merge_review_queue(scope, status, priority);
CREATE INDEX IF NOT EXISTS idx_relation_merge_review_queue_witness ON relation_merge_review_queue(witness_id, status);

CREATE TABLE IF NOT EXISTS reconsolidation_approvals (
  id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL,
  query TEXT NOT NULL,
  frame_json TEXT NOT NULL,
  frame_fingerprint TEXT NOT NULL,
  params_json TEXT NOT NULL,
  rollback_witness_json TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_approvals_status ON reconsolidation_approvals(scope, status, expires_at);

CREATE TABLE IF NOT EXISTS reconsolidation_witnesses (
  id TEXT PRIMARY KEY,
  approval_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  query TEXT NOT NULL,
  capsule_id TEXT NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  frame_fingerprint TEXT NOT NULL,
  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(approval_id) REFERENCES reconsolidation_approvals(id),
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_witnesses_approval ON reconsolidation_witnesses(approval_id, created_at);
CREATE INDEX IF NOT EXISTS idx_reconsolidation_witnesses_scope ON reconsolidation_witnesses(scope, created_at);

CREATE TABLE IF NOT EXISTS reconsolidation_review_queue (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  approval_id TEXT,
  witness_id TEXT,
  capsule_id TEXT,
  action TEXT NOT NULL,
  priority REAL NOT NULL,
  reason TEXT NOT NULL,
  review_status TEXT NOT NULL,
  review_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY(approval_id) REFERENCES reconsolidation_approvals(id),
  FOREIGN KEY(witness_id) REFERENCES reconsolidation_witnesses(id),
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_review_queue_status ON reconsolidation_review_queue(scope, status, priority);
CREATE INDEX IF NOT EXISTS idx_reconsolidation_review_queue_witness ON reconsolidation_review_queue(witness_id, status);

CREATE TABLE IF NOT EXISTS reconsolidation_rollback_approvals (
  id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL,
  backup_path TEXT NOT NULL,
  witness_id TEXT NOT NULL,
  approval_id TEXT NOT NULL,
  capsule_id TEXT NOT NULL,
  shadow_json TEXT NOT NULL,
  backup_identity_json TEXT NOT NULL,
  capsule_snapshot_json TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at TEXT,
  FOREIGN KEY(witness_id) REFERENCES reconsolidation_witnesses(id),
  FOREIGN KEY(approval_id) REFERENCES reconsolidation_approvals(id),
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_rollback_approvals_status
ON reconsolidation_rollback_approvals(status, expires_at);

CREATE TABLE IF NOT EXISTS reconsolidation_rollback_witnesses (
  id TEXT PRIMARY KEY,
  rollback_approval_id TEXT NOT NULL,
  reconsolidation_witness_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  capsule_id TEXT NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(rollback_approval_id) REFERENCES reconsolidation_rollback_approvals(id),
  FOREIGN KEY(reconsolidation_witness_id) REFERENCES reconsolidation_witnesses(id),
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_rollback_witnesses_scope
ON reconsolidation_rollback_witnesses(scope, created_at);

CREATE TABLE IF NOT EXISTS reconsolidation_exception_witnesses (
  id TEXT PRIMARY KEY,
  reconsolidation_witness_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  capsule_id TEXT NOT NULL,
  field TEXT NOT NULL,
  before_digest TEXT NOT NULL,
  after_digest TEXT NOT NULL,
  reason TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(reconsolidation_witness_id) REFERENCES reconsolidation_witnesses(id),
  FOREIGN KEY(capsule_id) REFERENCES capsules(id)
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_exception_witnesses_target
ON reconsolidation_exception_witnesses(reconsolidation_witness_id, capsule_id, field, status);

CREATE TABLE IF NOT EXISTS reconsolidation_action_approvals (
  id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL,
  action TEXT NOT NULL,
  query TEXT NOT NULL,
  backup_path TEXT NOT NULL,
  preflight_json TEXT NOT NULL,
  action_gate_json TEXT NOT NULL,
  backup_identity_json TEXT NOT NULL,
  confirmation_required TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reconsolidation_action_approvals_status
ON reconsolidation_action_approvals(scope, action, status, expires_at);

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

CREATE TABLE IF NOT EXISTS working_memory_impacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  cue TEXT NOT NULL,
  cue_terms_json TEXT NOT NULL,
  capsule_id TEXT NOT NULL,
  outcome TEXT NOT NULL,
  helped INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id),
  UNIQUE(event_id, capsule_id)
);

CREATE INDEX IF NOT EXISTS idx_wmi_capsule_scope ON working_memory_impacts(capsule_id, scope, created_at);
CREATE INDEX IF NOT EXISTS idx_wmi_scope_time ON working_memory_impacts(scope, created_at);

CREATE TABLE IF NOT EXISTS recall_policy_impacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  query TEXT NOT NULL,
  query_terms_json TEXT NOT NULL,
  intent TEXT NOT NULL,
  strategy TEXT NOT NULL,
  action_name TEXT NOT NULL,
  outcome TEXT NOT NULL,
  helped INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id),
  UNIQUE(event_id, action_name)
);

CREATE INDEX IF NOT EXISTS idx_rpi_scope_time ON recall_policy_impacts(scope, created_at);
CREATE INDEX IF NOT EXISTS idx_rpi_intent_action ON recall_policy_impacts(scope, intent, action_name, created_at);
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
            old_version = _current_schema_version(conn)
            conn.executescript(SCHEMA)
            if old_version < 4:
                _rebuild_capsule_source_events(conn)
            if old_version < 5:
                _sync_capsules_fts(conn)
            if old_version < 6:
                _sync_working_memory_impacts(conn)
            if old_version < 9:
                _sync_recall_policy_impacts(conn)
            if old_version < 10:
                _sync_capsules_fts(conn)
            if old_version < 11:
                _sync_relation_graph_from_temporal_edges(conn)
            if old_version < 13:
                _ensure_relation_merge_v13_columns(conn)
            if old_version < 16:
                _ensure_reconsolidation_v16_columns(conn)
            if old_version < 18:
                _ensure_reconsolidation_rollback_v18_tables(conn)
            if old_version < 19:
                _ensure_reconsolidation_exception_v19_tables(conn)
            if old_version < 20:
                _ensure_reconsolidation_action_v20_tables(conn)
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
        conn.execute("PRAGMA foreign_keys=ON")
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
                _sync_working_memory_impact_event(conn, row_to_event(existing))
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
            _sync_working_memory_impact_event(conn, event)
            _sync_recall_policy_impact_event(conn, event)
            return event

    def record_working_memory_impact(self, event: Event) -> None:
        self.init()
        with self.session() as conn:
            _sync_working_memory_impact_event(conn, event)

    def record_recall_policy_impact(self, event: Event) -> None:
        self.init()
        with self.session() as conn:
            _sync_recall_policy_impact_event(conn, event)

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
            _sync_capsule_fts_payload(
                conn,
                capsule_id=capsule.id,
                title=capsule.title,
                body=capsule.body,
                kind=capsule.kind.value,
                scope=capsule.scope,
                tags=capsule.tags,
                status=capsule.status.value,
            )
            conn.execute("DELETE FROM capsule_source_events WHERE capsule_id = ?", (capsule.id,))
            conn.executemany(
                "INSERT OR IGNORE INTO capsule_source_events(capsule_id, event_id) VALUES (?, ?)",
                [(capsule.id, event_id) for event_id in sorted(set(capsule.source_event_ids))],
            )

    def list_unconsolidated_events(self, limit: int = 100) -> list[sqlite3.Row]:
        self.init()
        with self.session() as conn:
            return list(
                conn.execute(
                    """
                    SELECT e.*
                    FROM events e
                    LEFT JOIN capsule_source_events l
                      ON l.event_id = e.id
                    WHERE l.capsule_id IS NULL
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
            def salience_rows(
                *,
                limit: int,
                recall_match_source: str,
                excluded_ids: list[str],
            ) -> list[sqlite3.Row]:
                if limit <= 0:
                    return []
                salience_scope_filter = "(scope = ? OR scope = 'global')" if include_global else "scope = ?"
                excluded_clause = ""
                params: list[Any] = [recall_match_source, scope]
                if excluded_ids:
                    placeholders = ", ".join("?" for _ in excluded_ids)
                    excluded_clause = f"AND id NOT IN ({placeholders})"
                    params.extend(excluded_ids)
                params.append(limit)
                return list(
                    conn.execute(
                        f"""
                        SELECT *, NULL AS bm25_score, ? AS recall_match_source
                        FROM capsules
                        WHERE status IN ('candidate', 'stable')
                          AND {salience_scope_filter}
                          {excluded_clause}
                        ORDER BY salience DESC, updated_at DESC
                        LIMIT ?
                        """,
                        tuple(params),
                    )
                )

            if fts_query:
                rows = list(
                    conn.execute(
                        f"""
                        SELECT c.*, bm25(capsules_fts) AS bm25_score, 'fts' AS recall_match_source
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
                if len(rows) >= limit:
                    return rows
                return rows + salience_rows(
                    limit=limit - len(rows),
                    recall_match_source="salience_supplement" if rows else "salience_fallback",
                    excluded_ids=[str(row["id"]) for row in rows],
                )
            return salience_rows(limit=limit, recall_match_source="salience_fallback", excluded_ids=[])

    def recent_capsules(
        self,
        *,
        scope: str,
        limit: int,
        include_global: bool = True,
        excluded_ids: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        self.init()
        if limit <= 0:
            return []
        excluded_ids = excluded_ids or []
        scope_filter = "(scope = ? OR scope = 'global')" if include_global else "scope = ?"
        excluded_clause = ""
        params: list[Any] = [scope]
        if excluded_ids:
            placeholders = ", ".join("?" for _ in excluded_ids)
            excluded_clause = f"AND id NOT IN ({placeholders})"
            params.extend(excluded_ids)
        params.append(limit)
        with self.session() as conn:
            return list(
                conn.execute(
                    f"""
                    SELECT *, NULL AS bm25_score, 'recent_supplement' AS recall_match_source
                    FROM capsules
                    WHERE status IN ('candidate', 'stable')
                      AND {scope_filter}
                      {excluded_clause}
                    ORDER BY updated_at DESC, salience DESC, id ASC
                    LIMIT ?
                    """,
                    tuple(params),
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
            _upsert_relation_edge(
                conn,
                subject=subject,
                predicate=predicate,
                object_=object_,
                scope=scope,
                source_capsule_id=source_capsule_id,
                confidence=confidence,
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

    def graph_activation_edges(
        self,
        terms: Iterable[str],
        *,
        seed_capsule_ids: Iterable[str],
        scope: str,
        limit: int = 80,
        include_global: bool = True,
    ) -> list[sqlite3.Row]:
        self.init()
        terms = [t for t in terms if len(t) >= 3]
        seed_ids = list(dict.fromkeys(str(capsule_id) for capsule_id in seed_capsule_ids if str(capsule_id)))
        if not terms and not seed_ids:
            return []
        scope_filter = "(scope = ? OR scope = 'global')" if include_global else "scope = ?"
        rows: list[sqlite3.Row] = []
        seen_edge_ids: set[int] = set()

        def append_rows(query_rows: list[sqlite3.Row]) -> None:
            for row in query_rows:
                row_id = int(row["id"])
                if row_id in seen_edge_ids:
                    continue
                rows.append(row)
                seen_edge_ids.add(row_id)
                if len(rows) >= limit:
                    break

        with self.session() as conn:
            if seed_ids:
                placeholders = ", ".join("?" for _ in seed_ids)
                seed_limit = min(limit, max(12, min(48, len(seed_ids) * 4)))
                append_rows(
                    list(
                        conn.execute(
                            f"""
                            SELECT *
                            FROM temporal_edges
                            WHERE source_capsule_id IN ({placeholders})
                              AND {scope_filter}
                              AND valid_to IS NULL
                              AND source_capsule_id IS NOT NULL
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT ?
                            """,
                            tuple([*seed_ids, scope, seed_limit]),
                        )
                    )
                )
            if terms and len(rows) < limit:
                term_clause = " OR ".join(["subject LIKE ? OR predicate LIKE ? OR object LIKE ?" for _ in terms])
                args: list[Any] = []
                for term in terms:
                    args.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
                args.extend([scope, limit - len(rows)])
                append_rows(
                    list(
                        conn.execute(
                            f"""
                            SELECT *
                            FROM temporal_edges
                            WHERE ({term_clause})
                              AND {scope_filter}
                              AND valid_to IS NULL
                              AND source_capsule_id IS NOT NULL
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT ?
                            """,
                            tuple(args),
                        )
                    )
                )
        return rows

    def relation_activation_edges(
        self,
        terms: Iterable[str],
        *,
        seed_capsule_ids: Iterable[str],
        scope: str,
        limit: int = 80,
        include_global: bool = True,
    ) -> list[sqlite3.Row]:
        self.init()
        terms = [t for t in terms if len(t) >= 3]
        seed_ids = list(dict.fromkeys(str(capsule_id) for capsule_id in seed_capsule_ids if str(capsule_id)))
        if not terms and not seed_ids:
            return []
        scope_filter = "(re.scope = ? OR re.scope = 'global')" if include_global else "re.scope = ?"
        rows: list[sqlite3.Row] = []
        seen_edge_ids: set[str] = set()

        def append_rows(query_rows: list[sqlite3.Row]) -> None:
            for row in query_rows:
                row_id = str(row["id"])
                if row_id in seen_edge_ids:
                    continue
                rows.append(row)
                seen_edge_ids.add(row_id)
                if len(rows) >= limit:
                    break

        with self.session() as conn:
            if seed_ids:
                placeholders = ", ".join("?" for _ in seed_ids)
                seed_limit = min(limit, max(12, min(48, len(seed_ids) * 4)))
                append_rows(
                    list(
                        conn.execute(
                            f"""
                            SELECT
                              re.id,
                              sn.label AS subject,
                              re.predicate AS predicate,
                              onode.label AS object,
                              re.scope,
                              re.source_capsule_id,
                              re.confidence,
                              re.updated_at AS created_at,
                              re.evidence_count,
                              'relation' AS edge_source
                            FROM relation_edges re
                            JOIN relation_nodes sn ON sn.id = re.subject_node_id
                            JOIN relation_nodes onode ON onode.id = re.object_node_id
                            WHERE re.source_capsule_id IN ({placeholders})
                              AND {scope_filter}
                              AND re.source_capsule_id IS NOT NULL
                            ORDER BY re.confidence DESC, re.evidence_count DESC, re.updated_at DESC
                            LIMIT ?
                            """,
                            tuple([*seed_ids, scope, seed_limit]),
                        )
                    )
                )
            if terms and len(rows) < limit:
                term_clause = " OR ".join(
                    [
                        "sn.normalized_label LIKE ? OR re.predicate_norm LIKE ? OR onode.normalized_label LIKE ?"
                        for _ in terms
                    ]
                )
                args: list[Any] = []
                for term in terms:
                    normalized = _normalize_relation_text(term)
                    args.extend([f"%{normalized}%", f"%{normalized}%", f"%{normalized}%"])
                args.extend([scope, limit - len(rows)])
                append_rows(
                    list(
                        conn.execute(
                            f"""
                            SELECT
                              re.id,
                              sn.label AS subject,
                              re.predicate AS predicate,
                              onode.label AS object,
                              re.scope,
                              re.source_capsule_id,
                              re.confidence,
                              re.updated_at AS created_at,
                              re.evidence_count,
                              'relation' AS edge_source
                            FROM relation_edges re
                            JOIN relation_nodes sn ON sn.id = re.subject_node_id
                            JOIN relation_nodes onode ON onode.id = re.object_node_id
                            WHERE ({term_clause})
                              AND {scope_filter}
                              AND re.source_capsule_id IS NOT NULL
                            ORDER BY re.confidence DESC, re.evidence_count DESC, re.updated_at DESC
                            LIMIT ?
                            """,
                            tuple(args),
                        )
                    )
                )
        return rows

    def relation_nodes_for_merge(
        self,
        *,
        scope: str,
        limit: int = 800,
        include_global: bool = False,
    ) -> list[dict[str, Any]]:
        self.init()
        scope_filter = "(rn.scope = ? OR rn.scope = 'global')" if include_global else "rn.scope = ?"
        with self.session() as conn:
            node_rows = list(
                conn.execute(
                    f"""
                    SELECT
                      rn.id,
                      rn.label,
                      rn.normalized_label,
                      rn.scope,
                      COUNT(re.id) AS degree
                    FROM relation_nodes rn
                    LEFT JOIN relation_edges re
                      ON re.subject_node_id = rn.id
                      OR re.object_node_id = rn.id
                    WHERE {scope_filter}
                    GROUP BY rn.id
                    ORDER BY degree DESC, rn.updated_at DESC
                    LIMIT ?
                    """,
                    (scope, limit),
                )
            )
            node_ids = [str(row["id"]) for row in node_rows]
            neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
            if node_ids:
                for chunk in _chunks(node_ids, SQLITE_IN_CHUNK_SIZE):
                    placeholders = ", ".join("?" for _ in chunk)
                    edge_rows = conn.execute(
                        f"""
                        SELECT
                          subject_node_id,
                          object_node_id,
                          predicate_norm
                        FROM relation_edges
                        WHERE subject_node_id IN ({placeholders})
                           OR object_node_id IN ({placeholders})
                        """,
                        tuple([*chunk, *chunk]),
                    )
                    for row in edge_rows:
                        subject = str(row["subject_node_id"])
                        object_ = str(row["object_node_id"])
                        predicate = str(row["predicate_norm"] or "")
                        if subject in neighbors:
                            neighbors[subject].add(f"{predicate}->{object_}")
                        if object_ in neighbors:
                            neighbors[object_].add(f"{subject}->{predicate}")
            return [
                {
                    "id": str(row["id"]),
                    "label": str(row["label"]),
                    "normalized_label": str(row["normalized_label"]),
                    "scope": str(row["scope"]),
                    "degree": int(row["degree"] or 0),
                    "neighbor_keys": sorted(neighbors.get(str(row["id"]), set()))[:64],
                }
                for row in node_rows
            ]

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
                "relation_nodes": conn.execute("SELECT COUNT(*) FROM relation_nodes").fetchone()[0],
                "relation_edges": conn.execute("SELECT COUNT(*) FROM relation_edges").fetchone()[0],
                "source_event_links": conn.execute("SELECT COUNT(*) FROM capsule_source_events").fetchone()[0],
            }

    def schema_version(self) -> int:
        self.init()
        with self.session() as conn:
            row = conn.execute("SELECT value FROM memory_meta WHERE key = 'schema_version'").fetchone()
            return int(row[0]) if row else 0

    def storage_breakdown(self) -> dict[str, int]:
        self.init()
        breakdown = {key: 0 for key in STORAGE_CATEGORY_KEYS}
        for path in self.root.rglob("*"):
            if path.is_file():
                size = path.stat().st_size
                breakdown[_storage_category(self.root, path)] += size
        breakdown["live_storage_bytes"] = sum(breakdown[key] for key in LIVE_STORAGE_KEYS)
        breakdown["evidence_storage_bytes"] = sum(breakdown[key] for key in EVIDENCE_STORAGE_KEYS)
        breakdown["archive_bytes"] = (
            breakdown["archive_object_bytes"]
            + breakdown["cold_export_bytes"]
            + breakdown["retention_cycle_bytes"]
            + breakdown["archive_other_bytes"]
        )
        breakdown["storage_bytes"] = sum(breakdown[key] for key in STORAGE_CATEGORY_KEYS)
        return breakdown

    def storage_bytes(self) -> int:
        return self.storage_breakdown()["storage_bytes"]

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
            now = utc_now()
            cur = conn.execute(
                "UPDATE capsules SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, capsule_id),
            )
            updated = conn.execute("SELECT * FROM capsules WHERE id = ?", (capsule_id,)).fetchone()
            if updated is not None:
                _sync_capsule_fts_row(conn, updated)
            conn.execute(
                """
                INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (status.value, capsule_id, row["scope"], reason, actor, utc_now()),
            )
            _invalidate_hot_scope(self.hot_dir, row["scope"])
            return cur.rowcount > 0

    def update_capsule_status_if_current(
        self,
        capsule_id: str,
        current_status: MemoryStatus,
        status: MemoryStatus,
        *,
        actor: str = "manual",
        reason: str = "",
    ) -> bool:
        self.init()
        with self.session() as conn:
            row = conn.execute("SELECT scope, status FROM capsules WHERE id = ?", (capsule_id,)).fetchone()
            if row is None or row["status"] != current_status.value:
                return False
            now = utc_now()
            cur = conn.execute(
                "UPDATE capsules SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (status.value, now, capsule_id, current_status.value),
            )
            if cur.rowcount <= 0:
                return False
            updated = conn.execute("SELECT * FROM capsules WHERE id = ?", (capsule_id,)).fetchone()
            if updated is not None:
                _sync_capsule_fts_row(conn, updated)
            conn.execute(
                """
                INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (status.value, capsule_id, row["scope"], reason, actor, utc_now()),
            )
            _invalidate_hot_scope(self.hot_dir, row["scope"])
            return True

    def update_capsule_projection_if_current(
        self,
        capsule_id: str,
        *,
        title: str,
        body: str,
        tags: list[str],
        expected_status: str | MemoryStatus,
        expected_title: str,
        expected_body: str,
        expected_tags: list[str],
        actor: str = "manual",
        reason: str = "",
        action: str = "redact-sensitive-review",
        review_queue_id: str | None = None,
    ) -> bool:
        self.init()
        expected_status_value = _status_value(expected_status)
        expected_tags_json = json.dumps(list(expected_tags), ensure_ascii=False)
        tags_json = json.dumps(list(tags), ensure_ascii=False)
        with self.session() as conn:
            row = conn.execute(
                "SELECT scope, status, title, body, tags_json FROM capsules WHERE id = ?",
                (capsule_id,),
            ).fetchone()
            if row is None:
                return False
            if (
                row["status"] != expected_status_value
                or row["title"] != expected_title
                or row["body"] != expected_body
                or row["tags_json"] != expected_tags_json
            ):
                return False
            now = utc_now()
            cur = conn.execute(
                """
                UPDATE capsules
                SET title = ?, body = ?, tags_json = ?, updated_at = ?
                WHERE id = ? AND status = ? AND title = ? AND body = ? AND tags_json = ?
                """,
                (
                    title,
                    body,
                    tags_json,
                    now,
                    capsule_id,
                    expected_status_value,
                    expected_title,
                    expected_body,
                    expected_tags_json,
                ),
            )
            if cur.rowcount <= 0:
                return False
            updated = conn.execute("SELECT * FROM capsules WHERE id = ?", (capsule_id,)).fetchone()
            if updated is not None:
                _sync_capsule_fts_row(conn, updated)
            conn.execute(
                """
                INSERT INTO capsule_redaction_witnesses(
                  id, capsule_id, scope, action, actor, reason, review_queue_id,
                  original_title_digest, redacted_title_digest,
                  original_body_digest, redacted_body_digest,
                  original_tags_digest, redacted_tags_digest,
                  created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("red"),
                    capsule_id,
                    row["scope"],
                    action,
                    actor,
                    reason,
                    review_queue_id,
                    _text_digest(expected_title),
                    _text_digest(title),
                    _text_digest(expected_body),
                    _text_digest(body),
                    _tags_digest(expected_tags),
                    _tags_digest(tags),
                    utc_now(),
                ),
            )
            conn.execute(
                """
                INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (action, capsule_id, row["scope"], reason, actor, utc_now()),
            )
            _invalidate_hot_scope(self.hot_dir, row["scope"])
            return True

    def update_capsule_source_events(
        self,
        capsule_id: str,
        source_event_ids: list[str],
        *,
        expected_source_event_ids: list[str] | None = None,
        expected_statuses: Iterable[str | MemoryStatus] | None = None,
        actor: str = "manual",
        reason: str = "",
        action: str = "update-source-events",
    ) -> bool:
        return self.update_capsule_source_events_batch(
            [
                {
                    "capsule_id": capsule_id,
                    "source_event_ids": source_event_ids,
                    "expected_source_event_ids": expected_source_event_ids,
                    "expected_statuses": expected_statuses,
                    "reason": reason,
                }
            ],
            actor=actor,
            action=action,
        )

    def update_capsule_source_events_batch(
        self,
        updates: list[dict[str, Any]],
        *,
        actor: str = "manual",
        action: str = "update-source-events",
    ) -> bool:
        self.init()
        if not updates:
            return True
        scopes: set[str] = set()
        try:
            with self.session() as conn:
                for update in updates:
                    scope = self._update_capsule_source_events_in_conn(
                        conn,
                        capsule_id=str(update["capsule_id"]),
                        source_event_ids=update["source_event_ids"],
                        expected_source_event_ids=update.get("expected_source_event_ids"),
                        expected_statuses=update.get("expected_statuses"),
                        witness_original_source_event_ids=update.get("witness_original_source_event_ids"),
                        witness_reason=str(update.get("witness_reason") or update.get("reason") or ""),
                        actor=actor,
                        reason=str(update.get("reason") or ""),
                        action=action,
                    )
                    if scope is None:
                        raise _SourceEventUpdateConflict()
                    scopes.add(scope)
        except _SourceEventUpdateConflict:
            return False
        for scope in scopes:
            _invalidate_hot_scope(self.hot_dir, scope)
        return True

    def _update_capsule_source_events_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        capsule_id: str,
        source_event_ids: list[str],
        expected_source_event_ids: list[str] | None,
        expected_statuses: Iterable[str | MemoryStatus] | None,
        witness_original_source_event_ids: list[str] | None,
        witness_reason: str,
        actor: str,
        reason: str,
        action: str,
    ) -> str | None:
        unique_ids = _unique_source_event_ids(source_event_ids)
        expected_ids = _unique_source_event_ids(expected_source_event_ids) if expected_source_event_ids is not None else None
        statuses = {_status_value(status) for status in expected_statuses or []}
        row = conn.execute(
            "SELECT scope, status, source_event_ids_json FROM capsules WHERE id = ?",
            (capsule_id,),
        ).fetchone()
        if row is None:
            return None
        if statuses and row["status"] not in statuses:
            return None
        if expected_ids is not None:
            current_ids = _unique_source_event_ids(json.loads(row["source_event_ids_json"]))
            if current_ids != expected_ids:
                return None
        if unique_ids:
            existing: set[str] = set()
            for chunk in _chunks(unique_ids, SQLITE_IN_CHUNK_SIZE):
                placeholders = ",".join("?" for _ in chunk)
                existing.update(
                    event_row["id"]
                    for event_row in conn.execute(
                        f"SELECT id FROM events WHERE id IN ({placeholders})",
                        chunk,
                    )
                )
            if existing != set(unique_ids):
                return None
        witness_payload: tuple[list[str], str, str | None] | None = None
        if witness_original_source_event_ids is not None:
            original_ids = _unique_source_event_ids(witness_original_source_event_ids)
            if not original_ids or not set(unique_ids).issubset(set(original_ids)):
                return None
            original_digest = _source_event_ids_digest(original_ids)
            if not original_digest:
                return None
            witness_payload = (original_ids, original_digest, _source_event_ids_digest(unique_ids))
        now = utc_now()
        clauses = ["id = ?"]
        args: list[Any] = [capsule_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            args.extend(sorted(statuses))
        if expected_ids is not None:
            clauses.append("source_event_ids_json = ?")
            args.append(row["source_event_ids_json"])
        cur = conn.execute(
            f"UPDATE capsules SET source_event_ids_json = ?, updated_at = ? WHERE {' AND '.join(clauses)}",
            (json.dumps(unique_ids, ensure_ascii=False), now, *args),
        )
        if cur.rowcount <= 0:
            return None
        conn.execute("DELETE FROM capsule_source_events WHERE capsule_id = ?", (capsule_id,))
        conn.executemany(
            "INSERT OR IGNORE INTO capsule_source_events(capsule_id, event_id) VALUES (?, ?)",
            [(capsule_id, event_id) for event_id in unique_ids],
        )
        if witness_payload is not None:
            original_ids, original_digest, retained_digest = witness_payload
            conn.execute(
                """
                INSERT INTO provenance_witnesses(
                  id, capsule_id, scope, action, actor, reason,
                  original_source_event_ids_json, retained_source_event_ids_json,
                  original_source_event_count, retained_source_event_count,
                  original_source_event_digest, retained_source_event_digest,
                  created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("wit"),
                    capsule_id,
                    row["scope"],
                    action,
                    actor,
                    witness_reason,
                    json.dumps(original_ids, ensure_ascii=False),
                    json.dumps(unique_ids, ensure_ascii=False),
                    len(original_ids),
                    len(unique_ids),
                    original_digest,
                    retained_digest,
                    utc_now(),
                ),
            )
        conn.execute(
            """
            INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (action, capsule_id, row["scope"], reason, actor, utc_now()),
        )
        return str(row["scope"])

    def provenance_witness_source_event_ids_for_capsules(self, capsule_ids: list[str]) -> list[str]:
        self.init()
        if not capsule_ids:
            return []
        unique_capsule_ids = list(dict.fromkeys(str(capsule_id) for capsule_id in capsule_ids if str(capsule_id)))
        source_ids: set[str] = set()
        with self.session() as conn:
            for chunk in _chunks(unique_capsule_ids, SQLITE_IN_CHUNK_SIZE):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT original_source_event_ids_json
                    FROM provenance_witnesses
                    WHERE capsule_id IN ({placeholders})
                    """,
                    chunk,
                )
                for row in rows:
                    try:
                        source_ids.update(
                            str(event_id)
                            for event_id in json.loads(row["original_source_event_ids_json"])
                            if str(event_id)
                        )
                    except (TypeError, json.JSONDecodeError):
                        continue
        return sorted(source_ids)

    def get_capsule(self, capsule_id: str) -> sqlite3.Row | None:
        self.init()
        with self.session() as conn:
            return conn.execute("SELECT * FROM capsules WHERE id = ?", (capsule_id,)).fetchone()

    def get_capsules_by_ids(
        self,
        capsule_ids: list[str],
        *,
        scope: str | None = None,
        include_global: bool = True,
        active_only: bool = True,
    ) -> list[sqlite3.Row]:
        self.init()
        unique_ids = list(dict.fromkeys(str(capsule_id) for capsule_id in capsule_ids if str(capsule_id)))
        if not unique_ids:
            return []
        rows: list[sqlite3.Row] = []
        scope_clause = ""
        if scope:
            scope_clause = "AND (scope = ? OR scope = 'global')" if include_global and scope != "global" else "AND scope = ?"
        status_clause = "AND status IN ('candidate', 'stable')" if active_only else ""
        with self.session() as conn:
            for chunk in _chunks(unique_ids, SQLITE_IN_CHUNK_SIZE):
                placeholders = ", ".join("?" for _ in chunk)
                params: list[Any] = [*chunk]
                if scope:
                    params.append(scope)
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT *
                        FROM capsules
                        WHERE id IN ({placeholders})
                          {status_clause}
                          {scope_clause}
                        ORDER BY salience DESC, updated_at DESC
                        """,
                        tuple(params),
                    )
                )
        return rows

    def get_events(self, event_ids: list[str]) -> list[sqlite3.Row]:
        self.init()
        if not event_ids:
            return []
        unique_ids = list(dict.fromkeys(event_ids))
        rows: list[sqlite3.Row] = []
        with self.session() as conn:
            for chunk in _chunks(unique_ids, SQLITE_IN_CHUNK_SIZE):
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    conn.execute(
                        f"SELECT * FROM events WHERE id IN ({placeholders})",
                        chunk,
                    )
                )
        return rows

    def source_event_ids_for_statuses(
        self,
        statuses: Iterable[str],
        *,
        scope: str | None = None,
    ) -> set[str]:
        self.init()
        status_values = list(statuses)
        if not status_values:
            return set()
        placeholders = ",".join("?" for _ in status_values)
        clauses = [f"c.status IN ({placeholders})"]
        args: list[Any] = [*status_values]
        if scope:
            clauses.append("c.scope = ?")
            args.append(scope)
        with self.session() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT l.event_id
                FROM capsule_source_events l
                JOIN capsules c ON c.id = l.capsule_id
                WHERE {' AND '.join(clauses)}
                """,
                args,
            )
            return {row["event_id"] for row in rows}

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

    def list_working_memory_impacts(
        self,
        *,
        scope: str,
        capsule_ids: list[str],
        include_global: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        self.init()
        if not capsule_ids or limit <= 0:
            return []
        unique_ids = list(dict.fromkeys(capsule_ids))
        rows: list[sqlite3.Row] = []
        scope_filter = "(scope = ? OR scope = 'global')" if include_global and scope != "global" else "scope = ?"
        with self.session() as conn:
            for chunk in _chunks(unique_ids, SQLITE_IN_CHUNK_SIZE):
                placeholders = ",".join("?" for _ in chunk)
                params: list[Any] = [scope, *chunk, limit]
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT *
                        FROM working_memory_impacts
                        WHERE {scope_filter}
                          AND capsule_id IN ({placeholders})
                        ORDER BY created_at DESC, id ASC
                        LIMIT ?
                        """,
                        params,
                    )
                )
        out: list[dict[str, Any]] = []
        for row in rows[:limit]:
            item = dict(row)
            item["cue_terms"] = json.loads(item.pop("cue_terms_json"))
            out.append(item)
        return out

    def list_recall_policy_impacts(
        self,
        *,
        scope: str,
        include_global: bool = True,
        intent: str | None = None,
        action_names: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        self.init()
        if limit <= 0:
            return []
        clauses = ["(scope = ? OR scope = 'global')" if include_global and scope != "global" else "scope = ?"]
        params: list[Any] = [scope]
        if intent:
            clauses.append("intent = ?")
            params.append(intent)
        unique_actions = list(dict.fromkeys(str(item) for item in action_names or [] if str(item).strip()))
        if unique_actions:
            placeholders = ",".join("?" for _ in unique_actions)
            clauses.append(f"action_name IN ({placeholders})")
            params.extend(unique_actions)
        params.append(limit)
        with self.session() as conn:
            rows = list(
                conn.execute(
                    f"""
                    SELECT *
                    FROM recall_policy_impacts
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC, id ASC
                    LIMIT ?
                    """,
                    params,
                )
            )
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["query_terms"] = json.loads(item.pop("query_terms_json"))
            out.append(item)
        return out

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


def _normalize_relation_text(text: str) -> str:
    normalized = re.sub(r"[_\s]+", " ", str(text or "").lower()).strip()
    normalized = re.sub(r"[^a-z0-9가-힣 .:-]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _relation_node_id(scope: str, normalized_label: str) -> str:
    digest = sha256(f"{scope}\0{normalized_label}".encode("utf-8")).hexdigest()[:16]
    return f"rel_node_{digest}"


def _relation_edge_id(
    *,
    subject_node_id: str,
    predicate_norm: str,
    object_node_id: str,
    scope: str,
    source_capsule_id: str | None,
) -> str:
    source = source_capsule_id or ""
    digest = sha256(
        f"{subject_node_id}\0{predicate_norm}\0{object_node_id}\0{scope}\0{source}".encode("utf-8")
    ).hexdigest()[:18]
    return f"rel_edge_{digest}"


def _ensure_relation_merge_v13_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(relation_merge_approvals)")
    }
    if "candidate_limit" not in columns:
        conn.execute(
            "ALTER TABLE relation_merge_approvals ADD COLUMN candidate_limit INTEGER NOT NULL DEFAULT 20"
        )
    if "include_global" not in columns:
        conn.execute(
            "ALTER TABLE relation_merge_approvals ADD COLUMN include_global INTEGER NOT NULL DEFAULT 0"
        )


def _ensure_reconsolidation_v16_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(reconsolidation_approvals)")
    }
    if "rollback_witness_json" not in columns:
        conn.execute(
            "ALTER TABLE reconsolidation_approvals ADD COLUMN rollback_witness_json TEXT NOT NULL DEFAULT '{}'"
        )


def _ensure_reconsolidation_rollback_v18_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reconsolidation_rollback_approvals (
          id TEXT PRIMARY KEY,
          token_hash TEXT NOT NULL UNIQUE,
          scope TEXT NOT NULL,
          backup_path TEXT NOT NULL,
          witness_id TEXT NOT NULL,
          approval_id TEXT NOT NULL,
          capsule_id TEXT NOT NULL,
          shadow_json TEXT NOT NULL,
          backup_identity_json TEXT NOT NULL,
          capsule_snapshot_json TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          used_at TEXT,
          FOREIGN KEY(witness_id) REFERENCES reconsolidation_witnesses(id),
          FOREIGN KEY(approval_id) REFERENCES reconsolidation_approvals(id),
          FOREIGN KEY(capsule_id) REFERENCES capsules(id)
        );

        CREATE INDEX IF NOT EXISTS idx_reconsolidation_rollback_approvals_status
        ON reconsolidation_rollback_approvals(status, expires_at);

        CREATE TABLE IF NOT EXISTS reconsolidation_rollback_witnesses (
          id TEXT PRIMARY KEY,
          rollback_approval_id TEXT NOT NULL,
          reconsolidation_witness_id TEXT NOT NULL,
          scope TEXT NOT NULL,
          capsule_id TEXT NOT NULL,
          action TEXT NOT NULL,
          reason TEXT NOT NULL,
          before_json TEXT NOT NULL,
          after_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(rollback_approval_id) REFERENCES reconsolidation_rollback_approvals(id),
          FOREIGN KEY(reconsolidation_witness_id) REFERENCES reconsolidation_witnesses(id),
          FOREIGN KEY(capsule_id) REFERENCES capsules(id)
        );

        CREATE INDEX IF NOT EXISTS idx_reconsolidation_rollback_witnesses_scope
        ON reconsolidation_rollback_witnesses(scope, created_at);
        """
    )


def _ensure_reconsolidation_exception_v19_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reconsolidation_exception_witnesses (
          id TEXT PRIMARY KEY,
          reconsolidation_witness_id TEXT NOT NULL,
          scope TEXT NOT NULL,
          capsule_id TEXT NOT NULL,
          field TEXT NOT NULL,
          before_digest TEXT NOT NULL,
          after_digest TEXT NOT NULL,
          reason TEXT NOT NULL,
          evidence_json TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(reconsolidation_witness_id) REFERENCES reconsolidation_witnesses(id),
          FOREIGN KEY(capsule_id) REFERENCES capsules(id)
        );

        CREATE INDEX IF NOT EXISTS idx_reconsolidation_exception_witnesses_target
        ON reconsolidation_exception_witnesses(reconsolidation_witness_id, capsule_id, field, status);
        """
    )


def _ensure_reconsolidation_action_v20_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS reconsolidation_action_approvals (
          id TEXT PRIMARY KEY,
          token_hash TEXT NOT NULL UNIQUE,
          scope TEXT NOT NULL,
          action TEXT NOT NULL,
          query TEXT NOT NULL,
          backup_path TEXT NOT NULL,
          preflight_json TEXT NOT NULL,
          action_gate_json TEXT NOT NULL,
          backup_identity_json TEXT NOT NULL,
          confirmation_required TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          used_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_reconsolidation_action_approvals_status
        ON reconsolidation_action_approvals(scope, action, status, expires_at);
        """
    )


def _upsert_relation_node(conn: sqlite3.Connection, *, label: str, scope: str, now: str) -> str | None:
    normalized = _normalize_relation_text(label)
    if not normalized:
        return None
    node_id = _relation_node_id(scope, normalized)
    conn.execute(
        """
        INSERT INTO relation_nodes(id, label, normalized_label, scope, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope, normalized_label) DO UPDATE SET
          label=CASE
            WHEN length(excluded.label) < length(relation_nodes.label) THEN excluded.label
            ELSE relation_nodes.label
          END,
          updated_at=excluded.updated_at
        """,
        (node_id, str(label or "").strip(), normalized, scope, now, now),
    )
    row = conn.execute(
        "SELECT id FROM relation_nodes WHERE scope = ? AND normalized_label = ?",
        (scope, normalized),
    ).fetchone()
    return str(row["id"]) if row else node_id


def _upsert_relation_edge(
    conn: sqlite3.Connection,
    *,
    subject: str,
    predicate: str,
    object_: str,
    scope: str,
    source_capsule_id: str | None,
    confidence: float,
) -> None:
    now = utc_now()
    subject_node_id = _upsert_relation_node(conn, label=subject, scope=scope, now=now)
    object_node_id = _upsert_relation_node(conn, label=object_, scope=scope, now=now)
    predicate_norm = _normalize_relation_text(predicate)
    if not subject_node_id or not object_node_id or not predicate_norm:
        return
    edge_id = _relation_edge_id(
        subject_node_id=subject_node_id,
        predicate_norm=predicate_norm,
        object_node_id=object_node_id,
        scope=scope,
        source_capsule_id=source_capsule_id,
    )
    conn.execute(
        """
        INSERT INTO relation_edges(
          id, subject_node_id, predicate, predicate_norm, object_node_id, scope,
          source_capsule_id, confidence, evidence_count, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          confidence=max(relation_edges.confidence, excluded.confidence),
          evidence_count=relation_edges.evidence_count + 1,
          updated_at=excluded.updated_at
        """,
        (
            edge_id,
            subject_node_id,
            str(predicate or "").strip(),
            predicate_norm,
            object_node_id,
            scope,
            source_capsule_id,
            max(0.0, min(1.0, float(confidence))),
            now,
            now,
        ),
    )


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _sync_working_memory_impacts(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT *
        FROM events
        WHERE source = 'working-memory-impact'
           OR metadata_json LIKE '%working_memory_impact%'
        ORDER BY created_at ASC
        """
    )
    for row in rows:
        _sync_working_memory_impact_event(conn, row_to_event(row))


def _sync_working_memory_impact_event(conn: sqlite3.Connection, event: Event) -> None:
    payload = event.metadata.get("working_memory_impact")
    if not isinstance(payload, dict):
        return
    rows = _working_memory_impact_rows(event, payload)
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO working_memory_impacts(
          event_id, scope, cue, cue_terms_json, capsule_id, outcome, helped, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _working_memory_impact_rows(event: Event, payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    cue = str(payload.get("cue") or "").strip()
    outcome = str(payload.get("outcome") or "").strip()
    capsule_ids = [
        str(capsule_id).strip()
        for capsule_id in payload.get("capsule_ids", [])
        if str(capsule_id).strip()
    ]
    if not capsule_ids:
        return []
    helped_raw = payload.get("helped")
    helped = 1 if helped_raw is True else 0 if helped_raw is False else None
    cue_terms = extract_keywords(cue or outcome, limit=24)
    cue_terms_json = json.dumps(cue_terms, ensure_ascii=False)
    rows = []
    for capsule_id in dict.fromkeys(capsule_ids):
        rows.append(
            (
                event.id,
                event.scope,
                cue,
                cue_terms_json,
                capsule_id,
                outcome,
                helped,
                event.created_at,
            )
        )
    return rows


def _sync_recall_policy_impacts(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT *
        FROM events
        WHERE source = 'recall-policy-impact'
           OR metadata_json LIKE '%recall_policy_impact%'
        ORDER BY created_at ASC
        """
    )
    for row in rows:
        _sync_recall_policy_impact_event(conn, row_to_event(row))


def _sync_recall_policy_impact_event(conn: sqlite3.Connection, event: Event) -> None:
    payload = event.metadata.get("recall_policy_impact")
    if not isinstance(payload, dict):
        return
    rows = _recall_policy_impact_rows(event, payload)
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO recall_policy_impacts(
          event_id, scope, query, query_terms_json, intent, strategy, action_name, outcome, helped, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _recall_policy_impact_rows(event: Event, payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    query = str(payload.get("query") or "").strip()
    outcome = str(payload.get("outcome") or "").strip()
    intent = str(payload.get("intent") or "").strip() or "unknown"
    strategy = str(payload.get("strategy") or "").strip() or "unknown"
    action_names = [
        str(action).strip()
        for action in payload.get("action_names", [])
        if str(action).strip()
    ]
    if not action_names:
        return []
    helped_raw = payload.get("helped")
    helped = 1 if helped_raw is True else 0 if helped_raw is False else None
    query_terms = extract_keywords(query or outcome, limit=24)
    query_terms_json = json.dumps(query_terms, ensure_ascii=False)
    rows = []
    for action_name in dict.fromkeys(action_names):
        rows.append(
            (
                event.id,
                event.scope,
                query,
                query_terms_json,
                intent,
                strategy,
                action_name,
                outcome,
                helped,
                event.created_at,
            )
        )
    return rows


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


def _current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'memory_meta'
        """
    ).fetchone()
    if row is None:
        return 0
    value = conn.execute("SELECT value FROM memory_meta WHERE key = 'schema_version'").fetchone()
    return int(value[0]) if value else 0


def _sync_relation_graph_from_temporal_edges(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM relation_edges")
    conn.execute("DELETE FROM relation_nodes")
    rows = conn.execute(
        """
        SELECT subject, predicate, object, scope, source_capsule_id, confidence
        FROM temporal_edges
        WHERE valid_to IS NULL
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()
    for row in rows:
        _upsert_relation_edge(
            conn,
            subject=row["subject"],
            predicate=row["predicate"],
            object_=row["object"],
            scope=row["scope"],
            source_capsule_id=row["source_capsule_id"],
            confidence=float(row["confidence"]),
        )


def _rebuild_capsule_source_events(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM capsule_source_events")
    rows = conn.execute("SELECT id, source_event_ids_json FROM capsules").fetchall()
    links: list[tuple[str, str]] = []
    for row in rows:
        try:
            source_event_ids = json.loads(row["source_event_ids_json"])
        except (TypeError, json.JSONDecodeError):
            source_event_ids = []
        links.extend((row["id"], event_id) for event_id in sorted(set(source_event_ids)))
    conn.executemany(
        "INSERT OR IGNORE INTO capsule_source_events(capsule_id, event_id) VALUES (?, ?)",
        links,
    )


def _sync_capsules_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM capsules_fts")
    rows = conn.execute("SELECT * FROM capsules").fetchall()
    for row in rows:
        _sync_capsule_fts_row(conn, row)


def _sync_capsule_fts_row(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    try:
        tags = json.loads(row["tags_json"])
    except (TypeError, json.JSONDecodeError):
        tags = []
    _sync_capsule_fts_payload(
        conn,
        capsule_id=row["id"],
        title=row["title"],
        body=row["body"],
        kind=row["kind"],
        scope=row["scope"],
        tags=tags,
        status=row["status"],
    )


def _sync_capsule_fts_payload(
    conn: sqlite3.Connection,
    *,
    capsule_id: str,
    title: str,
    body: str,
    kind: str,
    scope: str,
    tags: list[str],
    status: str,
) -> None:
    conn.execute("DELETE FROM capsules_fts WHERE id = ?", (capsule_id,))
    if status not in ACTIVE_FTS_STATUSES:
        return
    conn.execute(
        "INSERT INTO capsules_fts(id, title, body, kind, scope, tags) VALUES (?, ?, ?, ?, ?, ?)",
        (
            capsule_id,
            title,
            search_projection(title=title, body=body, kind=kind, tags=tags),
            kind,
            scope,
            " ".join(tags),
        ),
    )


def _text_digest(value: str) -> str:
    return sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _tags_digest(tags: Iterable[Any]) -> str:
    payload = json.dumps([str(tag) for tag in tags], ensure_ascii=False, sort_keys=True)
    return _text_digest(payload)


def _event_fingerprint(event: Event) -> str:
    payload = {
        "kind": event.kind.value,
        "text": event.text,
        "source": event.source,
        "scope": event.scope,
        "metadata": _fingerprint_metadata(event.metadata),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _source_event_ids_digest(source_event_ids: Iterable[Any]) -> str | None:
    normalized = sorted(str(event_id) for event_id in source_event_ids if str(event_id))
    if not normalized:
        return None
    return sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _fingerprint_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = dict(metadata)
    payload.pop("turn_captured_at", None)
    return payload


def _fts_query(query: str) -> str:
    tokens = []
    for token in query.replace('"', " ").replace("'", " ").split():
        cleaned = "".join(ch for ch in token if ch.isalnum() or ch == "_")
        if is_search_term(cleaned):
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


def _storage_category(root: Path, path: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return "support_bytes"
    if not parts:
        return "support_bytes"
    first = parts[0]
    if first in {"memory.db", "memory.db-wal", "memory.db-shm"}:
        return "db_bytes"
    if first == "ledger":
        return "ledger_bytes"
    if first == "hot":
        return "hot_bytes"
    if first == "spool":
        return "spool_bytes"
    if first == "backups":
        return "backup_bytes"
    if first == "archive":
        second = parts[1] if len(parts) > 1 else ""
        if second == "objects":
            return "archive_object_bytes"
        if second == "cold":
            return "cold_export_bytes"
        if second == "retention-cycles":
            return "retention_cycle_bytes"
        return "archive_other_bytes"
    return "support_bytes"

