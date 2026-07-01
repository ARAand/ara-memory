from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ara_memory.archive_safety import (
    MAX_JSONL_BYTES,
    MAX_JSONL_LINE_BYTES,
    MAX_MANIFEST_BYTES,
    inspect_zip,
    iter_member_lines,
    read_member_text,
)
from ara_memory.backup import (
    ENTRY_HASHES_FIELD,
    MANIFEST_SIGNATURE_FIELD,
    SIGNATURE_ALGORITHM,
    _backup_key_id,
    _load_or_create_backup_signing_key,
    _manifest_json,
    _manifest_signature,
    _skipped_hash_report,
    _verify_entry_hashes,
    _verify_manifest_signature,
)
from ara_memory.models import MemoryStatus, utc_now
from ara_memory.storage import MemoryStore, row_to_capsule


COLD_EXPORT_FORMAT = "ara-memory-cold-export-v2"
DEFAULT_COLD_STATUSES = (
    MemoryStatus.SUPERSEDED,
    MemoryStatus.REJECTED,
    MemoryStatus.QUARANTINED,
)


@dataclass(slots=True)
class ColdExportResult:
    path: Path
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "manifest": self.manifest,
        }


def export_cold_capsules(
    store: MemoryStore,
    *,
    output: Path | None = None,
    scope: str | None = None,
    statuses: Iterable[MemoryStatus] | None = None,
    limit: int | None = None,
    include_events: bool = True,
    order: str = "newest",
) -> ColdExportResult:
    store.init()
    status_values = tuple(statuses or DEFAULT_COLD_STATUSES)
    if not status_values:
        raise ValueError("At least one status is required for cold export.")
    if order not in {"newest", "oldest"}:
        raise ValueError("order must be 'newest' or 'oldest'.")

    output_path = output or _default_cold_export_path(store.root, scope=scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signing_key = _load_or_create_backup_signing_key(store.root)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        capsule_path = tmp_root / "capsules.jsonl"
        event_path = tmp_root / "events.jsonl"
        capsule_count, source_event_ids, capsule_hash = _write_capsules_jsonl(
            store,
            capsule_path,
            scope=scope,
            statuses=status_values,
            limit=limit,
            order=order,
        )
        source_event_id_list = sorted(source_event_ids)
        event_rows = _iter_events(store, source_event_id_list) if include_events else []
        event_count, event_hash = _write_jsonl_rows(event_path, event_rows)

        manifest = {
            "created_at": utc_now(),
            "format": COLD_EXPORT_FORMAT,
            "schema_version": store.schema_version(),
            "signature_algorithm": SIGNATURE_ALGORITHM,
            "backup_key_id": _backup_key_id(signing_key),
            "scope": scope,
            "statuses": [status.value for status in status_values],
            "limit": limit,
            "order": order,
            "include_events": include_events,
            "capsule_count": capsule_count,
            "event_count": event_count,
            "source_event_id_count": len(source_event_id_list),
            "source_stats": store.stats(),
        }
        manifest[ENTRY_HASHES_FIELD] = {
            "capsules.jsonl": capsule_hash,
            "events.jsonl": event_hash,
        }
        manifest["entry_count"] = len(manifest[ENTRY_HASHES_FIELD])
        manifest[MANIFEST_SIGNATURE_FIELD] = _manifest_signature(manifest, signing_key)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(capsule_path, "capsules.jsonl")
            zf.write(event_path, "events.jsonl")
            zf.writestr("manifest.json", _manifest_json(manifest))

    return ColdExportResult(path=output_path, manifest=manifest)


def verify_cold_export(path: Path, *, trust_root: Path | None = None) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        safety = inspect_zip(zf)
        names = set(safety["names"])
        required = {"manifest.json", "capsules.jsonl", "events.jsonl"}
        missing = sorted(required - names)
        manifest_error = None
        if safety["passed"] and "manifest.json" in names:
            try:
                manifest = json.loads(read_member_text(zf, "manifest.json", max_bytes=MAX_MANIFEST_BYTES))
            except Exception as exc:
                manifest = {}
                manifest_error = f"{type(exc).__name__}: {exc}"
        else:
            manifest = {}
        statuses = set(manifest.get("statuses", []))
        try:
            capsule_scan = (
                _scan_capsules_jsonl(zf, "capsules.jsonl", allowed_statuses=statuses)
                if safety["passed"] and "capsules.jsonl" in names
                else _empty_capsule_scan()
            )
            event_scan = (
                _scan_events_jsonl(zf, "events.jsonl")
                if safety["passed"] and "events.jsonl" in names
                else _empty_event_scan()
            )
            jsonl_error = None
        except Exception as exc:
            capsule_scan = _empty_capsule_scan()
            event_scan = _empty_event_scan()
            jsonl_error = f"{type(exc).__name__}: {exc}"
        hash_report = _verify_entry_hashes(zf, names, manifest) if safety["passed"] else _skipped_hash_report()
        signature_report = _verify_manifest_signature(
            manifest,
            trust_root=trust_root.resolve() if trust_root else None,
        )
        format_ok = manifest.get("format") == COLD_EXPORT_FORMAT
        signature_algorithm_ok = manifest.get("signature_algorithm") == SIGNATURE_ALGORITHM
        capsule_count_ok = manifest.get("capsule_count") == capsule_scan["count"]
        event_count_ok = manifest.get("event_count") == event_scan["count"]
        source_event_ids = sorted(capsule_scan["source_event_ids"])
        exported_event_ids = event_scan["event_ids"]
        source_event_id_count_ok = manifest.get("source_event_id_count") == len(source_event_ids)
        missing_source_event_ids = sorted(set(source_event_ids) - exported_event_ids)
        provenance_ok = not missing_source_event_ids
        status_ok = capsule_scan["status_ok"]
        return {
            "path": str(path),
            "passed": (
                safety["passed"]
                and not missing
                and manifest_error is None
                and jsonl_error is None
                and format_ok
                and signature_algorithm_ok
                and capsule_count_ok
                and event_count_ok
                and source_event_id_count_ok
                and provenance_ok
                and status_ok
                and hash_report["status"] == "ok"
                and signature_report["status"] == "ok"
            ),
            "missing": missing,
            "archive_safety": safety,
            "manifest_error": manifest_error,
            "jsonl_error": jsonl_error,
            "format_ok": format_ok,
            "signature_algorithm_ok": signature_algorithm_ok,
            "capsule_count_ok": capsule_count_ok,
            "event_count_ok": event_count_ok,
            "source_event_id_count_ok": source_event_id_count_ok,
            "provenance_ok": provenance_ok,
            "missing_source_event_ids": missing_source_event_ids[:20],
            "status_ok": status_ok,
            "capsule_ids": sorted(capsule_scan["capsule_ids"]),
            "entry_hashes": hash_report,
            "manifest_signature": signature_report,
            "manifest": manifest,
            "entries": len(names),
        }


def _iter_capsules(
    store: MemoryStore,
    *,
    scope: str | None,
    statuses: tuple[MemoryStatus, ...],
    limit: int | None,
    order: str,
) -> Iterator[dict[str, Any]]:
    placeholders = ",".join("?" for _ in statuses)
    clauses = [f"status IN ({placeholders})"]
    args: list[Any] = [status.value for status in statuses]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    ordering = "updated_at ASC, id ASC" if order == "oldest" else "updated_at DESC, id ASC"
    sql = f"""
        SELECT *
        FROM capsules
        WHERE {' AND '.join(clauses)}
        ORDER BY {ordering}
    """
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    with store.session() as conn:
        for row in conn.execute(sql, args):
            yield row_to_capsule(row)


def _iter_events(store: MemoryStore, event_ids: list[str]) -> Iterator[dict[str, Any]]:
    for row in sorted(store.get_events(event_ids), key=lambda item: (item["created_at"], item["id"])):
        event = dict(row)
        event["metadata"] = json.loads(event.pop("metadata_json"))
        yield event


def _write_capsules_jsonl(
    store: MemoryStore,
    path: Path,
    *,
    scope: str | None,
    statuses: tuple[MemoryStatus, ...],
    limit: int | None,
    order: str,
) -> tuple[int, set[str], str]:
    source_event_ids: set[str] = set()

    def rows() -> Iterator[dict[str, Any]]:
        for capsule in _iter_capsules(store, scope=scope, statuses=statuses, limit=limit, order=order):
            source_event_ids.update(str(event_id) for event_id in capsule["source_event_ids"] if event_id)
            yield capsule

    count, digest = _write_jsonl_rows(path, rows())
    return count, source_event_ids, digest


def _write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    with path.open("wb") as handle:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            if len(line) > MAX_JSONL_LINE_BYTES:
                raise ValueError(f"JSONL row exceeds export line limit: {path.name}")
            total += len(line)
            if total > MAX_JSONL_BYTES:
                raise ValueError(f"JSONL export exceeds byte limit: {path.name}")
            handle.write(line)
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def _scan_capsules_jsonl(
    zf: zipfile.ZipFile,
    name: str,
    *,
    allowed_statuses: set[str],
) -> dict[str, Any]:
    count = 0
    capsule_ids: set[str] = set()
    source_event_ids: set[str] = set()
    status_ok = True
    for row in _iter_jsonl(zf, name):
        count += 1
        capsule_id = row.get("id")
        if capsule_id:
            capsule_ids.add(str(capsule_id))
        if row.get("status") not in allowed_statuses:
            status_ok = False
        source_event_ids.update(str(event_id) for event_id in row.get("source_event_ids", []) if event_id)
    return {"count": count, "capsule_ids": capsule_ids, "source_event_ids": source_event_ids, "status_ok": status_ok}


def _scan_events_jsonl(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    count = 0
    event_ids: set[str] = set()
    for row in _iter_jsonl(zf, name):
        count += 1
        event_id = row.get("id")
        if event_id:
            event_ids.add(str(event_id))
    return {"count": count, "event_ids": event_ids}


def _iter_jsonl(zf: zipfile.ZipFile, name: str) -> Iterator[dict[str, Any]]:
    for line in iter_member_lines(zf, name, max_bytes=MAX_JSONL_BYTES):
        if line.strip():
            yield json.loads(line)


def _empty_capsule_scan() -> dict[str, Any]:
    return {"count": 0, "capsule_ids": set(), "source_event_ids": set(), "status_ok": True}


def _empty_event_scan() -> dict[str, Any]:
    return {"count": 0, "event_ids": set()}


def _default_cold_export_path(root: Path, *, scope: str | None) -> Path:
    stamp = utc_now().replace(":", "").replace("+", "Z")
    scope_part = _safe_name(scope or "all")
    return root / "archive" / "cold" / f"{scope_part}-cold-{stamp}.zip"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-") or "all"
