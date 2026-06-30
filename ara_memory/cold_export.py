from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ara_memory.models import MemoryStatus, utc_now
from ara_memory.storage import MemoryStore, row_to_capsule


COLD_EXPORT_FORMAT = "ara-memory-cold-export-v1"
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

    capsules = _select_capsules(store, scope=scope, statuses=status_values, limit=limit, order=order)
    source_event_ids = sorted({event_id for capsule in capsules for event_id in capsule["source_event_ids"]})
    events = _select_events(store, source_event_ids) if include_events else []
    output_path = output or _default_cold_export_path(store.root, scope=scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": utc_now(),
        "format": COLD_EXPORT_FORMAT,
        "schema_version": store.schema_version(),
        "scope": scope,
        "statuses": [status.value for status in status_values],
        "limit": limit,
        "order": order,
        "include_events": include_events,
        "capsule_count": len(capsules),
        "event_count": len(events),
        "source_event_id_count": len(source_event_ids),
        "source_stats": store.stats(),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        zf.writestr("capsules.jsonl", _jsonl(capsules))
        zf.writestr("events.jsonl", _jsonl(events))

    return ColdExportResult(path=output_path, manifest=manifest)


def verify_cold_export(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        required = {"manifest.json", "capsules.jsonl", "events.jsonl"}
        missing = sorted(required - names)
        manifest = json.loads(zf.read("manifest.json").decode("utf-8")) if "manifest.json" in names else {}
        capsules = _read_jsonl(zf, "capsules.jsonl") if "capsules.jsonl" in names else []
        events = _read_jsonl(zf, "events.jsonl") if "events.jsonl" in names else []
        format_ok = manifest.get("format") == COLD_EXPORT_FORMAT
        capsule_count_ok = manifest.get("capsule_count") == len(capsules)
        event_count_ok = manifest.get("event_count") == len(events)
        statuses = set(manifest.get("statuses", []))
        status_ok = all(capsule.get("status") in statuses for capsule in capsules)
        return {
            "path": str(path),
            "passed": not missing and format_ok and capsule_count_ok and event_count_ok and status_ok,
            "missing": missing,
            "format_ok": format_ok,
            "capsule_count_ok": capsule_count_ok,
            "event_count_ok": event_count_ok,
            "status_ok": status_ok,
            "manifest": manifest,
            "entries": len(names),
        }


def _select_capsules(
    store: MemoryStore,
    *,
    scope: str | None,
    statuses: tuple[MemoryStatus, ...],
    limit: int | None,
    order: str,
) -> list[dict[str, Any]]:
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
        return [row_to_capsule(row) for row in conn.execute(sql, args)]


def _select_events(store: MemoryStore, event_ids: list[str]) -> list[dict[str, Any]]:
    rows = store.get_events(event_ids)
    events = []
    for row in rows:
        event = dict(row)
        event["metadata"] = json.loads(event.pop("metadata_json"))
        events.append(event)
    return sorted(events, key=lambda item: (item["created_at"], item["id"]))


def _jsonl(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _read_jsonl(zf: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    raw = zf.read(name).decode("utf-8")
    rows = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _default_cold_export_path(root: Path, *, scope: str | None) -> Path:
    stamp = utc_now().replace(":", "").replace("+", "Z")
    scope_part = _safe_name(scope or "all")
    return root / "archive" / "cold" / f"{scope_part}-cold-{stamp}.zip"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value).strip("-") or "all"
