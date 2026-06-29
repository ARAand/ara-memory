from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara_memory.models import new_id, utc_now
from ara_memory.turn import remember_turn


@dataclass(slots=True)
class SpoolRecord:
    spool_id: str
    path: str
    state: str

    def as_dict(self) -> dict[str, Any]:
        return {"spool_id": self.spool_id, "path": self.path, "state": self.state}


@dataclass(slots=True)
class DrainItem:
    spool_id: str
    state: str
    path: str
    result: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"spool_id": self.spool_id, "state": self.state, "path": self.path}
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(slots=True)
class DrainReport:
    processed: int
    succeeded: int
    failed: int
    recovered: int = 0
    items: list[DrainItem] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "recovered": self.recovered,
            "items": [item.as_dict() for item in self.items],
        }


def enqueue_turn(
    memory: Any,
    turn: dict[str, Any],
    *,
    scope: str = "global",
    source: str = "codex-turn",
    consolidate: bool = True,
    sleep: bool = False,
    hot_budget: int = 1200,
    capture_cwd: Path | None = None,
    include_untracked_content: bool = False,
    max_file_chars: int = 8000,
    max_text_chars: int = 12000,
) -> SpoolRecord:
    memory.init()
    paths = _spool_paths(memory.store.root)
    turn_payload = dict(turn)
    turn_id = str(turn_payload.get("turn_id") or new_id("turn"))
    turn_payload["turn_id"] = turn_id
    spool_id = _new_spool_id(turn_id)
    created_at = utc_now()
    metadata = dict(turn_payload.get("metadata")) if isinstance(turn_payload.get("metadata"), dict) else {}
    metadata.setdefault("turn_captured_at", created_at)
    metadata.setdefault("spool_id", spool_id)
    metadata.setdefault("spooled_at", created_at)
    turn_payload["metadata"] = metadata
    payload = {
        "format": "ara-memory-spooled-turn-v1",
        "spool_id": spool_id,
        "logical_turn_id": turn_id,
        "created_at": created_at,
        "turn": turn_payload,
        "options": {
            "scope": scope,
            "source": source,
            "consolidate": consolidate,
            "sleep": sleep,
            "hot_budget": hot_budget,
            "capture_cwd": str(capture_cwd.resolve()) if capture_cwd else None,
            "include_untracked_content": include_untracked_content,
            "max_file_chars": max_file_chars,
            "max_text_chars": max_text_chars,
        },
    }
    final_path = paths["pending"] / f"{_safe_name(spool_id)}.json"
    temp_path = paths["pending"] / f".{_safe_name(spool_id)}.{new_id('tmp')}.tmp"
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        _publish_without_overwrite(temp_path, final_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return SpoolRecord(spool_id=spool_id, path=str(final_path), state="pending")


def drain_spool(
    memory: Any,
    *,
    limit: int = 25,
    stop_on_error: bool = False,
    processing_stale_seconds: int = 3600,
) -> DrainReport:
    memory.init()
    paths = _spool_paths(memory.store.root)
    recovered = _recover_stale_processing(paths, stale_seconds=processing_stale_seconds)
    items: list[DrainItem] = []
    for pending_path in sorted(paths["pending"].glob("*.json"))[:limit]:
        processing_path = paths["processing"] / pending_path.name
        try:
            pending_path.replace(processing_path)
        except FileNotFoundError:
            continue
        spool_id = processing_path.stem
        try:
            payload = json.loads(processing_path.read_text(encoding="utf-8"))
            if payload.get("format") != "ara-memory-spooled-turn-v1":
                raise ValueError("Unsupported spool record format.")
            options = _options(payload.get("options"))
            result = remember_turn(
                memory,
                _turn(payload.get("turn")),
                scope=options["scope"],
                source=options["source"],
                consolidate=options["consolidate"],
                sleep=options["sleep"],
                hot_budget=options["hot_budget"],
                capture_cwd=Path(options["capture_cwd"]) if options["capture_cwd"] else None,
                include_untracked_content=options["include_untracked_content"],
                max_file_chars=options["max_file_chars"],
                max_text_chars=options["max_text_chars"],
            )
            done_path = _archive_spool_record(paths["done"], processing_path, payload, result=result)
            items.append(DrainItem(spool_id=spool_id, state="done", path=str(done_path), result=result))
        except Exception as exc:  # Keep the original envelope for inspection and retry decisions.
            failed_path = _archive_spool_record(paths["failed"], processing_path, _safe_payload(processing_path), error=exc)
            items.append(DrainItem(spool_id=spool_id, state="failed", path=str(failed_path), error=str(exc)))
            if stop_on_error:
                break
    return DrainReport(
        processed=len(items),
        succeeded=sum(1 for item in items if item.state == "done"),
        failed=sum(1 for item in items if item.state == "failed"),
        recovered=recovered,
        items=items,
    )


def spool_stats(memory: Any) -> dict[str, int]:
    memory.init()
    paths = _spool_paths(memory.store.root)
    return {name: len(list(path.glob("*.json"))) for name, path in paths.items()}


def _spool_paths(root: Path) -> dict[str, Path]:
    base = root / "spool"
    paths = {
        "pending": base / "pending",
        "processing": base / "processing",
        "done": base / "done",
        "failed": base / "failed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _recover_stale_processing(paths: dict[str, Path], *, stale_seconds: int) -> int:
    if stale_seconds <= 0:
        return 0
    now = time.time()
    recovered = 0
    for processing_path in sorted(paths["processing"].glob("*.json")):
        try:
            age = now - processing_path.stat().st_mtime
        except FileNotFoundError:
            continue
        if age < stale_seconds:
            continue
        pending_path = paths["pending"] / processing_path.name
        if pending_path.exists():
            pending_path = paths["pending"] / f"{processing_path.stem}.recovered-{new_id('spool')}.json"
        try:
            processing_path.replace(pending_path)
            recovered += 1
        except FileNotFoundError:
            continue
    return recovered


def _archive_spool_record(
    target_dir: Path,
    processing_path: Path,
    payload: dict[str, Any],
    *,
    result: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> Path:
    payload = dict(payload)
    payload["drained_at"] = utc_now()
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    target_path = target_dir / processing_path.name
    temp_path = target_dir / f".{processing_path.stem}.{new_id('tmp')}.tmp"
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(target_path)
    try:
        processing_path.unlink()
    except FileNotFoundError:
        pass
    return target_path


def _safe_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"raw": payload}
    except Exception:
        failed_payload = {"format": "ara-memory-spooled-turn-v1", "spool_id": path.stem}
        raw_copy = path.with_suffix(".raw")
        shutil.copy2(path, raw_copy)
        failed_payload["raw_copy"] = str(raw_copy)
        return failed_payload


def _options(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    return {
        "scope": str(raw.get("scope", "global")),
        "source": str(raw.get("source", "codex-turn")),
        "consolidate": bool(raw.get("consolidate", True)),
        "sleep": bool(raw.get("sleep", False)),
        "hot_budget": int(raw.get("hot_budget", 1200)),
        "capture_cwd": raw.get("capture_cwd"),
        "include_untracked_content": bool(raw.get("include_untracked_content", False)),
        "max_file_chars": int(raw.get("max_file_chars", 8000)),
        "max_text_chars": int(raw.get("max_text_chars", 12000)),
    }


def _turn(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Spooled turn must contain a JSON object in 'turn'.")
    return value


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:120]


def _new_spool_id(turn_id: str) -> str:
    base = _safe_name(turn_id)[:80].strip("._-") or "turn"
    return f"{base}-{new_id('spool')}"


def _publish_without_overwrite(temp_path: Path, final_path: Path) -> None:
    try:
        os.link(temp_path, final_path)
    except FileExistsError:
        raise
    except OSError:
        data = temp_path.read_bytes()
        with final_path.open("xb") as handle:
            handle.write(data)
