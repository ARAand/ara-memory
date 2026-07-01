from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from hashlib import sha256
import hmac
from pathlib import Path
from typing import Any

from ara_memory.models import new_id, utc_now
from ara_memory.privacy import guard_turn_payload, privacy_safe_label
from ara_memory.turn import remember_turn
from ara_memory.worktree import snapshot_worktree


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
    stabilization: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.failed == 0 and (self.stabilization is None or bool(self.stabilization.get("passed", True)))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "recovered": self.recovered,
            "items": [item.as_dict() for item in self.items],
        }
        if self.stabilization is not None:
            payload["stabilization"] = self.stabilization
        return payload


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
    allow_raw_private: bool = False,
) -> SpoolRecord:
    memory.init()
    paths = _spool_paths(memory.store.root)
    turn_payload = dict(turn)
    raw_turn_id = str(turn_payload.get("turn_id") or new_id("turn"))
    turn_id = privacy_safe_label(raw_turn_id, label="turn", allow_raw_private=allow_raw_private)
    turn_payload["turn_id"] = turn_id
    spool_id = _new_spool_id(turn_id)
    created_at = utc_now()
    safe_scope = privacy_safe_label(scope, label="scope", allow_raw_private=allow_raw_private)
    safe_source = privacy_safe_label(source, label="source", allow_raw_private=allow_raw_private)
    metadata = dict(turn_payload.get("metadata")) if isinstance(turn_payload.get("metadata"), dict) else {}
    metadata.setdefault("turn_captured_at", created_at)
    metadata.setdefault("spool_id", spool_id)
    metadata.setdefault("spooled_at", created_at)
    turn_payload["metadata"] = metadata
    snapshots: dict[str, Any] = {}
    artifact_snapshots = _snapshot_turn_artifacts(
        turn_payload,
        root=memory.store.root,
        spool_id=spool_id,
    )
    if artifact_snapshots:
        snapshots["artifacts"] = artifact_snapshots
    if capture_cwd:
        worktree_snapshot = snapshot_worktree(
            cwd=capture_cwd.resolve(),
            include_untracked_content=include_untracked_content,
            max_file_chars=max_file_chars,
        )
        if worktree_snapshot:
            turn_payload["worktree_snapshot"] = worktree_snapshot
        snapshots["worktree"] = {
            "cwd": str(capture_cwd.resolve()),
            "events": len(worktree_snapshot),
            "captured_at": created_at,
        }
    raw_turn_payload = turn_payload
    turn_guard = guard_turn_payload(turn_payload, allow_raw_private=allow_raw_private)
    turn_payload = turn_guard.value
    _restore_snapshot_artifact_paths(turn_payload, raw_turn_payload, root=memory.store.root)
    if turn_guard.reasons or turn_guard.redacted or turn_guard.allow_raw_private:
        metadata = dict(turn_payload.get("metadata")) if isinstance(turn_payload.get("metadata"), dict) else {}
        metadata["spool_privacy"] = turn_guard.metadata_note()
        turn_payload["metadata"] = metadata
    snapshot_guard = guard_turn_payload({"snapshots": snapshots}, allow_raw_private=allow_raw_private)
    snapshots = snapshot_guard.value.get("snapshots", snapshots)
    payload = {
        "format": "ara-memory-spooled-turn-v1",
        "spool_id": spool_id,
        "logical_turn_id": turn_id,
        "created_at": created_at,
        "turn": turn_payload,
        "snapshots": snapshots,
        "options": {
            "scope": safe_scope,
            "source": safe_source,
            "consolidate": consolidate,
            "sleep": sleep,
            "hot_budget": hot_budget,
            "capture_cwd": str(capture_cwd.resolve()) if capture_cwd else None,
            "include_untracked_content": include_untracked_content,
            "max_file_chars": max_file_chars,
            "max_text_chars": max_text_chars,
            "allow_raw_private": allow_raw_private,
        },
    }
    payload["seal"] = _seal_payload(payload, root=memory.store.root)
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
    scope: str | None = None,
    limit: int = 25,
    stop_on_error: bool = False,
    processing_stale_seconds: int = 3600,
    stabilize: bool = False,
    stabilization_scope: str | None = None,
    stabilization_episode_min_group_size: int = 5,
    stabilization_session_min_group_size: int = 20,
    stabilization_candidate_min_group_size: int = 3,
    stabilization_limit: int = 80,
) -> DrainReport:
    memory.init()
    paths = _spool_paths(memory.store.root)
    scope_filter = _scope_filter(scope)
    recovered = _recover_stale_processing(paths, stale_seconds=processing_stale_seconds, scope_filter=scope_filter)
    items: list[DrainItem] = []
    succeeded_scopes: set[str] = set()
    scope_hot_budgets: dict[str, int] = {}
    for pending_path in sorted(paths["pending"].glob("*.json")):
        if len(items) >= limit:
            break
        if scope_filter is not None and _spool_record_scope(pending_path) not in {None, *scope_filter}:
            continue
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
            validate_spool_envelope(payload, root=memory.store.root)
            options = _options(payload.get("options"))
            turn_payload = _turn(payload.get("turn"))
            capture_cwd = (
                None
                if "worktree_snapshot" in turn_payload
                else Path(options["capture_cwd"])
                if options["capture_cwd"]
                else None
            )
            result = remember_turn(
                memory,
                turn_payload,
                scope=options["scope"],
                source=options["source"],
                consolidate=options["consolidate"],
                sleep=options["sleep"],
                hot_budget=options["hot_budget"],
                capture_cwd=capture_cwd,
                include_untracked_content=options["include_untracked_content"],
                max_file_chars=options["max_file_chars"],
                max_text_chars=options["max_text_chars"],
                allow_raw_private=options.get("allow_raw_private", False),
            )
            done_path = _archive_spool_record(paths["done"], processing_path, payload, result=result)
            items.append(DrainItem(spool_id=spool_id, state="done", path=str(done_path), result=result))
            succeeded_scopes.add(options["scope"])
            scope_hot_budgets[options["scope"]] = max(
                int(options["hot_budget"]),
                scope_hot_budgets.get(options["scope"], 0),
            )
        except Exception as exc:  # Keep the original envelope for inspection and retry decisions.
            failed_path = _archive_spool_record(
                paths["failed"],
                processing_path,
                _safe_payload(processing_path, failed_dir=paths["failed"]),
                error=exc,
            )
            items.append(DrainItem(spool_id=spool_id, state="failed", path=str(failed_path), error=str(exc)))
            if stop_on_error:
                break
    if stabilization_scope:
        stabilization_scope = _canonical_scope_label(stabilization_scope)
        succeeded_scopes.add(stabilization_scope)
        scope_hot_budgets.setdefault(stabilization_scope, 1200)
    stabilization = (
        _stabilize_after_drain(
            memory,
            scopes=sorted(succeeded_scopes),
            hot_budgets=scope_hot_budgets,
            episode_min_group_size=stabilization_episode_min_group_size,
            session_min_group_size=stabilization_session_min_group_size,
            candidate_min_group_size=stabilization_candidate_min_group_size,
            limit=stabilization_limit,
        )
        if stabilize and succeeded_scopes
        else None
    )
    return DrainReport(
        processed=len(items),
        succeeded=sum(1 for item in items if item.state == "done"),
        failed=sum(1 for item in items if item.state == "failed"),
        recovered=recovered,
        items=items,
        stabilization=stabilization,
    )


def spool_stats(memory: Any) -> dict[str, int]:
    memory.init()
    paths = _spool_paths(memory.store.root)
    return {name: len(list(path.glob("*.json"))) for name, path in paths.items()}


def validate_spool_envelope(payload: dict[str, Any], *, root: Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Spooled turn must be a JSON object.")
    if payload.get("format") != "ara-memory-spooled-turn-v1":
        raise ValueError("Unsupported spool record format.")
    _verify_spool_seal(payload, root=root)
    options = payload.get("options")
    if not isinstance(options, dict):
        raise ValueError("Spooled turn must include options.")
    for key in ("consolidate", "sleep", "include_untracked_content"):
        if not isinstance(options.get(key), bool):
            raise ValueError(f"Spooled option {key} must be a JSON boolean.")
    if "allow_raw_private" in options and not isinstance(options.get("allow_raw_private"), bool):
        raise ValueError("Spooled option allow_raw_private must be a JSON boolean.")
    _bounded_int(options.get("hot_budget"), "hot_budget", minimum=0, maximum=5000)
    _bounded_int(options.get("max_file_chars"), "max_file_chars", minimum=0, maximum=100000)
    _bounded_int(options.get("max_text_chars"), "max_text_chars", minimum=100, maximum=200000)
    if not isinstance(options.get("scope", ""), str) or len(str(options.get("scope", ""))) > 160:
        raise ValueError("Spooled scope must be a short string.")
    if not isinstance(options.get("source", ""), str) or len(str(options.get("source", ""))) > 160:
        raise ValueError("Spooled source must be a short string.")
    turn_payload = payload.get("turn")
    if not isinstance(turn_payload, dict):
        raise ValueError("Spooled turn must contain a JSON object in 'turn'.")
    total_text = _turn_text_size(turn_payload)
    max_text = int(options.get("max_text_chars", 12000))
    if total_text > max(200000, max_text * 20):
        raise ValueError("Spooled turn text exceeds deterministic capture bounds.")
    capture_cwd = options.get("capture_cwd")
    if capture_cwd and "worktree_snapshot" not in turn_payload:
        raise ValueError("Spooled capture_cwd must be represented by an enqueue-time worktree snapshot.")
    _validate_artifact_snapshot_metadata(turn_payload, root=root)


def _seal_payload(payload: dict[str, Any], *, root: Path) -> dict[str, str]:
    key = _spool_key(root, create=True)
    digest = hmac.new(key, _canonical_spool_payload(payload), sha256).hexdigest()
    return {"algorithm": "hmac-sha256", "value": digest}


def _verify_spool_seal(payload: dict[str, Any], *, root: Path) -> None:
    seal = payload.get("seal")
    if not isinstance(seal, dict):
        raise ValueError("Spooled turn is missing a local seal.")
    if seal.get("algorithm") != "hmac-sha256":
        raise ValueError("Unsupported spool seal algorithm.")
    actual = seal.get("value")
    if not isinstance(actual, str) or not actual:
        raise ValueError("Spooled turn seal is empty.")
    key = _spool_key(root, create=False)
    expected = hmac.new(key, _canonical_spool_payload(payload), sha256).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError("Spooled turn seal verification failed.")


def _spool_key(root: Path, *, create: bool) -> bytes:
    path = root / "spool" / ".seal-key"
    if not path.exists():
        if not create:
            raise ValueError("Local spool seal key is missing.")
        path.parent.mkdir(parents=True, exist_ok=True)
        _create_spool_key_file(path)
    return _read_spool_key(path)


def _create_spool_key_file(path: Path) -> None:
    key_hex = os.urandom(32).hex()
    temp_path = path.with_name(f".seal-key.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(key_hex)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            pass
        except OSError:
            if path.exists():
                return
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                return
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(key_hex)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _read_spool_key(path: Path) -> bytes:
    deadline = time.monotonic() + 1.0
    last_error: ValueError | None = None
    while True:
        try:
            raw = path.read_text(encoding="ascii").strip()
            key = bytes.fromhex(raw)
        except FileNotFoundError:
            last_error = ValueError("Local spool seal key is missing.")
        except ValueError:
            last_error = ValueError("Local spool seal key is invalid.")
        else:
            if len(key) >= 32:
                return key
            last_error = ValueError("Local spool seal key is too short.")
        if time.monotonic() >= deadline:
            raise last_error or ValueError("Local spool seal key is invalid.")
        time.sleep(0.01)


def _canonical_spool_payload(payload: dict[str, Any]) -> bytes:
    sealed = dict(payload)
    for key in ("seal", "drained_at", "result", "error"):
        sealed.pop(key, None)
    return json.dumps(sealed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Spooled option {name} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(f"Spooled option {name} is outside allowed bounds.")
    return value


def _turn_text_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_turn_text_size(item) for item in value.values())
    if isinstance(value, list):
        return sum(_turn_text_size(item) for item in value)
    return len(str(value))


def _validate_artifact_snapshot_metadata(turn_payload: dict[str, Any], *, root: Path) -> None:
    snapshot_root = (root / "spool" / "snapshots").resolve()
    for key in ("files", "images"):
        for item in _as_list(turn_payload.get(key)):
            path, _, metadata = _artifact_item(item)
            resolved = _resolve_snapshot_path(path, root=root)
            try:
                resolved.relative_to(snapshot_root)
            except ValueError as exc:
                raise ValueError("Spooled artifacts must use enqueue-time snapshots.") from exc
            if not resolved.is_file():
                raise ValueError("Spooled artifact snapshot is missing.")
            expected = metadata.get("spool_snapshot_sha256")
            if not isinstance(expected, str) or not expected:
                raise ValueError("Spooled artifact snapshot is missing sha256 metadata.")
            if _sha256_file(resolved) != expected:
                raise ValueError("Spooled artifact snapshot sha256 mismatch.")
            if isinstance(item, dict):
                item["path"] = str(resolved)
                item_metadata = dict(item.get("metadata")) if isinstance(item.get("metadata"), dict) else {}
                item_metadata["spool_snapshot_path"] = str(resolved)
                item["metadata"] = item_metadata


def _restore_snapshot_artifact_paths(guarded_turn: dict[str, Any], raw_turn: dict[str, Any], *, root: Path) -> None:
    snapshot_root = (root / "spool" / "snapshots").resolve()
    for key in ("files", "images"):
        guarded_items = guarded_turn.get(key)
        raw_items = raw_turn.get(key)
        if not isinstance(guarded_items, list) or not isinstance(raw_items, list):
            continue
        for index, guarded_item in enumerate(guarded_items):
            if index >= len(raw_items) or not isinstance(guarded_item, dict):
                continue
            try:
                raw_path, _, _ = _artifact_item(raw_items[index])
                resolved = _resolve_snapshot_path(raw_path, root=root)
                resolved.relative_to(snapshot_root)
            except (TypeError, ValueError):
                continue
            guarded_item["path"] = str(resolved)


def _resolve_snapshot_path(path: Path, *, root: Path) -> Path:
    resolved = path.resolve()
    snapshot_root = (root / "spool" / "snapshots").resolve()
    try:
        resolved.relative_to(snapshot_root)
        return resolved
    except ValueError:
        pass
    parts = resolved.parts
    for index in range(len(parts) - 1):
        if parts[index].lower() == "spool" and parts[index + 1].lower() == "snapshots":
            candidate = (root / Path(*parts[index:])).resolve()
            if candidate.exists():
                return candidate
    return resolved


def _stabilize_after_drain(
    memory: Any,
    *,
    scopes: list[str],
    hot_budgets: dict[str, int],
    episode_min_group_size: int,
    session_min_group_size: int,
    candidate_min_group_size: int,
    limit: int,
) -> dict[str, Any]:
    scope_reports = []
    total_summaries = 0
    total_superseded = 0
    for scope in scopes:
        scope_summaries = 0
        scope_superseded = 0
        episode_patterns = []
        for pattern in ("command", "file_artifact", "git_status", "session"):
            if pattern == "session":
                min_group_size = session_min_group_size
            elif pattern == "git_status":
                min_group_size = min(episode_min_group_size, 3)
            else:
                min_group_size = episode_min_group_size
            report = memory.episode_summary(
                scope=scope,
                pattern=pattern,
                min_group_size=min_group_size,
                limit=limit,
                dry_run=False,
            ).as_dict()
            episode_patterns.append(report)
            scope_summaries += int(report["summaries_created"])
            scope_superseded += int(report["superseded"])
        candidate_report = memory.candidate_summary(
            scope=scope,
            pattern="all",
            min_group_size=candidate_min_group_size,
            limit=limit,
            dry_run=False,
        ).as_dict()
        scope_summaries += int(candidate_report["summaries_created"])
        scope_superseded += int(candidate_report["superseded"])
        total_summaries += scope_summaries
        total_superseded += scope_superseded
        hot_state = None
        hot_budget = hot_budgets.get(scope, 0)
        if hot_budget > 0 and (scope_summaries or scope_superseded):
            state = memory.build_hot(scope=scope, budget=hot_budget)
            hot_state = {"path": str(state.path), "estimated_tokens": state.estimated_tokens}
        scope_reports.append(
            {
                "scope": scope,
                "episode_summary": {"patterns": episode_patterns},
                "candidate_summary": candidate_report,
                "hot": hot_state,
            }
        )
    return {
        "passed": True,
        "scopes": scope_reports,
        "summaries_created": total_summaries,
        "superseded": total_superseded,
    }


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


def _recover_stale_processing(paths: dict[str, Path], *, stale_seconds: int, scope_filter: set[str] | None = None) -> int:
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
        if scope_filter is not None and _spool_record_scope(processing_path) not in {None, *scope_filter}:
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


def _spool_record_scope(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("format") != "ara-memory-spooled-turn-v1":
        return None
    options = payload.get("options")
    if not isinstance(options, dict):
        return "global"
    return str(options.get("scope", "global"))


def _scope_filter(scope: str | None) -> set[str] | None:
    if scope is None:
        return None
    return {str(scope), _canonical_scope_label(scope)}


def _canonical_scope_label(scope: str) -> str:
    return privacy_safe_label(str(scope), label="scope")


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


def _safe_payload(path: Path, *, failed_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"raw": payload}
    except Exception:
        failed_payload = {"format": "ara-memory-spooled-turn-v1", "spool_id": path.stem}
        raw_copy = failed_dir / f"{path.stem}.raw"
        failed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, raw_copy)
        failed_payload["raw_copy"] = str(raw_copy)
        return failed_payload


def _snapshot_turn_artifacts(
    turn: dict[str, Any],
    *,
    root: Path,
    spool_id: str,
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for key in ("files", "images"):
        if key not in turn:
            continue
        original_items = _as_list(turn.get(key))
        updated_items = []
        for index, item in enumerate(original_items):
            updated, snapshot = _snapshot_artifact_item(
                item,
                root=root,
                spool_id=spool_id,
                role=key[:-1],
                index=index,
            )
            updated_items.append(updated)
            if snapshot:
                snapshots.append(snapshot)
        turn[key] = updated_items
    return snapshots


def _snapshot_artifact_item(
    item: Any,
    *,
    root: Path,
    spool_id: str,
    role: str,
    index: int,
) -> tuple[Any, dict[str, Any] | None]:
    path, caption, metadata = _artifact_item(item)
    resolved = path.resolve()
    if not resolved.is_file():
        return item, None
    digest = _sha256_file(resolved)
    snapshot_dir = root / "spool" / "snapshots" / _safe_name(spool_id) / "artifacts"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = resolved.suffix if resolved.suffix and all(ch.isalnum() or ch == "." for ch in resolved.suffix) else ""
    snapshot_name = f"{role}-{index:03d}-{digest[:12]}{safe_suffix}"
    snapshot_path = snapshot_dir / snapshot_name
    if not snapshot_path.exists():
        shutil.copy2(resolved, snapshot_path)
    updated_metadata = dict(metadata)
    updated_metadata.update(
        {
            "spool_original_path": str(resolved),
            "spool_snapshot_path": str(snapshot_path),
            "spool_snapshot_sha256": digest,
            "spool_snapshot_bytes": resolved.stat().st_size,
        }
    )
    updated = {
        "path": str(snapshot_path),
        "caption": caption,
        "metadata": updated_metadata,
    }
    return updated, {
        "role": role,
        "index": index,
        "original_path": str(resolved),
        "snapshot_path": str(snapshot_path),
        "sha256": digest,
        "bytes": resolved.stat().st_size,
    }


def _artifact_item(item: Any) -> tuple[Path, str, dict[str, Any]]:
    if isinstance(item, str):
        return Path(item), "", {}
    if isinstance(item, dict):
        path = item.get("path")
        if not path:
            raise ValueError("Artifact item must include a path.")
        metadata = dict(item.get("metadata")) if isinstance(item.get("metadata"), dict) else {}
        caption = str(item.get("caption") or "").strip()
        return Path(str(path)), caption, metadata
    raise ValueError(f"Unsupported artifact item: {item!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "allow_raw_private": bool(raw.get("allow_raw_private", False)),
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
