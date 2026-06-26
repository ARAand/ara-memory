from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ara_memory.compressors import estimate_tokens
from ara_memory.core import AraMemory
from ara_memory.ingest import ingest_file
from ara_memory.models import new_id, utc_now
from ara_memory.worktree import capture_worktree


def remember_turn(
    memory: AraMemory,
    turn: Mapping[str, Any],
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
) -> dict[str, Any]:
    memory.init()
    turn_id = str(turn.get("turn_id") or new_id("turn"))
    base_metadata = {
        "turn_id": turn_id,
        "turn_captured_at": utc_now(),
        "turn_source": source,
    }
    base_metadata.update(_mapping(turn.get("metadata")))

    event_ids: list[str] = []
    file_event_ids: list[str] = []
    worktree_event_ids: list[str] = []

    prompt = _text(turn.get("prompt") or turn.get("user"))
    if prompt:
        event_ids.append(
            memory.retain(
                kind="prompt",
                text=prompt,
                source=source,
                scope=scope,
                metadata={**base_metadata, "turn_role": "user"},
            ).id
        )

    assistant = _text(turn.get("assistant") or turn.get("response"))
    if assistant:
        event_ids.append(
            memory.retain(
                kind="assistant",
                text=assistant,
                source=source,
                scope=scope,
                metadata={**base_metadata, "turn_role": "assistant"},
            ).id
        )

    for index, note in enumerate(_items(turn.get("notes"))):
        event_ids.append(
            memory.retain(
                kind="note",
                text=_text(note),
                source=source,
                scope=scope,
                metadata={**base_metadata, "turn_role": "note", "turn_index": index},
            ).id
        )

    for index, decision in enumerate(_items(turn.get("decisions"))):
        event_ids.append(
            memory.retain(
                kind="decision",
                text=_text(decision),
                source=source,
                scope=scope,
                metadata={**base_metadata, "turn_role": "decision", "turn_index": index},
            ).id
        )

    for index, command in enumerate(_items(turn.get("commands"))):
        event_ids.append(
            memory.retain(
                kind="command",
                text=_command_text(command),
                source=source,
                scope=scope,
                metadata={**base_metadata, "turn_role": "command", "turn_index": index},
            ).id
        )

    for index, item in enumerate([*_items(turn.get("files")), *_items(turn.get("images"))]):
        path, caption, item_metadata = _artifact_item(item)
        file_event_ids.append(
            ingest_file(
                memory,
                path=path,
                scope=scope,
                source=source,
                caption=caption,
                max_text_chars=max_text_chars,
                metadata_extra={
                    **base_metadata,
                    **item_metadata,
                    "file": str(path),
                    "turn_role": "artifact",
                    "turn_index": index,
                },
            )
        )

    if capture_cwd is not None:
        worktree_event_ids = capture_worktree(
            memory,
            cwd=capture_cwd.resolve(),
            scope=scope,
            include_untracked_content=include_untracked_content,
            max_file_chars=max_file_chars,
        )

    capsules_created = memory.consolidate() if consolidate else 0
    sleep_report = memory.sleep(scope=scope).as_dict() if sleep else None
    hot_state = memory.build_hot(scope=scope, budget=hot_budget) if hot_budget > 0 else None

    return {
        "turn_id": turn_id,
        "events_retained": event_ids,
        "artifact_events_retained": file_event_ids,
        "worktree_events_retained": worktree_event_ids,
        "capsules_created": capsules_created,
        "sleep": sleep_report,
        "hot": {
            "path": str(hot_state.path),
            "estimated_tokens": hot_state.estimated_tokens,
        }
        if hot_state
        else None,
    }


def plan_turn_ingress(
    turn: Mapping[str, Any],
    *,
    capture_cwd: Path | None = None,
    max_file_chars: int = 8000,
    max_text_chars: int = 12000,
    direct_text_threshold: int = 16000,
) -> dict[str, Any]:
    """Return a bounded, model-free ingress plan for a turn envelope."""

    text_items = _turn_text_items(turn)
    raw_text_chars = sum(len(item["text"]) for item in text_items)
    raw_text_tokens = sum(estimate_tokens(item["text"]) for item in text_items)
    artifact_items = [*_items(turn.get("files")), *_items(turn.get("images"))]
    artifact_plans = [_artifact_plan(item, max_text_chars=max_text_chars) for item in artifact_items]
    artifact_bytes = sum(item.get("bytes", 0) for item in artifact_plans if item.get("exists"))
    missing_artifacts = [item for item in artifact_plans if not item.get("exists")]
    worktree = _worktree_plan(capture_cwd, max_file_chars=max_file_chars) if capture_cwd else None
    recommended_mode = "spool-turn" if raw_text_chars > direct_text_threshold or artifact_items or capture_cwd else "remember-turn"
    recall_preview_tokens = estimate_tokens(_recall_preview(text_items, artifact_plans))
    return {
        "recommended_mode": recommended_mode,
        "raw_text": {
            "items": len(text_items),
            "chars": raw_text_chars,
            "estimated_tokens_if_recalled_whole": raw_text_tokens,
        },
        "recall_preview": {
            "estimated_tokens": recall_preview_tokens,
            "max_text_chars_per_artifact": max_text_chars,
        },
        "artifacts": {
            "items": artifact_plans,
            "count": len(artifact_plans),
            "missing": len(missing_artifacts),
            "bytes": artifact_bytes,
        },
        "worktree": worktree,
        "policy": {
            "raw_preservation": "append-only local events and archived artifact objects",
            "token_control": "recall uses hot memory plus budgeted capsules; raw text is not reread unless selected",
            "cost_control": "planning, spooling, hashing, and local consolidation do not require an AI API call",
        },
    }


def execute_turn_ingress(
    memory: AraMemory,
    turn: Mapping[str, Any],
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
    direct_text_threshold: int = 16000,
    mode: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = plan_turn_ingress(
        turn,
        capture_cwd=capture_cwd,
        max_file_chars=max_file_chars,
        max_text_chars=max_text_chars,
        direct_text_threshold=direct_text_threshold,
    )
    selected_mode = plan["recommended_mode"] if mode == "auto" else mode
    if selected_mode not in {"remember-turn", "spool-turn"}:
        raise ValueError(f"Unsupported ingress mode: {mode}")
    if dry_run:
        return {"dry_run": True, "selected_mode": selected_mode, "plan": plan}
    if selected_mode == "spool-turn":
        record = memory.spool_turn(
            dict(turn),
            scope=scope,
            source=source,
            consolidate=consolidate,
            sleep=sleep,
            hot_budget=hot_budget,
            capture_cwd=capture_cwd,
            include_untracked_content=include_untracked_content,
            max_file_chars=max_file_chars,
            max_text_chars=max_text_chars,
        )
        return {
            "dry_run": False,
            "selected_mode": selected_mode,
            "plan": plan,
            "result": record.as_dict(),
        }
    result = remember_turn(
        memory,
        turn,
        scope=scope,
        source=source,
        consolidate=consolidate,
        sleep=sleep,
        hot_budget=hot_budget,
        capture_cwd=capture_cwd,
        include_untracked_content=include_untracked_content,
        max_file_chars=max_file_chars,
        max_text_chars=max_text_chars,
    )
    return {"dry_run": False, "selected_mode": selected_mode, "plan": plan, "result": result}


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _artifact_item(item: Any) -> tuple[Path, str, dict[str, Any]]:
    if isinstance(item, str):
        return Path(item), "", {}
    if isinstance(item, Mapping):
        path = item.get("path")
        if not path:
            raise ValueError("Artifact item must include a path.")
        metadata = _mapping(item.get("metadata"))
        caption = _text(item.get("caption"))
        return Path(str(path)), caption, metadata
    raise ValueError(f"Unsupported artifact item: {item!r}")


def _command_text(command: Any) -> str:
    if isinstance(command, str):
        return command.strip()
    if not isinstance(command, Mapping):
        return str(command).strip()
    parts = []
    if command.get("cmd"):
        parts.append(f"Command: {command['cmd']}")
    if command.get("cwd"):
        parts.append(f"CWD: {command['cwd']}")
    if "exit_code" in command:
        parts.append(f"Exit code: {command['exit_code']}")
    if command.get("output"):
        parts.append(f"Output:\n{command['output']}")
    return "\n".join(parts).strip()


def _turn_text_items(turn: Mapping[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for role, value in (("prompt", turn.get("prompt") or turn.get("user")), ("assistant", turn.get("assistant") or turn.get("response"))):
        text = _text(value)
        if text:
            items.append({"role": role, "text": text})
    for role, values in (("note", turn.get("notes")), ("decision", turn.get("decisions"))):
        for value in _items(values):
            text = _text(value)
            if text:
                items.append({"role": role, "text": text})
    for command in _items(turn.get("commands")):
        text = _command_text(command)
        if text:
            items.append({"role": "command", "text": text})
    return items


def _artifact_plan(item: Any, *, max_text_chars: int) -> dict[str, Any]:
    path, caption, metadata = _artifact_item(item)
    resolved = path.resolve()
    plan: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "caption": caption,
        "metadata_keys": sorted(metadata.keys()),
        "exists": resolved.is_file(),
    }
    if not resolved.is_file():
        return plan
    size = resolved.stat().st_size
    plan.update(
        {
            "bytes": size,
            "sha256": _sha256_file(resolved),
            "stored_as": "archive-object",
            "recall_text_limit": max_text_chars,
        }
    )
    return plan


def _worktree_plan(capture_cwd: Path | None, *, max_file_chars: int) -> dict[str, Any] | None:
    if capture_cwd is None:
        return None
    resolved = capture_cwd.resolve()
    return {
        "cwd": str(resolved),
        "exists": resolved.is_dir(),
        "diffs": "captured as bounded git status and diff events",
        "untracked_content": "manifest by default; full content only with include_untracked_content",
        "max_file_chars": max_file_chars,
    }


def _recall_preview(text_items: list[dict[str, str]], artifact_plans: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in text_items:
        parts.append(f"{item['role']}: {item['text'][:600]}")
    for artifact in artifact_plans:
        parts.append(
            "artifact: "
            f"{artifact.get('path')} "
            f"caption={artifact.get('caption') or ''} "
            f"sha256={artifact.get('sha256', 'missing')}"
        )
    return "\n".join(parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
