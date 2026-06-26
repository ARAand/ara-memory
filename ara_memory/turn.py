from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
