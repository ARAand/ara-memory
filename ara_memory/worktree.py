from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text
from ara_memory.core import AraMemory


def capture_worktree(
    memory: AraMemory,
    *,
    cwd: Path,
    scope: str,
    include_untracked_content: bool = False,
    max_file_chars: int = 8000,
    allow_raw_private: bool = False,
) -> list[str]:
    snapshot = snapshot_worktree(
        cwd=cwd,
        include_untracked_content=include_untracked_content,
        max_file_chars=max_file_chars,
    )
    return retain_worktree_snapshot(memory, snapshot, scope=scope, allow_raw_private=allow_raw_private)


def snapshot_worktree(
    *,
    cwd: Path,
    include_untracked_content: bool = False,
    max_file_chars: int = 8000,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    status = _git(cwd, ["status", "--short", "--branch"])
    if status.strip():
        events.append(
            {
                "kind": "note",
                "text": f"Git status for {cwd}:\n{status}",
                "source": "git-status",
                "metadata": {"cwd": str(cwd)},
            }
        )

    diff_stat = _git(cwd, ["diff", "--stat"])
    if diff_stat.strip():
        events.append(
            {
                "kind": "diff",
                "text": f"Git diff stat for {cwd}:\n{diff_stat}",
                "source": "git-diff-stat",
                "metadata": {"cwd": str(cwd)},
            }
        )

    diff_names = _git(cwd, ["diff", "--name-only"])
    if diff_names.strip():
        diff = _git(cwd, ["diff", "--", *diff_names.splitlines()])
        events.append(
            {
                "kind": "diff",
                "text": compact_text(f"Git diff for {cwd}:\n{diff}", limit=6000),
                "source": "git-diff",
                "metadata": {"cwd": str(cwd), "files": diff_names.splitlines()},
            }
        )

    untracked = _git(cwd, ["ls-files", "--others", "--exclude-standard"])
    if untracked.strip():
        names = untracked.splitlines()
        events.append(
            {
                "kind": "file",
                "text": "Untracked file manifest:\n" + "\n".join(names),
                "source": "git-untracked-manifest",
                "metadata": {"cwd": str(cwd), "files": names, "file": "manifest"},
            }
        )
        if include_untracked_content:
            for name in names:
                path = (cwd / name).resolve()
                if not _is_safe_child(cwd.resolve(), path) or not path.is_file():
                    continue
                if path.stat().st_size > max_file_chars * 4:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                events.append(
                    {
                        "kind": "file",
                        "text": compact_text(f"Untracked file {name}:\n{text}", limit=max_file_chars),
                        "source": "git-untracked-file",
                        "metadata": {"cwd": str(cwd), "file": name},
                    }
                )
    return events


def retain_worktree_snapshot(
    memory: AraMemory,
    snapshot: list[dict[str, Any]],
    *,
    scope: str,
    metadata_extra: dict[str, Any] | None = None,
    allow_raw_private: bool = False,
) -> list[str]:
    event_ids: list[str] = []
    for index, item in enumerate(snapshot):
        metadata = dict(item.get("metadata") or {})
        metadata.update(metadata_extra or {})
        metadata["worktree_snapshot_index"] = index
        event = memory.retain(
            kind=str(item["kind"]),
            text=str(item["text"]),
            source=str(item.get("source") or "git-snapshot"),
            scope=scope,
            metadata=metadata,
            allow_raw_private=allow_raw_private,
        )
        event_ids.append(event.id)
    return event_ids


def _is_safe_child(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git(cwd: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "git executable not found"
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout
