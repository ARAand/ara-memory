from __future__ import annotations

import subprocess
from pathlib import Path

from ara_memory.compressors import compact_text
from ara_memory.core import AraMemory


def capture_worktree(
    memory: AraMemory,
    *,
    cwd: Path,
    scope: str,
    include_untracked_content: bool = False,
    max_file_chars: int = 8000,
) -> list[str]:
    event_ids: list[str] = []
    status = _git(cwd, ["status", "--short", "--branch"])
    if status.strip():
        event = memory.retain(
            kind="note",
            text=f"Git status for {cwd}:\n{status}",
            source="git-status",
            scope=scope,
            metadata={"cwd": str(cwd)},
        )
        event_ids.append(event.id)

    diff_stat = _git(cwd, ["diff", "--stat"])
    if diff_stat.strip():
        event = memory.retain(
            kind="diff",
            text=f"Git diff stat for {cwd}:\n{diff_stat}",
            source="git-diff-stat",
            scope=scope,
            metadata={"cwd": str(cwd)},
        )
        event_ids.append(event.id)

    diff_names = _git(cwd, ["diff", "--name-only"])
    if diff_names.strip():
        diff = _git(cwd, ["diff", "--", *diff_names.splitlines()])
        event = memory.retain(
            kind="diff",
            text=compact_text(f"Git diff for {cwd}:\n{diff}", limit=6000),
            source="git-diff",
            scope=scope,
            metadata={"cwd": str(cwd), "files": diff_names.splitlines()},
        )
        event_ids.append(event.id)

    untracked = _git(cwd, ["ls-files", "--others", "--exclude-standard"])
    if untracked.strip():
        names = untracked.splitlines()
        event = memory.retain(
            kind="file",
            text="Untracked file manifest:\n" + "\n".join(names),
            source="git-untracked-manifest",
            scope=scope,
            metadata={"cwd": str(cwd), "files": names, "file": "manifest"},
        )
        event_ids.append(event.id)
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
                event = memory.retain(
                    kind="file",
                    text=compact_text(f"Untracked file {name}:\n{text}", limit=max_file_chars),
                    source="git-untracked-file",
                    scope=scope,
                    metadata={"cwd": str(cwd), "file": name},
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
