from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ara_memory.models import utc_now


@dataclass(slots=True)
class LockResult:
    acquired: bool
    path: Path
    metadata: dict[str, Any]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "acquired": self.acquired,
            "path": str(self.path),
            "metadata": self.metadata,
            "reason": self.reason,
        }


class FileLock:
    def __init__(self, root: Path, name: str, *, stale_seconds: int = 3600) -> None:
        self.root = root
        self.name = name
        self.stale_seconds = stale_seconds
        self.path = root / "locks" / f"{name}.lock"
        self._held = False
        self.result: LockResult | None = None

    def acquire(self) -> LockResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": self.name,
            "pid": os.getpid(),
            "created_at": utc_now(),
            "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "",
        }
        while True:
            try:
                self.path.mkdir()
                (self.path / "owner.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                self._held = True
                self.result = LockResult(acquired=True, path=self.path, metadata=metadata)
                return self.result
            except FileExistsError:
                existing = _read_metadata(self.path)
                if self._is_stale():
                    _remove_lock_dir(self.path)
                    continue
                self.result = LockResult(
                    acquired=False,
                    path=self.path,
                    metadata=existing,
                    reason="lock_already_held",
                )
                return self.result

    def release(self) -> None:
        if not self._held:
            return
        _remove_lock_dir(self.path)
        self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        return self.stale_seconds > 0 and age > self.stale_seconds


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _remove_lock_dir(path: Path) -> None:
    try:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    except FileNotFoundError:
        return
