from __future__ import annotations

import hashlib
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text
from ara_memory.core import AraMemory


TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".htm",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

IMAGE_PREFIX = "image/"


def ingest_file(
    memory: AraMemory,
    *,
    path: Path,
    scope: str,
    source: str = "file-ingest",
    caption: str = "",
    max_text_chars: int = 12000,
    metadata_extra: dict[str, Any] | None = None,
    allow_raw_private: bool = False,
) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(str(path))

    digest = _sha256(resolved)
    mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    archive_path = memory.store.archive_dir / "objects" / digest[:2] / digest
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        shutil.copy2(resolved, archive_path)

    metadata = {
        "original_path": str(resolved),
        "archive_path": str(archive_path),
        "sha256": digest,
        "bytes": resolved.stat().st_size,
        "mime_type": mime_type,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    if mime_type.startswith(IMAGE_PREFIX):
        text = (
            f"Image artifact: {resolved.name}\n"
            f"Caption: {caption or 'No caption provided.'}\n"
            f"SHA256: {digest}\n"
            f"Archived at: {archive_path}"
        )
        event = memory.retain(
            kind="image",
            text=text,
            source=source,
            scope=scope,
            metadata=metadata,
            allow_raw_private=allow_raw_private,
        )
        return event.id

    if _looks_textual(resolved, mime_type):
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        text = compact_text(f"File artifact: {resolved.name}\nSHA256: {digest}\n\n{content}", limit=max_text_chars)
        event = memory.retain(
            kind="file",
            text=text,
            source=source,
            scope=scope,
            metadata=metadata,
            allow_raw_private=allow_raw_private,
        )
        return event.id

    text = (
        f"Binary artifact: {resolved.name}\n"
        f"MIME: {mime_type}\n"
        f"SHA256: {digest}\n"
        f"Archived at: {archive_path}"
    )
    event = memory.retain(
        kind="file",
        text=text,
        source=source,
        scope=scope,
        metadata=metadata,
        allow_raw_private=allow_raw_private,
    )
    return event.id


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_textual(path: Path, mime_type: str) -> bool:
    if mime_type.startswith("text/"):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES
