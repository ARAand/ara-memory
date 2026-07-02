from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any

from ara_memory.archive_crypto import ensure_encrypted_archive_object
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
    archived = ensure_encrypted_archive_object(memory.store.root, resolved, digest)
    archive_path = archived.path
    archive_relative_path = archive_path.relative_to(memory.store.root).as_posix()

    metadata = {
        "original_path": str(resolved),
        "archive_path": archive_relative_path,
        "sha256": digest,
        "bytes": resolved.stat().st_size,
        "mime_type": mime_type,
    }
    metadata.update(archived.metadata())
    if metadata_extra:
        metadata.update(metadata_extra)

    if mime_type.startswith(IMAGE_PREFIX):
        text = (
            f"Image artifact: {resolved.name}\n"
            f"Caption: {caption or 'No caption provided.'}\n"
            f"SHA256: {digest}\n"
            f"Archived at: {archive_relative_path}"
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
        f"Archived at: {archive_relative_path}"
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


def ingest_artifact_bytes(
    memory: AraMemory,
    *,
    name: str,
    content: bytes,
    archive_relative_path: str,
    digest: str,
    scope: str,
    source: str = "file-ingest",
    caption: str = "",
    max_text_chars: int = 12000,
    metadata_extra: dict[str, Any] | None = None,
    allow_raw_private: bool = False,
) -> str:
    display_name = name or "artifact"
    mime_type = mimetypes.guess_type(display_name)[0] or "application/octet-stream"
    metadata = {
        "original_name": display_name,
        "archive_path": archive_relative_path,
        "sha256": digest,
        "bytes": len(content),
        "mime_type": mime_type,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    if mime_type.startswith(IMAGE_PREFIX):
        text = (
            f"Image artifact: {display_name}\n"
            f"Caption: {caption or 'No caption provided.'}\n"
            f"SHA256: {digest}\n"
            f"Archived at: {archive_relative_path}"
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

    if _looks_textual_name(display_name, mime_type):
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = content.decode("utf-8", errors="replace")
        text = compact_text(f"File artifact: {display_name}\nSHA256: {digest}\n\n{decoded}", limit=max_text_chars)
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
        f"Binary artifact: {display_name}\n"
        f"MIME: {mime_type}\n"
        f"SHA256: {digest}\n"
        f"Archived at: {archive_relative_path}"
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


def _looks_textual_name(name: str, mime_type: str) -> bool:
    if mime_type.startswith("text/"):
        return True
    return Path(name).suffix.lower() in TEXT_SUFFIXES
