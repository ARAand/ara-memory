from __future__ import annotations

import hashlib
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Iterator


MAX_ZIP_ENTRIES = 100_000
MAX_ZIP_ENTRY_BYTES = 512 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200.0
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_JSONL_BYTES = 512 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def inspect_zip(
    zf: zipfile.ZipFile,
    *,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_entry_bytes: int = MAX_ZIP_ENTRY_BYTES,
    max_total_uncompressed_bytes: int = MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES,
    max_compression_ratio: float = MAX_ZIP_COMPRESSION_RATIO,
) -> dict[str, Any]:
    infos = zf.infolist()
    inspected_infos = infos[: max_entries + 1] if len(infos) > max_entries else infos
    issues: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    collision_seen: set[str] = set()
    duplicate_names: list[str] = []
    normalized_duplicate_names: list[str] = []
    unsafe_names: list[str] = []
    oversized_entries: list[dict[str, Any]] = []
    high_ratio_entries: list[dict[str, Any]] = []
    total_uncompressed = 0
    total_compressed = 0

    if len(infos) > max_entries:
        issues.append("too_many_entries")
    for info in inspected_infos:
        name = info.filename
        names.append(name)
        if name in seen:
            duplicate_names.append(name)
        seen.add(name)
        collision_key = _archive_collision_key(name)
        if collision_key in collision_seen:
            normalized_duplicate_names.append(name)
        collision_seen.add(collision_key)
        if not _safe_archive_name(name):
            unsafe_names.append(name)
        total_uncompressed += int(info.file_size)
        total_compressed += int(info.compress_size)
        if info.file_size > max_entry_bytes:
            oversized_entries.append({"name": name, "bytes": int(info.file_size)})
        ratio = _compression_ratio(info)
        if ratio > max_compression_ratio:
            high_ratio_entries.append({"name": name, "ratio": ratio})

    if total_uncompressed > max_total_uncompressed_bytes:
        issues.append("total_uncompressed_bytes_exceeded")
    if duplicate_names or normalized_duplicate_names:
        issues.append("duplicate_names")
    if unsafe_names:
        issues.append("unsafe_names")
    if oversized_entries:
        issues.append("entry_bytes_exceeded")
    if high_ratio_entries:
        issues.append("compression_ratio_exceeded")

    return {
        "passed": not issues,
        "issues": issues,
        "entries": len(infos),
        "total_uncompressed_bytes": total_uncompressed,
        "total_compressed_bytes": total_compressed,
        "max_entries": max_entries,
        "max_entry_bytes": max_entry_bytes,
        "max_total_uncompressed_bytes": max_total_uncompressed_bytes,
        "max_compression_ratio": max_compression_ratio,
        "duplicate_names": duplicate_names[:20],
        "normalized_duplicate_names": normalized_duplicate_names[:20],
        "unsafe_names": unsafe_names[:20],
        "oversized_entries": oversized_entries[:20],
        "high_ratio_entries": high_ratio_entries[:20],
        "names": names,
    }


def read_member_bytes(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > max_bytes:
        raise ValueError(f"Archive member exceeds read limit: {name}")
    chunks: list[bytes] = []
    total = 0
    with zf.open(info, "r") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Archive member exceeds read limit while reading: {name}")
            chunks.append(chunk)
    return b"".join(chunks)


def read_member_text(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> str:
    return read_member_bytes(zf, name, max_bytes=max_bytes).decode("utf-8")


def iter_member_lines(
    zf: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int,
    max_line_bytes: int = MAX_JSONL_LINE_BYTES,
) -> Iterator[str]:
    info = zf.getinfo(name)
    if info.file_size > max_bytes:
        raise ValueError(f"Archive member exceeds read limit: {name}")
    total = 0
    with zf.open(info, "r") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                break
            total += len(line)
            if total > max_bytes:
                raise ValueError(f"Archive member exceeds read limit while reading: {name}")
            if len(line) > max_line_bytes:
                raise ValueError(f"Archive member line exceeds read limit: {name}")
            yield line.decode("utf-8")


def member_sha256(zf: zipfile.ZipFile, name: str, *, max_bytes: int = MAX_ZIP_ENTRY_BYTES) -> str:
    info = zf.getinfo(name)
    if info.file_size > max_bytes:
        raise ValueError(f"Archive member exceeds hash limit: {name}")
    digest = hashlib.sha256()
    total = 0
    with zf.open(info, "r") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Archive member exceeds hash limit while reading: {name}")
            digest.update(chunk)
    return digest.hexdigest()


def copy_member_to_path(zf: zipfile.ZipFile, name: str, target: Path, *, max_bytes: int = MAX_ZIP_ENTRY_BYTES) -> None:
    info = zf.getinfo(name)
    if info.file_size > max_bytes:
        raise ValueError(f"Archive member exceeds extract limit: {name}")
    total = 0
    with zf.open(info, "r") as source, target.open("wb") as dest:
        while True:
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Archive member exceeds extract limit while reading: {name}")
            dest.write(chunk)


def _safe_archive_name(name: str) -> bool:
    normalized = _normalized_archive_name(name)
    if not normalized or normalized.startswith("/") or "\x00" in normalized:
        return False
    parts = normalized.split("/")
    if any(_unsafe_path_part(part) for part in parts):
        return False
    path = Path(normalized)
    if path.drive or ".." in path.parts:
        return False
    return True


def _normalized_archive_name(name: str) -> str:
    return name.replace("\\", "/")


def _archive_collision_key(name: str) -> str:
    parts = _normalized_archive_name(name).split("/")
    return "/".join(unicodedata.normalize("NFC", part).rstrip(" .").casefold() for part in parts)


def _unsafe_path_part(part: str) -> bool:
    if part in {"", "."}:
        return True
    if ":" in part:
        return True
    if part != part.rstrip(" ."):
        return True
    normalized = unicodedata.normalize("NFC", part)
    stem = normalized.split(".", 1)[0].casefold()
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    return stem in reserved


def _compression_ratio(info: zipfile.ZipInfo) -> float:
    if info.file_size <= 0:
        return 0.0
    if info.compress_size <= 0:
        return float("inf")
    return float(info.file_size) / float(info.compress_size)
