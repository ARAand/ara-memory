from __future__ import annotations

import json
import stat
from dataclasses import dataclass, field
from os import stat_result
from pathlib import Path
from typing import Any

from ara_memory.backup import verify_backup
from ara_memory.models import utc_now
from ara_memory.storage import MemoryStore


BACKUP_DELETE_CONFIRMATION = "DELETE OLD BACKUPS"


@dataclass(slots=True)
class BackupStewardshipItem:
    path: str
    bytes: int
    verified: bool
    created_at: str | None
    keep_reasons: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)
    delete_candidate: bool = False
    deleted: bool = False
    verification_cached: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "bytes": self.bytes,
            "verified": self.verified,
            "created_at": self.created_at,
            "keep_reasons": self.keep_reasons,
            "referenced_by": self.referenced_by,
            "delete_candidate": self.delete_candidate,
            "deleted": self.deleted,
            "verification_cached": self.verification_cached,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(slots=True)
class BackupStewardshipReport:
    root: str
    dry_run: bool
    passed: bool
    totals: dict[str, Any]
    items: list[BackupStewardshipItem]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "dry_run": self.dry_run,
            "passed": self.passed,
            "totals": self.totals,
            "items": [item.as_dict() for item in self.items],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            "# Ara Backup Stewardship",
            f"status: {'pass' if self.passed else 'blocked'}",
            f"dry_run: {self.dry_run}",
            (
                "totals: "
                f"backups={self.totals['backups']}, "
                f"bytes={self.totals['bytes']}, "
                f"delete_candidates={self.totals['delete_candidates']}, "
                f"candidate_bytes={self.totals['candidate_bytes']}, "
                f"deleted={self.totals['deleted']}, "
                f"delete_errors={self.totals['delete_errors']}, "
                f"verification_cache_hits={self.totals['verification_cache_hits']}"
            ),
        ]
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        lines.append("## Backups")
        for item in self.items[:20]:
            status = "delete-candidate" if item.delete_candidate else "keep"
            if item.deleted:
                status = "deleted"
            reasons = ", ".join(item.keep_reasons) if item.keep_reasons else "redundant"
            lines.append(f"- {status}: {item.path} ({item.bytes} bytes; {reasons})")
        if len(self.items) > 20:
            lines.append(f"- ... {len(self.items) - 20} more")
        return "\n".join(lines)


def run_backup_stewardship(
    store: MemoryStore,
    *,
    keep_latest: int = 3,
    keep_retention_cycles: int = 2,
    apply: bool = False,
    confirm: str = "",
) -> BackupStewardshipReport:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    if keep_retention_cycles < 0:
        raise ValueError("keep_retention_cycles must be non-negative")
    store.init()
    root = store.root
    backup_dir = root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(backup_dir):
        totals = _empty_totals(keep_latest=keep_latest, keep_retention_cycles=keep_retention_cycles)
        return BackupStewardshipReport(
            root=str(root),
            dry_run=not apply,
            passed=False,
            totals=totals,
            items=[],
            recommendations=[
                f"Refusing backup stewardship because backup directory is a symlink, junction, or reparse point: {backup_dir}",
                "Replace it with a real directory before running backup deletion review.",
            ],
        )

    referenced = _referenced_backup_paths(root, keep_retention_cycles=keep_retention_cycles)
    verification_cache = _load_verification_cache(root)
    current_cache: dict[str, Any] = {}
    scanned, scan_errors = _scan_backup_paths(backup_dir)
    scanned.sort(key=lambda item: item[1].st_mtime, reverse=True)
    items: list[BackupStewardshipItem] = []
    for index, (path, metadata) in enumerate(scanned):
        resolved = path.resolve(strict=False)
        info = _backup_item(path, metadata, verification_cache=verification_cache, current_cache=current_cache)
        if index < keep_latest:
            info.keep_reasons.append(f"latest-{index + 1}-of-{keep_latest}")
        if resolved in referenced:
            info.keep_reasons.append("referenced-by-retention-cycle")
            info.referenced_by.extend(referenced[resolved])
        if not info.verified:
            info.keep_reasons.append("verification-failed-inspect-manually")
        info.delete_candidate = info.verified and not info.keep_reasons
        items.append(info)
    items.extend(scan_errors)

    passed = True
    if apply and confirm != BACKUP_DELETE_CONFIRMATION:
        passed = False
        recommendations = [
            f"Refusing deletion without confirm={BACKUP_DELETE_CONFIRMATION!r}.",
            "Rerun without --apply for review, or pass the exact confirmation string after reviewing candidates.",
        ]
    else:
        recommendations = _recommend(items, apply=apply)

    deleted_bytes = 0
    deleted = 0
    deletion_errors = 0
    if apply and passed:
        for item in items:
            if not item.delete_candidate:
                continue
            path = Path(item.path)
            try:
                _assert_safe_backup_path(backup_dir, path)
                size = path.stat().st_size
                verification = verify_backup(path)
                if not verification.get("passed"):
                    item.verified = False
                    item.error = json.dumps(verification.get("missing") or verification, ensure_ascii=False)[:500]
                    item.keep_reasons.append("verification-failed-before-delete")
                    item.delete_candidate = False
                    deletion_errors += 1
                    passed = False
                    continue
                path.unlink()
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
                item.keep_reasons.append("delete-failed-inspect-manually")
                item.delete_candidate = False
                deletion_errors += 1
                passed = False
                continue
            item.deleted = True
            deleted += 1
            deleted_bytes += size

    if apply and deletion_errors:
        recommendations = [
            f"{deletion_errors} backup deletions failed; inspect item errors before rerunning.",
            "Successfully deleted candidates remain deleted; failed candidates were preserved in the report.",
        ]
    elif apply and passed and confirm == BACKUP_DELETE_CONFIRMATION:
        recommendations = _recommend(items, apply=True)

    candidate_bytes = sum(item.bytes for item in items if item.delete_candidate)
    totals = {
        "backups": len(items),
        "bytes": sum(item.bytes for item in items),
        "verified": sum(1 for item in items if item.verified),
        "verification_cache_hits": sum(1 for item in items if item.verification_cached),
        "delete_candidates": sum(1 for item in items if item.delete_candidate),
        "candidate_bytes": candidate_bytes,
        "deleted": deleted,
        "deleted_bytes": deleted_bytes,
        "delete_errors": deletion_errors,
        "keep_latest": keep_latest,
        "keep_retention_cycles": keep_retention_cycles,
    }
    _save_verification_cache(root, current_cache, deleted_paths={_item_cache_key(item) for item in items if item.deleted})
    return BackupStewardshipReport(
        root=str(root),
        dry_run=not apply,
        passed=passed,
        totals=totals,
        items=items,
        recommendations=recommendations,
    )


def _item_cache_key(item: BackupStewardshipItem) -> str:
    return str(Path(item.path).resolve(strict=False))


def _backup_item(
    path: Path,
    metadata: stat_result,
    *,
    verification_cache: dict[str, Any],
    current_cache: dict[str, Any],
) -> BackupStewardshipItem:
    cache_key = str(path.resolve(strict=False))
    cached = _cached_verification(verification_cache, cache_key=cache_key, metadata=metadata)
    verification_cached = cached is not None
    try:
        if _is_reparse_point(path):
            raise ValueError(f"Refusing to verify backup symlink or reparse point: {path}")
        verification = cached if cached is not None else verify_backup(path)
        verified = bool(verification.get("passed"))
        created_at = (verification.get("manifest") or {}).get("created_at")
        error = None if verified else json.dumps(verification.get("missing") or verification, ensure_ascii=False)[:500]
        if not verification_cached:
            current_cache[cache_key] = _cache_entry(metadata=metadata, verification=verification)
        else:
            current_cache[cache_key] = verification_cache["entries"][cache_key]
    except Exception as exc:
        verified = False
        created_at = None
        error = f"{type(exc).__name__}: {exc}"
    return BackupStewardshipItem(
        path=str(path),
        bytes=metadata.st_size,
        verified=verified,
        created_at=created_at,
        verification_cached=verification_cached,
        error=error,
    )


def _scan_backup_paths(backup_dir: Path) -> tuple[list[tuple[Path, stat_result]], list[BackupStewardshipItem]]:
    scanned: list[tuple[Path, stat_result]] = []
    errors: list[BackupStewardshipItem] = []
    for path in backup_dir.glob("*.zip"):
        try:
            metadata = path.stat()
        except OSError as exc:
            errors.append(
                BackupStewardshipItem(
                    path=str(path),
                    bytes=0,
                    verified=False,
                    created_at=None,
                    keep_reasons=["metadata-unavailable-inspect-manually"],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        scanned.append((path, metadata))
    return scanned, errors


def _referenced_backup_paths(root: Path, *, keep_retention_cycles: int) -> dict[Path, list[str]]:
    if keep_retention_cycles <= 0:
        return {}
    cycle_dir = root / "archive" / "retention-cycles"
    if not cycle_dir.exists():
        return {}
    refs: dict[Path, list[str]] = {}
    seen = 0
    for path in sorted(cycle_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not payload.get("passed"):
            continue
        backup_path = ((payload.get("backup") or {}).get("path") or "").strip()
        if not backup_path:
            continue
        resolved = _resolve_recorded_path(root, backup_path)
        refs.setdefault(resolved, []).append(str(path))
        seen += 1
        if seen >= keep_retention_cycles:
            break
    return refs


def _resolve_recorded_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = [path, root.parent / path, root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (root.parent / path).resolve()


def _assert_safe_backup_path(backup_dir: Path, path: Path) -> None:
    if _is_reparse_point(backup_dir):
        raise ValueError(f"Refusing to delete from backup symlink, junction, or reparse point: {backup_dir}")
    if _is_reparse_point(path):
        raise ValueError(f"Refusing to delete backup symlink or reparse point: {path}")
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Refusing to delete non-zip backup: {path}")
    lexical_backup_dir = backup_dir.absolute()
    lexical_path = path.absolute()
    if lexical_backup_dir not in (lexical_path, *lexical_path.parents):
        raise ValueError(f"Refusing to delete backup outside backup directory: {path}")
    resolved_backup_dir = backup_dir.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_backup_dir not in (resolved_path, *resolved_path.parents):
        raise ValueError(f"Refusing to delete backup outside resolved backup directory: {path}")


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _empty_totals(*, keep_latest: int, keep_retention_cycles: int) -> dict[str, Any]:
    return {
        "backups": 0,
        "bytes": 0,
        "verified": 0,
        "verification_cache_hits": 0,
        "delete_candidates": 0,
        "candidate_bytes": 0,
        "deleted": 0,
        "deleted_bytes": 0,
        "delete_errors": 0,
        "keep_latest": keep_latest,
        "keep_retention_cycles": keep_retention_cycles,
    }


def _cache_path(root: Path) -> Path:
    return root / "archive" / "backup-stewardship-cache.json"


def _load_verification_cache(root: Path) -> dict[str, Any]:
    path = _cache_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": {}}
    if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
        return {"version": 1, "entries": {}}
    return payload


def _save_verification_cache(root: Path, entries: dict[str, Any], *, deleted_paths: set[str]) -> None:
    path = _cache_path(root)
    filtered = {
        key: value
        for key, value in entries.items()
        if key not in deleted_paths and value.get("verification", {}).get("passed") is not None
    }
    payload = {"version": 1, "updated_at": utc_now(), "entries": filtered}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return


def _cached_verification(cache: dict[str, Any], *, cache_key: str, metadata: stat_result) -> dict[str, Any] | None:
    entry = (cache.get("entries") or {}).get(cache_key)
    if not isinstance(entry, dict):
        return None
    if entry.get("size") != metadata.st_size or entry.get("mtime_ns") != metadata.st_mtime_ns:
        return None
    verification = entry.get("verification")
    return verification if isinstance(verification, dict) else None


def _cache_entry(*, metadata: stat_result, verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "verified_at": utc_now(),
        "verification": verification,
    }


def _recommend(items: list[BackupStewardshipItem], *, apply: bool) -> list[str]:
    candidates = [item for item in items if item.delete_candidate]
    failed = [item for item in items if not item.verified]
    if apply:
        return [
            f"Deleted {len(candidates)} redundant verified backups.",
            "Latest backups, retention-cycle-referenced backups, and failed-verification backups were preserved.",
        ]
    if candidates:
        return [
            f"{len(candidates)} redundant verified backups can be deleted after review.",
            f"Run with --apply --confirm {BACKUP_DELETE_CONFIRMATION!r} to delete only those candidates.",
        ]
    if failed:
        return ["No verified redundant backups found; inspect failed-verification backups manually."]
    return ["Backup pressure is low under the current keep policy."]
