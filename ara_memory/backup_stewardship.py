from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from dataclasses import dataclass, field
from os import stat_result
from pathlib import Path
from typing import Any

from ara_memory.backup import verify_backup
from ara_memory.lock import FileLock
from ara_memory.models import utc_now
from ara_memory.storage import MemoryStore


BACKUP_DELETE_CONFIRMATION = "DELETE OLD BACKUPS"
FAILED_BACKUP_QUARANTINE_CONFIRMATION = "QUARANTINE FAILED BACKUPS"
VERIFICATION_CACHE_VERSION = 3
DEFAULT_TARGET_BACKUP_BYTES = 64 * 1024 * 1024


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
    quarantine_candidate: bool = False
    quarantined: bool = False
    quarantine_path: str | None = None
    quarantine_manifest_path: str | None = None
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
            "quarantine_candidate": self.quarantine_candidate,
            "quarantined": self.quarantined,
            "quarantine_path": self.quarantine_path,
            "quarantine_manifest_path": self.quarantine_manifest_path,
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
    lock: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "dry_run": self.dry_run,
            "passed": self.passed,
            "totals": self.totals,
            "items": [item.as_dict() for item in self.items],
            "recommendations": self.recommendations,
            "lock": self.lock,
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
                f"target_backup_bytes={self.totals['target_backup_bytes']}, "
                f"delete_candidates={self.totals['delete_candidates']}, "
                f"candidate_bytes={self.totals['candidate_bytes']}, "
                f"quarantine_candidates={self.totals['quarantine_candidates']}, "
                f"quarantine_candidate_bytes={self.totals['quarantine_candidate_bytes']}, "
                f"bytes_after_candidates={self.totals['bytes_after_candidates']}, "
                f"bytes_after_quarantine_candidates={self.totals['bytes_after_quarantine_candidates']}, "
                f"deleted={self.totals['deleted']}, "
                f"quarantined={self.totals['quarantined']}, "
                f"delete_errors={self.totals['delete_errors']}, "
                f"quarantine_errors={self.totals['quarantine_errors']}, "
                f"verification_cache_hits={self.totals['verification_cache_hits']}"
            ),
        ]
        if self.lock:
            lines.append(
                "lock: "
                f"acquired={self.lock.get('acquired')}, "
                f"reason={self.lock.get('reason', '')}"
            )
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        lines.append("## Backups")
        for item in self.items[:20]:
            status = "delete-candidate" if item.delete_candidate else "keep"
            if item.deleted:
                status = "deleted"
            if item.quarantine_candidate:
                status = "quarantine-candidate"
            if item.quarantined:
                status = "quarantined"
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
    target_backup_bytes: int | None = DEFAULT_TARGET_BACKUP_BYTES,
    quarantine_failed: bool = False,
    apply: bool = False,
    confirm: str = "",
    quarantine_confirm: str = "",
    use_lock: bool = True,
    lock_stale_seconds: int = 3600,
    write_cache: bool = True,
) -> BackupStewardshipReport:
    _validate_stewardship_args(
        keep_latest=keep_latest,
        keep_retention_cycles=keep_retention_cycles,
        target_backup_bytes=target_backup_bytes,
    )
    if apply and not write_cache:
        raise ValueError("write_cache=False is only supported for dry-run backup stewardship")
    store.init()
    root = store.root
    if apply and use_lock:
        lock = FileLock(root, "worker", stale_seconds=lock_stale_seconds)
        lock_result = lock.acquire()
        lock_payload = lock_result.as_dict()
        if not lock_result.acquired:
            return BackupStewardshipReport(
                root=str(root),
                dry_run=False,
                passed=False,
                totals=_empty_totals(
                    keep_latest=keep_latest,
                    keep_retention_cycles=keep_retention_cycles,
                    target_backup_bytes=target_backup_bytes,
                    write_cache=write_cache,
                ),
                items=[],
                recommendations=[
                    "Backup stewardship apply skipped because the memory worker lock is already held.",
                    "Rerun after the active worker finishes, or use --no-lock only when the store is otherwise quiescent.",
                ],
                lock=lock_payload,
            )
        try:
            return _run_backup_stewardship_unlocked(
                store,
                keep_latest=keep_latest,
                keep_retention_cycles=keep_retention_cycles,
                target_backup_bytes=target_backup_bytes,
                quarantine_failed=quarantine_failed,
                apply=apply,
                confirm=confirm,
                quarantine_confirm=quarantine_confirm,
                lock=lock_payload,
                write_cache=write_cache,
            )
        finally:
            lock.release()
    return _run_backup_stewardship_unlocked(
        store,
        keep_latest=keep_latest,
        keep_retention_cycles=keep_retention_cycles,
        target_backup_bytes=target_backup_bytes,
        quarantine_failed=quarantine_failed,
        apply=apply,
        confirm=confirm,
        quarantine_confirm=quarantine_confirm,
        lock=None,
        write_cache=write_cache,
    )


def _run_backup_stewardship_unlocked(
    store: MemoryStore,
    *,
    keep_latest: int = 3,
    keep_retention_cycles: int = 2,
    target_backup_bytes: int | None = DEFAULT_TARGET_BACKUP_BYTES,
    quarantine_failed: bool = False,
    apply: bool = False,
    confirm: str = "",
    quarantine_confirm: str = "",
    lock: dict[str, Any] | None = None,
    write_cache: bool = True,
) -> BackupStewardshipReport:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    if keep_retention_cycles < 0:
        raise ValueError("keep_retention_cycles must be non-negative")
    if target_backup_bytes is not None and target_backup_bytes < 0:
        raise ValueError("target_backup_bytes must be non-negative")
    store.init()
    root = store.root
    backup_dir = root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if _is_reparse_point(backup_dir):
        totals = _empty_totals(
            keep_latest=keep_latest,
            keep_retention_cycles=keep_retention_cycles,
            target_backup_bytes=target_backup_bytes,
            write_cache=write_cache,
        )
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
            lock=lock,
        )

    retention_refs = _referenced_backup_paths(root, keep_retention_cycles=keep_retention_cycles)
    prune_refs = _active_prune_approval_backup_paths(store)
    verification_cache = _load_verification_cache(root)
    current_cache: dict[str, Any] = {}
    scanned, scan_errors = _scan_backup_paths(backup_dir)
    scanned.sort(key=lambda item: item[1].st_mtime, reverse=True)
    items: list[BackupStewardshipItem] = []
    for index, (path, metadata) in enumerate(scanned):
        resolved = path.resolve(strict=False)
        info = _backup_item(
            path,
            metadata,
            trust_root=root,
            verification_cache=verification_cache,
            current_cache=current_cache,
        )
        if index < keep_latest:
            info.keep_reasons.append(f"latest-{index + 1}-of-{keep_latest}")
        if resolved in retention_refs:
            info.keep_reasons.append("referenced-by-retention-cycle")
            info.referenced_by.extend(retention_refs[resolved])
        if resolved in prune_refs:
            info.keep_reasons.append("referenced-by-active-prune-approval")
            info.referenced_by.extend(prune_refs[resolved])
        if not info.verified:
            info.keep_reasons.append("verification-failed-inspect-manually")
        items.append(info)
    items.extend(scan_errors)
    bytes_after_candidates = _mark_delete_candidates(items, target_backup_bytes=target_backup_bytes)
    bytes_after_quarantine = _mark_quarantine_candidates(items, enabled=quarantine_failed)

    passed = True
    delete_candidates = sum(1 for item in items if item.delete_candidate)
    quarantine_candidates = sum(1 for item in items if item.quarantine_candidate)
    delete_confirmed = confirm == BACKUP_DELETE_CONFIRMATION
    quarantine_confirmed = quarantine_confirm == FAILED_BACKUP_QUARANTINE_CONFIRMATION
    delete_allowed = bool(apply and delete_candidates and delete_confirmed)
    quarantine_allowed = bool(apply and quarantine_candidates and quarantine_failed and quarantine_confirmed)
    delete_missing_confirm = bool(apply and delete_candidates and not delete_confirmed)
    quarantine_missing_confirm = bool(
        apply and quarantine_candidates and quarantine_failed and not quarantine_confirmed
    )
    if apply and delete_missing_confirm and not quarantine_allowed:
        passed = False
        recommendations = [
            f"Refusing deletion without confirm={BACKUP_DELETE_CONFIRMATION!r}.",
            "Rerun without --apply for review, or pass the exact confirmation string after reviewing candidates.",
        ]
    elif apply and quarantine_missing_confirm and not delete_allowed:
        passed = False
        recommendations = [
            f"Refusing failed-backup quarantine without quarantine_confirm={FAILED_BACKUP_QUARANTINE_CONFIRMATION!r}.",
            "Rerun without --apply for review, or pass the exact quarantine confirmation after reviewing failed backups.",
        ]
    elif apply and delete_missing_confirm and quarantine_allowed:
        recommendations = [
            "Applying failed-backup quarantine only; redundant verified backups still require delete confirmation.",
            f"Pass confirm={BACKUP_DELETE_CONFIRMATION!r} in a separate reviewed run to delete redundant verified backups.",
        ]
    elif apply and quarantine_missing_confirm and delete_allowed:
        recommendations = [
            "Applying reviewed backup deletion only; failed-backup quarantine still requires quarantine confirmation.",
            f"Pass quarantine_confirm={FAILED_BACKUP_QUARANTINE_CONFIRMATION!r} in a separate reviewed run to quarantine failed backups.",
        ]
    else:
        recommendations = _recommend(
            items,
            apply=apply,
            target_backup_bytes=target_backup_bytes,
            bytes_after_candidates=bytes_after_candidates,
            bytes_after_quarantine=bytes_after_quarantine,
        )

    deleted_bytes = 0
    deleted = 0
    deletion_errors = 0
    quarantined = 0
    quarantine_bytes = 0
    quarantine_errors = 0
    partial_apply_notes: list[str] = []
    if apply and delete_missing_confirm and quarantine_allowed:
        partial_apply_notes = [
            "Applied failed-backup quarantine only; redundant verified backups still require delete confirmation.",
            f"Pass confirm={BACKUP_DELETE_CONFIRMATION!r} in a separate reviewed run to delete redundant verified backups.",
        ]
    elif apply and quarantine_missing_confirm and delete_allowed:
        partial_apply_notes = [
            "Applied reviewed backup deletion only; failed-backup quarantine still requires quarantine confirmation.",
            f"Pass quarantine_confirm={FAILED_BACKUP_QUARANTINE_CONFIRMATION!r} in a separate reviewed run to quarantine failed backups.",
        ]
    if apply and passed and quarantine_allowed:
        quarantine_dir = root / "archive" / "failed-backups"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            if not item.quarantine_candidate:
                continue
            path = Path(item.path)
            target: Path | None = None
            manifest_path: Path | None = None
            try:
                _assert_safe_backup_path(backup_dir, path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                final_verification = _verify_before_quarantine(path, trust_root=root)
                if final_verification.get("passed"):
                    item.verified = True
                    item.quarantine_candidate = False
                    item.error = None
                    item.keep_reasons.append("verification-passed-before-quarantine")
                    continue
                target = _unique_quarantine_path(quarantine_dir, path.name)
                path.replace(target)
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
                item.keep_reasons.append("quarantine-failed-inspect-manually")
                item.quarantine_candidate = False
                quarantine_errors += 1
                passed = False
                continue
            try:
                manifest_path = _write_quarantine_manifest(
                    target,
                    original_path=path,
                    item=item,
                    final_verification=final_verification,
                    quarantine_confirm=quarantine_confirm,
                )
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
                item.keep_reasons.append("quarantine-manifest-failed-inspect-manually")
                quarantine_errors += 1
                passed = False
            item.quarantined = True
            item.quarantine_candidate = False
            item.quarantine_path = str(target)
            item.quarantine_manifest_path = str(manifest_path) if manifest_path else None
            quarantined += 1
            quarantine_bytes += item.bytes

    if apply and passed and delete_allowed:
        for item in items:
            if not item.delete_candidate:
                continue
            path = Path(item.path)
            try:
                _assert_safe_backup_path(backup_dir, path)
                size = path.stat().st_size
                verification = verify_backup(path, trust_root=root)
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
    elif apply and passed and delete_allowed:
        recommendations = _recommend(
            items,
            apply=True,
            target_backup_bytes=target_backup_bytes,
            bytes_after_candidates=bytes_after_candidates,
            bytes_after_quarantine=bytes_after_quarantine,
        )
    elif apply and passed and quarantine_allowed:
        recommendations = _recommend(
            items,
            apply=True,
            target_backup_bytes=target_backup_bytes,
            bytes_after_candidates=bytes_after_candidates,
            bytes_after_quarantine=bytes_after_quarantine,
        )
    if apply and quarantine_errors:
        recommendations = [
            f"{quarantine_errors} failed-backup quarantines failed; inspect item errors before rerunning.",
            "Successfully quarantined backups remain under archive/failed-backups.",
        ]
    if partial_apply_notes and passed:
        recommendations = [*partial_apply_notes, *recommendations]

    candidate_bytes = sum(item.bytes for item in items if item.delete_candidate)
    quarantine_candidate_bytes = sum(item.bytes for item in items if item.quarantine_candidate)
    totals = {
        "backups": len(items),
        "bytes": sum(item.bytes for item in items),
        "verified": sum(1 for item in items if item.verified),
        "verification_cache_hits": sum(1 for item in items if item.verification_cached),
        "target_backup_bytes": target_backup_bytes,
        "bytes_after_candidates": bytes_after_candidates,
        "bytes_after_quarantine_candidates": bytes_after_quarantine,
        "target_reached": target_backup_bytes is None or bytes_after_candidates <= target_backup_bytes,
        "protected_bytes": sum(item.bytes for item in items if item.keep_reasons),
        "eligible_bytes": sum(item.bytes for item in items if item.verified and not item.keep_reasons),
        "delete_candidates": sum(1 for item in items if item.delete_candidate),
        "candidate_bytes": candidate_bytes,
        "quarantine_candidates": sum(1 for item in items if item.quarantine_candidate),
        "quarantine_candidate_bytes": quarantine_candidate_bytes,
        "quarantined": quarantined,
        "quarantine_bytes": quarantine_bytes,
        "quarantine_errors": quarantine_errors,
        "deleted": deleted,
        "deleted_bytes": deleted_bytes,
        "delete_errors": deletion_errors,
        "keep_latest": keep_latest,
        "keep_retention_cycles": keep_retention_cycles,
        "verification_cache_write_enabled": write_cache,
        "verification_cache_entries": len(current_cache),
    }
    if write_cache:
        _save_verification_cache(
            root,
            current_cache,
            deleted_paths={_item_cache_key(item) for item in items if item.deleted or item.quarantined},
        )
    return BackupStewardshipReport(
        root=str(root),
        dry_run=not apply,
        passed=passed,
        totals=totals,
        items=items,
        recommendations=recommendations,
        lock=lock,
    )


def _validate_stewardship_args(
    *,
    keep_latest: int,
    keep_retention_cycles: int,
    target_backup_bytes: int | None,
) -> None:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    if keep_retention_cycles < 0:
        raise ValueError("keep_retention_cycles must be non-negative")
    if target_backup_bytes is not None and target_backup_bytes < 0:
        raise ValueError("target_backup_bytes must be non-negative")


def _item_cache_key(item: BackupStewardshipItem) -> str:
    return str(Path(item.path).resolve(strict=False))


def _mark_delete_candidates(items: list[BackupStewardshipItem], *, target_backup_bytes: int | None) -> int:
    for item in items:
        item.delete_candidate = False
    eligible = [item for item in items if item.verified and not item.keep_reasons]
    if target_backup_bytes is None:
        for item in eligible:
            item.delete_candidate = True
        return sum(item.bytes for item in items if not item.delete_candidate)

    remaining_bytes = sum(item.bytes for item in items)
    for item in reversed(eligible):
        if remaining_bytes <= target_backup_bytes:
            break
        item.delete_candidate = True
        remaining_bytes -= item.bytes
    return remaining_bytes


def _mark_quarantine_candidates(items: list[BackupStewardshipItem], *, enabled: bool) -> int:
    for item in items:
        item.quarantine_candidate = False
    verified_exists = any(item.verified for item in items)
    if not enabled or not verified_exists:
        return sum(item.bytes for item in items)
    for item in items:
        if item.verified or item.deleted:
            continue
        if not Path(item.path).name.lower().endswith(".zip"):
            continue
        item.quarantine_candidate = True
    return sum(item.bytes for item in items if not item.quarantine_candidate)


def _backup_item(
    path: Path,
    metadata: stat_result,
    *,
    trust_root: Path,
    verification_cache: dict[str, Any],
    current_cache: dict[str, Any],
) -> BackupStewardshipItem:
    cache_key = str(path.resolve(strict=False))
    cached = _cached_verification(verification_cache, cache_key=cache_key, metadata=metadata)
    verification_cached = cached is not None
    try:
        if _is_reparse_point(path):
            raise ValueError(f"Refusing to verify backup symlink or reparse point: {path}")
        verification = cached if cached is not None else verify_backup(path, trust_root=trust_root)
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


def _verify_before_quarantine(path: Path, *, trust_root: Path) -> dict[str, Any]:
    try:
        return verify_backup(path, trust_root=trust_root)
    except Exception as exc:
        return {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


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
    for path in _sorted_existing_json_files(cycle_dir):
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


def _sorted_existing_json_files(directory: Path) -> list[Path]:
    files: list[tuple[float, Path]] = []
    for path in directory.glob("*.json"):
        try:
            files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    return [path for _, path in sorted(files, key=lambda item: item[0], reverse=True)]


def _active_prune_approval_backup_paths(store: MemoryStore) -> dict[Path, list[str]]:
    refs: dict[Path, list[str]] = {}
    now = datetime.now(timezone.utc)
    with store.session() as conn:
        rows = conn.execute(
            """
            SELECT id, backup_path, expires_at
            FROM prune_approvals
            WHERE status = 'prepared'
            """
        ).fetchall()
    for row in rows:
        expires_at = _parse_datetime(row["expires_at"])
        if expires_at is None or expires_at <= now:
            continue
        resolved = _resolve_recorded_path(store.root, str(row["backup_path"]))
        refs.setdefault(resolved, []).append(f"prune-approval:{row['id']}")
    return refs


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _unique_quarantine_path(quarantine_dir: Path, name: str) -> Path:
    candidate = quarantine_dir / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(1, 10_000):
        candidate = quarantine_dir / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not allocate quarantine path for {name}")


def _write_quarantine_manifest(
    target: Path,
    *,
    original_path: Path,
    item: BackupStewardshipItem,
    final_verification: dict[str, Any],
    quarantine_confirm: str,
) -> Path:
    manifest_path = target.with_name(f"{target.name}.quarantine.json")
    payload = {
        "format": "ara-memory-failed-backup-quarantine-v1",
        "quarantined_at": utc_now(),
        "original_path": str(original_path),
        "quarantine_path": str(target),
        "bytes": item.bytes,
        "created_at": item.created_at,
        "initial_error": item.error,
        "final_verification": final_verification,
        "confirmation": quarantine_confirm,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


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


def _empty_totals(
    *,
    keep_latest: int,
    keep_retention_cycles: int,
    target_backup_bytes: int | None,
    write_cache: bool = True,
) -> dict[str, Any]:
    return {
        "backups": 0,
        "bytes": 0,
        "verified": 0,
        "verification_cache_hits": 0,
        "target_backup_bytes": target_backup_bytes,
        "bytes_after_candidates": 0,
        "bytes_after_quarantine_candidates": 0,
        "target_reached": True,
        "protected_bytes": 0,
        "eligible_bytes": 0,
        "delete_candidates": 0,
        "candidate_bytes": 0,
        "quarantine_candidates": 0,
        "quarantine_candidate_bytes": 0,
        "quarantined": 0,
        "quarantine_bytes": 0,
        "quarantine_errors": 0,
        "deleted": 0,
        "deleted_bytes": 0,
        "delete_errors": 0,
        "keep_latest": keep_latest,
        "keep_retention_cycles": keep_retention_cycles,
        "verification_cache_write_enabled": write_cache,
        "verification_cache_entries": 0,
    }


def _cache_path(root: Path) -> Path:
    return root / "archive" / "backup-stewardship-cache.json"


def _load_verification_cache(root: Path) -> dict[str, Any]:
    path = _cache_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": VERIFICATION_CACHE_VERSION, "entries": {}}
    if payload.get("version") != VERIFICATION_CACHE_VERSION or not isinstance(payload.get("entries"), dict):
        return {"version": VERIFICATION_CACHE_VERSION, "entries": {}}
    return payload


def _save_verification_cache(root: Path, entries: dict[str, Any], *, deleted_paths: set[str]) -> None:
    path = _cache_path(root)
    filtered = {
        key: value
        for key, value in entries.items()
        if key not in deleted_paths and value.get("verification", {}).get("passed") is not None
    }
    payload = {"version": VERIFICATION_CACHE_VERSION, "updated_at": utc_now(), "entries": filtered}
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


def _recommend(
    items: list[BackupStewardshipItem],
    *,
    apply: bool,
    target_backup_bytes: int | None,
    bytes_after_candidates: int,
    bytes_after_quarantine: int,
) -> list[str]:
    candidates = [item for item in items if item.delete_candidate]
    quarantine_candidates = [item for item in items if item.quarantine_candidate]
    quarantined = [item for item in items if item.quarantined]
    deleted = [item for item in items if item.deleted]
    failed = [item for item in items if not item.verified]
    if apply:
        if deleted:
            recommendations = [
                f"Deleted {len(deleted)} redundant verified backups.",
                "Latest backups, retention-cycle-referenced backups, and failed-verification backups were preserved.",
            ]
            if target_backup_bytes is not None and bytes_after_candidates > target_backup_bytes:
                recommendations.append(
                    "Backup bytes still exceed the target because protected backups alone are above the budget."
                )
            return recommendations
        if quarantined:
            recommendations = [
                f"Quarantined {len(quarantined)} failed-verification backups under archive/failed-backups.",
                "Verified backups were preserved in the live backup directory.",
            ]
            if candidates:
                recommendations.append(
                    f"{len(candidates)} redundant verified backups still require delete confirmation."
                )
            if target_backup_bytes is not None and bytes_after_quarantine > target_backup_bytes:
                recommendations.append(
                    "Backup bytes still exceed the target after quarantine because protected verified backups are above the budget."
                )
            return recommendations
        return ["No backup files were moved or deleted."]
    if candidates:
        recommendations = [
            f"{len(candidates)} redundant verified backups can be deleted after review.",
            f"Run with --apply --confirm {BACKUP_DELETE_CONFIRMATION!r} to delete only those candidates.",
        ]
        if target_backup_bytes is not None:
            recommendations.insert(
                1,
                f"Candidate set is sized to leave about {bytes_after_candidates} backup bytes against target {target_backup_bytes}.",
        )
        return recommendations
    if quarantine_candidates:
        recommendations = [
            f"{len(quarantine_candidates)} failed-verification backups can be moved to archive/failed-backups after review.",
            f"Run with --apply --quarantine-failed --quarantine-confirm {FAILED_BACKUP_QUARANTINE_CONFIRMATION!r} to preserve them outside the live backup pool.",
        ]
        if target_backup_bytes is not None:
            recommendations.insert(
                1,
                f"Quarantine would leave about {bytes_after_quarantine} live backup bytes against target {target_backup_bytes}.",
            )
        return recommendations
    if failed:
        return ["No verified redundant backups found; inspect failed-verification backups manually."]
    if target_backup_bytes is not None:
        if bytes_after_candidates > target_backup_bytes:
            return [
                "No redundant verified backups remain under the current keep policy, "
                f"but protected backups require {bytes_after_candidates} bytes against target {target_backup_bytes}."
            ]
        return [f"Backup pressure is low under the current target budget ({target_backup_bytes} bytes)."]
    return ["Backup pressure is low under the current keep policy."]
