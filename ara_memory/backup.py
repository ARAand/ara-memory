from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any

from ara_memory.archive_safety import (
    MAX_MANIFEST_BYTES,
    copy_member_to_path,
    inspect_zip,
    member_sha256,
    read_member_text,
)
from ara_memory.models import utc_now
from ara_memory.recall import RecallCompiler
from ara_memory.storage import MemoryStore, SCHEMA_VERSION


BACKUP_FORMAT = "ara-memory-backup-v2"
BACKUP_SIGNING_KEY_NAME = ".backup-signing-key"
ENTRY_HASHES_FIELD = "entry_hashes_sha256"
MANIFEST_SIGNATURE_FIELD = "manifest_hmac_sha256"
SIGNATURE_ALGORITHM = "hmac-sha256-manifest-v1"


@dataclass(slots=True)
class BackupResult:
    path: Path
    manifest: dict

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "manifest": self.manifest,
        }


def create_backup(store: MemoryStore, *, output: Path | None = None, include_archive: bool = True) -> BackupResult:
    store.init()
    output_path = (output or _default_backup_path(store.root)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_paths = {output_path}
    signing_key = _load_or_create_backup_signing_key(store.root)

    manifest = {
        "created_at": utc_now(),
        "schema_version": store.schema_version(),
        "stats": store.stats(),
        "include_archive": include_archive,
        "include_spool": True,
        "format": BACKUP_FORMAT,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "backup_key_id": _backup_key_id(signing_key),
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "memory.db"
        _snapshot_sqlite(store.db_path, tmp_db)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            entry_hashes: dict[str, str] = {}
            _write_file(zf, tmp_db, "memory.db", entry_hashes=entry_hashes)
            _write_tree(zf, store.ledger_dir, "ledger", exclude=excluded_paths, entry_hashes=entry_hashes)
            _write_tree(zf, store.hot_dir, "hot", exclude=excluded_paths, entry_hashes=entry_hashes)
            _write_tree(zf, store.root / "spool", "spool", exclude=excluded_paths, entry_hashes=entry_hashes)
            if include_archive:
                _write_tree(zf, store.archive_dir, "archive", exclude=excluded_paths, entry_hashes=entry_hashes)
            manifest[ENTRY_HASHES_FIELD] = entry_hashes
            manifest["entry_count"] = len(entry_hashes)
            manifest[MANIFEST_SIGNATURE_FIELD] = _manifest_signature(manifest, signing_key)
            zf.writestr("manifest.json", _manifest_json(manifest))

    return BackupResult(path=output_path, manifest=manifest)


def verify_backup(path: Path, *, trust_root: Path | None = None) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        safety = inspect_zip(zf)
        names = set(safety["names"])
        required = {"manifest.json", "memory.db"}
        missing = sorted(required - names)
        manifest_error = None
        if safety["passed"] and "manifest.json" in names:
            try:
                manifest = json.loads(read_member_text(zf, "manifest.json", max_bytes=MAX_MANIFEST_BYTES))
            except Exception as exc:
                manifest = {}
                manifest_error = f"{type(exc).__name__}: {exc}"
        else:
            manifest = {}
        hash_report = _verify_entry_hashes(zf, names, manifest) if safety["passed"] else _skipped_hash_report()
        signature_report = _verify_manifest_signature(
            manifest,
            trust_root=_infer_trust_root(path, trust_root=trust_root),
        )
        format_ok = manifest.get("format") == BACKUP_FORMAT
        signature_algorithm_ok = manifest.get("signature_algorithm") == SIGNATURE_ALGORITHM
        integrity = "not checked"
        foreign_key_violations: list[tuple[Any, ...]] = []
        sqlite_error = None
        if safety["passed"] and "memory.db" in names:
            with tempfile.TemporaryDirectory() as tmp:
                db_path = Path(tmp) / "memory.db"
                try:
                    copy_member_to_path(zf, "memory.db", db_path)
                    conn = sqlite3.connect(db_path)
                    try:
                        conn.execute("PRAGMA foreign_keys=ON")
                        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                        foreign_key_violations = [
                            tuple(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()
                        ]
                    finally:
                        conn.close()
                except Exception as exc:
                    sqlite_error = f"{type(exc).__name__}: {exc}"
        passed = (
            safety["passed"]
            and not missing
            and manifest_error is None
            and sqlite_error is None
            and format_ok
            and signature_algorithm_ok
            and integrity == "ok"
            and not foreign_key_violations
            and hash_report["status"] == "ok"
            and signature_report["status"] == "ok"
        )
        return {
            "path": str(path),
            "passed": passed,
            "missing": missing,
            "archive_safety": safety,
            "sqlite_integrity": integrity,
            "manifest_error": manifest_error,
            "sqlite_error": sqlite_error,
            "format_ok": format_ok,
            "signature_algorithm_ok": signature_algorithm_ok,
            "foreign_key_violations": foreign_key_violations[:20],
            "manifest": manifest,
            "entries": len(names),
            "entry_hashes": hash_report,
            "manifest_signature": signature_report,
        }


def restore_backup(path: Path, target_root: Path, *, force: bool = False, trust_root: Path | None = None) -> dict:
    path = path.resolve()
    effective_trust_root = _infer_trust_root(path, trust_root=trust_root)
    target_root = target_root.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        restore_source = Path(tmp) / path.name
        _copy_path(path, restore_source)
        verification = verify_backup(restore_source, trust_root=effective_trust_root)
        if not verification["passed"]:
            raise ValueError(f"Backup verification failed: {verification}")

        if target_root.exists() and any(target_root.iterdir()):
            if not force:
                raise FileExistsError(f"Target memory root is not empty: {target_root}")
            _clear_memory_root(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(restore_source, "r") as zf:
            safety = inspect_zip(zf)
            if not safety["passed"]:
                raise ValueError(f"Unsafe backup archive: {safety}")
            for info in zf.infolist():
                name = info.filename
                if name.endswith("/"):
                    continue
                destination = _safe_restore_destination(target_root, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                copy_member_to_path(zf, name, destination)

        restored = MemoryStore(target_root)
        restored.init()
        post = verify_backup(restore_source, trust_root=effective_trust_root)
        source_schema = int(verification["manifest"].get("schema_version", 0) or 0)
        restored_schema = restored.schema_version()
        schema_ok = (
            restored_schema == source_schema
            or (0 < source_schema <= SCHEMA_VERSION and restored_schema == SCHEMA_VERSION)
        )
        return {
            "path": str(path),
            "target_root": str(target_root),
            "passed": post["passed"] and schema_ok,
            "manifest": verification["manifest"],
            "source_schema_version": source_schema,
            "restored_schema_version": restored_schema,
            "stats": restored.stats(),
        }


def drill_restore_backup(
    path: Path,
    *,
    scope: str = "global",
    recall_query: str | None = None,
    recall_budget: int = 1200,
    trust_root: Path | None = None,
) -> dict:
    verification = verify_backup(path, trust_root=trust_root)
    if not verification["passed"]:
        return {
            "path": str(path),
            "passed": False,
            "backup_verification": verification,
            "restore": None,
            "recall": None,
        }

    with tempfile.TemporaryDirectory() as tmp:
        target_root = Path(tmp) / "restored-memory"
        restored = restore_backup(path, target_root, trust_root=trust_root)
        store = MemoryStore(target_root)
        recall = None
        if recall_query:
            result = RecallCompiler(store).recall(recall_query, scope=scope, budget=recall_budget)
            recall = {
                "query": recall_query,
                "scope": scope,
                "estimated_tokens": result.diagnostics["estimated_tokens_after"],
                "capsules_selected": result.diagnostics["capsules_selected"],
                "passed": bool(result.pack.strip()) and result.diagnostics["estimated_tokens_after"] <= recall_budget,
            }
        passed = bool(restored["passed"]) and (recall is None or recall["passed"])
        return {
            "path": str(path),
            "passed": passed,
            "backup_verification": verification,
            "restore": restored,
            "recall": recall,
        }


def _snapshot_sqlite(source: Path, target: Path) -> None:
    source_conn = sqlite3.connect(source)
    try:
        target_conn = sqlite3.connect(target)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
    finally:
        source_conn.close()


def _write_tree(
    zf: zipfile.ZipFile,
    root: Path,
    arc_root: str,
    *,
    exclude: set[Path] | None = None,
    entry_hashes: dict[str, str],
) -> None:
    if not root.exists():
        return
    excluded = exclude or set()
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        if resolved_path in excluded:
            continue
        if path.is_symlink() or _is_reparse_point(path):
            raise ValueError(f"Refusing to back up symlink or reparse point: {path}")
        if resolved_root not in (resolved_path, *resolved_path.parents):
            raise ValueError(f"Refusing to back up path outside tree: {path}")
        _write_file(zf, path, f"{arc_root}/{path.relative_to(root).as_posix()}", entry_hashes=entry_hashes)


def _write_file(zf: zipfile.ZipFile, path: Path, arcname: str, *, entry_hashes: dict[str, str]) -> None:
    entry_hashes[arcname] = _file_sha256(path)
    zf.write(path, arcname)


def _verify_entry_hashes(zf: zipfile.ZipFile, names: set[str], manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get(ENTRY_HASHES_FIELD)
    if not isinstance(expected, dict):
        return {
            "status": "missing",
            "checked": 0,
            "missing_hashes": [],
            "missing_entries": [],
            "extra_entries": [],
            "mismatches": [],
            "read_errors": [],
        }
    entries = {name for name in names if name != "manifest.json" and not name.endswith("/")}
    expected_names = {str(name) for name in expected}
    missing_hashes = sorted(entries - expected_names)
    missing_entries = sorted(expected_names - entries)
    mismatches = []
    read_errors = []
    for name in sorted(entries & expected_names):
        try:
            actual = member_sha256(zf, name)
        except Exception as exc:
            read_errors.append({"entry": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        wanted = str(expected[name])
        if not hmac.compare_digest(actual, wanted):
            mismatches.append({"entry": name, "expected": wanted, "actual": actual})
    status = "ok" if not missing_hashes and not missing_entries and not mismatches and not read_errors else "failed"
    return {
        "status": status,
        "checked": len(entries & expected_names),
        "missing_hashes": missing_hashes,
        "missing_entries": missing_entries,
        "extra_entries": missing_hashes,
        "mismatches": mismatches[:20],
        "read_errors": read_errors[:20],
    }


def _skipped_hash_report() -> dict[str, Any]:
    return {
        "status": "skipped",
        "checked": 0,
        "missing_hashes": [],
        "missing_entries": [],
        "extra_entries": [],
        "mismatches": [],
        "read_errors": [],
    }


def _verify_manifest_signature(manifest: dict[str, Any], *, trust_root: Path | None) -> dict[str, Any]:
    signature = manifest.get(MANIFEST_SIGNATURE_FIELD)
    if not isinstance(signature, str) or not signature:
        return {"status": "missing", "key_id": manifest.get("backup_key_id")}
    if trust_root is None:
        return {"status": "missing-key", "key_id": manifest.get("backup_key_id")}
    key = _load_backup_signing_key(trust_root)
    if key is None:
        return {"status": "missing-key", "key_id": manifest.get("backup_key_id")}
    actual = _manifest_signature(manifest, key)
    key_id = _backup_key_id(key)
    manifest_key_id = manifest.get("backup_key_id")
    signature_ok = hmac.compare_digest(actual, signature)
    key_id_ok = manifest_key_id == key_id
    return {
        "status": "ok" if signature_ok and key_id_ok else "failed",
        "key_id": key_id,
        "manifest_key_id": manifest_key_id,
        "key_id_ok": key_id_ok,
    }


def _manifest_signature(manifest: dict[str, Any], key: bytes) -> str:
    payload = dict(manifest)
    payload.pop(MANIFEST_SIGNATURE_FIELD, None)
    return hmac.new(key, _canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def _manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_path(source: Path, target: Path) -> None:
    with source.open("rb") as src, target.open("wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)


def _load_or_create_backup_signing_key(root: Path) -> bytes:
    existing = _load_backup_signing_key(root)
    if existing is not None:
        return existing
    path = root / BACKUP_SIGNING_KEY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_text(key.hex(), encoding="ascii")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def _load_backup_signing_key(root: Path) -> bytes | None:
    path = root / BACKUP_SIGNING_KEY_NAME
    try:
        raw = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        return None
    return key if len(key) >= 32 else None


def _backup_key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:16]


def _infer_trust_root(path: Path, *, trust_root: Path | None) -> Path | None:
    if trust_root is not None:
        return trust_root.resolve()
    return None


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attrs = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attrs & reparse_flag)


def _default_backup_path(root: Path) -> Path:
    stamp = utc_now().replace(":", "").replace("+", "Z")
    return root / "backups" / f"ara-memory-{stamp}.zip"


def _safe_restore_destination(root: Path, archive_name: str) -> Path:
    normalized = archive_name.replace("\\", "/")
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"Unsafe backup entry: {archive_name}")
    destination = (root / normalized).resolve()
    if root not in (destination, *destination.parents):
        raise ValueError(f"Unsafe backup entry: {archive_name}")
    return destination


def _clear_memory_root(root: Path) -> None:
    allowed = {"memory.db", "ledger", "archive", "hot", "spool", "backups", "manifest.json"}
    for child in root.iterdir():
        if child.name == BACKUP_SIGNING_KEY_NAME:
            continue
        if child.name not in allowed:
            raise FileExistsError(f"Refusing to remove unknown file in memory root: {child}")
        if child.is_dir():
            rmtree(child)
        else:
            child.unlink()
