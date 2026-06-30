from __future__ import annotations

import json
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any

from ara_memory.models import utc_now
from ara_memory.recall import RecallCompiler
from ara_memory.storage import MemoryStore, SCHEMA_VERSION


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
    output_path = output or _default_backup_path(store.root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": utc_now(),
        "schema_version": store.schema_version(),
        "stats": store.stats(),
        "include_archive": include_archive,
        "include_spool": True,
        "format": "ara-memory-backup-v1",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "memory.db"
        _snapshot_sqlite(store.db_path, tmp_db)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            zf.write(tmp_db, "memory.db")
            _write_tree(zf, store.ledger_dir, "ledger")
            _write_tree(zf, store.hot_dir, "hot")
            _write_tree(zf, store.root / "spool", "spool")
            if include_archive:
                _write_tree(zf, store.archive_dir, "archive")

    return BackupResult(path=output_path, manifest=manifest)


def verify_backup(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        required = {"manifest.json", "memory.db"}
        missing = sorted(required - names)
        manifest = json.loads(zf.read("manifest.json").decode("utf-8")) if "manifest.json" in names else {}
        integrity = "not checked"
        foreign_key_violations: list[tuple[Any, ...]] = []
        if "memory.db" in names:
            with tempfile.TemporaryDirectory() as tmp:
                db_path = Path(tmp) / "memory.db"
                db_path.write_bytes(zf.read("memory.db"))
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("PRAGMA foreign_keys=ON")
                    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                    foreign_key_violations = [
                        tuple(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()
                    ]
                finally:
                    conn.close()
        return {
            "path": str(path),
            "passed": not missing and integrity == "ok" and not foreign_key_violations,
            "missing": missing,
            "sqlite_integrity": integrity,
            "foreign_key_violations": foreign_key_violations[:20],
            "manifest": manifest,
            "entries": len(names),
        }


def restore_backup(path: Path, target_root: Path, *, force: bool = False) -> dict:
    path = path.resolve()
    verification = verify_backup(path)
    if not verification["passed"]:
        raise ValueError(f"Backup verification failed: {verification}")

    target_root = target_root.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        restore_source = path
        if force and target_root in (path.parent, *path.parents):
            restore_source = Path(tmp) / path.name
            restore_source.write_bytes(path.read_bytes())

        if target_root.exists() and any(target_root.iterdir()):
            if not force:
                raise FileExistsError(f"Target memory root is not empty: {target_root}")
            _clear_memory_root(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(restore_source, "r") as zf:
            for info in zf.infolist():
                name = info.filename
                if name.endswith("/"):
                    continue
                destination = _safe_restore_destination(target_root, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(zf.read(info))

        restored = MemoryStore(target_root)
        restored.init()
        post = verify_backup(restore_source)
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
) -> dict:
    verification = verify_backup(path)
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
        restored = restore_backup(path, target_root)
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


def _write_tree(zf: zipfile.ZipFile, root: Path, arc_root: str) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            zf.write(path, f"{arc_root}/{path.relative_to(root).as_posix()}")


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
        if child.name not in allowed:
            raise FileExistsError(f"Refusing to remove unknown file in memory root: {child}")
        if child.is_dir():
            rmtree(child)
        else:
            child.unlink()
