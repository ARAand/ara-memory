from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ara_memory.models import utc_now

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError as exc:  # pragma: no cover - exercised only when packaging is broken.
    Cipher = None  # type: ignore[assignment]
    algorithms = None  # type: ignore[assignment]
    modes = None  # type: ignore[assignment]
    _CRYPTOGRAPHY_IMPORT_ERROR = exc
else:
    _CRYPTOGRAPHY_IMPORT_ERROR = None


ARCHIVE_OBJECT_KEY_NAME = ".archive-object-key"
ARCHIVE_OBJECT_MAGIC = b"ARAARCHIVE1\n"
ARCHIVE_OBJECT_ALGORITHM = "aes-256-gcm-archive-object-v1"
ARCHIVE_KEY_ESCROW_ALGORITHM = "aes-256-gcm-archive-key-escrow-v1"
GCM_NONCE_BYTES = 12
GCM_TAG_BYTES = 16


@dataclass(slots=True)
class ArchiveObjectResult:
    path: Path
    plaintext_sha256: str
    encrypted: bool
    key_id: str
    bytes: int

    def metadata(self) -> dict[str, Any]:
        return {
            "archive_encrypted": self.encrypted,
            "archive_algorithm": ARCHIVE_OBJECT_ALGORITHM if self.encrypted else "plaintext",
            "archive_key_id": self.key_id,
            "archive_plaintext_sha256": self.plaintext_sha256,
        }


def ensure_encrypted_archive_object(root: Path, source_path: Path, plaintext_sha256: str) -> ArchiveObjectResult:
    archive_path = root / "archive" / "objects" / plaintext_sha256[:2] / plaintext_sha256
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    key = _load_or_create_archive_key(root)
    if archive_path.exists():
        header = read_archive_object_header(archive_path)
        if header:
            if header.get("plaintext_sha256") != plaintext_sha256:
                raise ValueError("Archive object digest collision or corrupt encrypted header.")
            key_id = _key_id(key)
            if header.get("key_id") == key_id:
                try:
                    decrypt_archive_object(root, archive_path)
                except Exception:
                    _encrypt_file(source_path, archive_path, key=key, plaintext_sha256=plaintext_sha256)
                return ArchiveObjectResult(
                    path=archive_path,
                    plaintext_sha256=plaintext_sha256,
                    encrypted=True,
                    key_id=key_id,
                    bytes=source_path.stat().st_size,
                )
            # A restored backup may contain encrypted archive objects but not the
            # source root's local archive key. The caller still has the plaintext
            # source file, so re-wrap the object with this root's current key.
            _encrypt_file(source_path, archive_path, key=key, plaintext_sha256=plaintext_sha256)
            return ArchiveObjectResult(
                path=archive_path,
                plaintext_sha256=plaintext_sha256,
                encrypted=True,
                key_id=key_id,
                bytes=source_path.stat().st_size,
            )
        existing_digest = _sha256_file(archive_path)
        if existing_digest != plaintext_sha256:
            raise ValueError("Archive object path exists with unexpected plaintext digest.")
    _encrypt_file(source_path, archive_path, key=key, plaintext_sha256=plaintext_sha256)
    return ArchiveObjectResult(
        path=archive_path,
        plaintext_sha256=plaintext_sha256,
        encrypted=True,
        key_id=_key_id(key),
        bytes=source_path.stat().st_size,
    )


def migrate_archive_objects(
    root: Path,
    *,
    apply: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    key = _load_or_create_archive_key(root) if apply else _load_archive_key(root)
    objects_root = root / "archive" / "objects"
    objects_root.mkdir(parents=True, exist_ok=True)
    scanned = 0
    encrypted = 0
    already_encrypted = 0
    encrypted_key_mismatches = 0
    encrypted_missing_key = 0
    raw = 0
    errors: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for path in sorted(item for item in objects_root.rglob("*") if item.is_file()):
        if path.name.startswith("."):
            continue
        scanned += 1
        try:
            header = read_archive_object_header(path)
        except ValueError as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        if header:
            already_encrypted += 1
            if key is None:
                encrypted_missing_key += 1
                items.append(
                    {
                        "path": str(path),
                        "encrypted": True,
                        "missing_key": True,
                        "key_id": header.get("key_id"),
                    }
                )
            elif header.get("key_id") != _key_id(key):
                encrypted_key_mismatches += 1
                items.append(
                    {
                        "path": str(path),
                        "encrypted": True,
                        "key_mismatch": True,
                        "key_id": header.get("key_id"),
                    }
                )
            else:
                try:
                    decrypt_archive_object(root, path)
                except Exception as exc:
                    errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if limit is not None and raw >= limit:
            break
        raw += 1
        digest = _sha256_file(path)
        item = {"path": str(path), "plaintext_sha256": digest, "bytes": path.stat().st_size}
        if apply:
            if key is None:
                key = _load_or_create_archive_key(root)
            _encrypt_file(path, path, key=key, plaintext_sha256=digest)
            encrypted += 1
            item["encrypted"] = True
        items.append(item)
    return {
        "passed": not errors and encrypted_key_mismatches == 0 and encrypted_missing_key == 0,
        "applied": apply,
        "scanned": scanned,
        "raw": raw,
        "already_encrypted": already_encrypted,
        "encrypted_key_mismatches": encrypted_key_mismatches,
        "encrypted_missing_key": encrypted_missing_key,
        "encrypted": encrypted,
        "errors": errors[:20],
        "errors_truncated": max(0, len(errors) - 20),
        "items": items[:50],
        "items_truncated": max(0, len(items) - 50),
        "key_id": _key_id(key) if key else None,
    }


def read_archive_object_header(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as fh:
            if fh.read(len(ARCHIVE_OBJECT_MAGIC)) != ARCHIVE_OBJECT_MAGIC:
                return None
            size_raw = fh.read(4)
            if len(size_raw) != 4:
                raise ValueError(f"Encrypted archive object header is truncated: {path}")
            size = int.from_bytes(size_raw, "big")
            if size <= 0 or size > 4096:
                raise ValueError(f"Encrypted archive object header size is invalid: {path}")
            header_bytes = fh.read(size)
            if len(header_bytes) != size:
                raise ValueError(f"Encrypted archive object header body is truncated: {path}")
            header = json.loads(header_bytes.decode("utf-8"))
            if header.get("algorithm") != ARCHIVE_OBJECT_ALGORITHM:
                raise ValueError(f"Encrypted archive object algorithm is unsupported: {path}")
            if not isinstance(header.get("plaintext_sha256"), str) or not isinstance(header.get("key_id"), str):
                raise ValueError(f"Encrypted archive object header is missing required fields: {path}")
            return header
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Encrypted archive object header is invalid: {path}") from exc


def decrypt_archive_object(root: Path, path: Path) -> bytes:
    with path.open("rb") as fh:
        magic = fh.read(len(ARCHIVE_OBJECT_MAGIC))
        if magic != ARCHIVE_OBJECT_MAGIC:
            return path.read_bytes()
        size_raw = fh.read(4)
        if len(size_raw) != 4:
            raise ValueError("Encrypted archive object header is truncated.")
        size = int.from_bytes(size_raw, "big")
        if size <= 0 or size > 4096:
            raise ValueError("Encrypted archive object header size is invalid.")
        header_bytes = fh.read(size)
        if len(header_bytes) != size:
            raise ValueError("Encrypted archive object header body is truncated.")
        header = json.loads(header_bytes.decode("utf-8"))
        payload = fh.read()
    if len(payload) < GCM_TAG_BYTES:
        raise ValueError("Encrypted archive object is truncated.")
    if header.get("algorithm") != ARCHIVE_OBJECT_ALGORITHM:
        raise ValueError("Encrypted archive object algorithm is unsupported.")
    key = _load_archive_key(root)
    if key is None:
        raise FileNotFoundError(f"Archive object key is missing: {root / ARCHIVE_OBJECT_KEY_NAME}")
    if header.get("key_id") != _key_id(key):
        raise ValueError("Encrypted archive object was written with a different archive key.")
    return _decrypt_archive_payload(header=header, header_bytes=header_bytes, payload=payload, key=key)


def decrypt_archive_object_bytes(data: bytes, key: bytes) -> bytes:
    if data[: len(ARCHIVE_OBJECT_MAGIC)] != ARCHIVE_OBJECT_MAGIC:
        return data
    cursor = len(ARCHIVE_OBJECT_MAGIC)
    size_raw = data[cursor : cursor + 4]
    if len(size_raw) != 4:
        raise ValueError("Encrypted archive object header is truncated.")
    cursor += 4
    size = int.from_bytes(size_raw, "big")
    if size <= 0 or size > 4096:
        raise ValueError("Encrypted archive object header size is invalid.")
    header_bytes = data[cursor : cursor + size]
    if len(header_bytes) != size:
        raise ValueError("Encrypted archive object header body is truncated.")
    cursor += size
    header = json.loads(header_bytes.decode("utf-8"))
    payload = data[cursor:]
    if header.get("key_id") != _key_id(key):
        raise ValueError("Encrypted archive object was written with a different archive key.")
    return _decrypt_archive_payload(header=header, header_bytes=header_bytes, payload=payload, key=key)


def _decrypt_archive_payload(*, header: dict[str, Any], header_bytes: bytes, payload: bytes, key: bytes) -> bytes:
    if len(payload) < GCM_TAG_BYTES:
        raise ValueError("Encrypted archive object is truncated.")
    if header.get("algorithm") != ARCHIVE_OBJECT_ALGORITHM:
        raise ValueError("Encrypted archive object algorithm is unsupported.")
    if not isinstance(header.get("nonce"), str):
        raise ValueError("Encrypted archive object nonce is missing.")
    try:
        nonce = base64.b64decode(header["nonce"], validate=True)
    except binascii.Error as exc:
        raise ValueError("Encrypted archive object nonce is invalid.") from exc
    if len(nonce) != GCM_NONCE_BYTES:
        raise ValueError("Encrypted archive object nonce size is invalid.")
    ciphertext = payload[:-GCM_TAG_BYTES]
    tag = payload[-GCM_TAG_BYTES:]
    _require_crypto()
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()  # type: ignore[union-attr]
    decryptor.authenticate_additional_data(header_bytes)
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    if sha256(plaintext).hexdigest() != header.get("plaintext_sha256"):
        raise ValueError("Encrypted archive object plaintext digest mismatch.")
    return plaintext


def create_archive_key_escrow(root: Path, wrapping_key: bytes) -> dict[str, Any] | None:
    archive_key = _load_archive_key(root)
    if archive_key is None:
        return None
    _require_crypto()
    nonce = secrets.token_bytes(GCM_NONCE_BYTES)
    escrow: dict[str, Any] = {
        "algorithm": ARCHIVE_KEY_ESCROW_ALGORITHM,
        "archive_key_id": _key_id(archive_key),
        "created_at": utc_now(),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "wrapping_key_id": _key_id(wrapping_key),
    }
    aad = _escrow_aad(escrow)
    encryptor = Cipher(algorithms.AES(_derive_escrow_key(wrapping_key)), modes.GCM(nonce)).encryptor()  # type: ignore[union-attr]
    encryptor.authenticate_additional_data(aad)
    ciphertext = encryptor.update(archive_key) + encryptor.finalize()
    escrow["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")
    escrow["tag"] = base64.b64encode(encryptor.tag).decode("ascii")
    return escrow


def unwrap_archive_key_escrow(escrow: dict[str, Any], wrapping_key: bytes) -> bytes:
    _require_crypto()
    _validate_escrow_header(escrow, wrapping_key=wrapping_key)
    nonce = _b64decode_field(escrow, "nonce")
    ciphertext = _b64decode_field(escrow, "ciphertext")
    tag = _b64decode_field(escrow, "tag")
    decryptor = Cipher(algorithms.AES(_derive_escrow_key(wrapping_key)), modes.GCM(nonce, tag)).decryptor()  # type: ignore[union-attr]
    decryptor.authenticate_additional_data(_escrow_aad(escrow))
    archive_key = decryptor.update(ciphertext) + decryptor.finalize()
    if len(archive_key) != 32:
        raise ValueError("Archive key escrow plaintext has invalid size.")
    if _key_id(archive_key) != escrow.get("archive_key_id"):
        raise ValueError("Archive key escrow key id mismatch.")
    return archive_key


def restore_archive_key_from_escrow(
    root: Path,
    escrow: dict[str, Any],
    wrapping_key: bytes,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    archive_key = unwrap_archive_key_escrow(escrow, wrapping_key)
    existing = _load_archive_key(root)
    if existing is not None and _key_id(existing) != _key_id(archive_key) and not overwrite:
        raise ValueError("Refusing to overwrite a different archive object key.")
    _write_archive_key(root, archive_key)
    return {"restored": existing is None or _key_id(existing) != _key_id(archive_key), "key_id": _key_id(archive_key)}


def _encrypt_file(source: Path, destination: Path, *, key: bytes, plaintext_sha256: str) -> None:
    _require_crypto()
    nonce = secrets.token_bytes(GCM_NONCE_BYTES)
    header = {
        "algorithm": ARCHIVE_OBJECT_ALGORITHM,
        "created_at": utc_now(),
        "key_id": _key_id(key),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "plaintext_sha256": plaintext_sha256,
    }
    header_bytes = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(destination.parent)) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(ARCHIVE_OBJECT_MAGIC)
        tmp.write(len(header_bytes).to_bytes(4, "big"))
        tmp.write(header_bytes)
        encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()  # type: ignore[union-attr]
        encryptor.authenticate_additional_data(header_bytes)
        with source.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                tmp.write(encryptor.update(chunk))
        tmp.write(encryptor.finalize())
        tmp.write(encryptor.tag)
    os.replace(tmp_path, destination)


def _load_or_create_archive_key(root: Path) -> bytes:
    key = _load_archive_key(root)
    if key is not None:
        return key
    key_path = root / ARCHIVE_OBJECT_KEY_NAME
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(key_path), flags, 0o600)
    except FileExistsError:
        winner = _load_archive_key(root)
        if winner is None:
            raise
        return winner
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(key.hex())
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return key


def _write_archive_key(root: Path, key: bytes) -> None:
    if len(key) != 32:
        raise ValueError("Archive object key must be 32 bytes.")
    key_path = root / ARCHIVE_OBJECT_KEY_NAME
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key.hex(), encoding="ascii")
    try:
        key_path.chmod(0o600)
    except OSError:
        pass


def _load_archive_key(root: Path) -> bytes | None:
    key_path = root / ARCHIVE_OBJECT_KEY_NAME
    try:
        raw = key_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError("Archive object key is invalid.") from exc
    if len(key) != 32:
        raise ValueError("Archive object key must be 32 bytes.")
    return key


def _key_id(key: bytes) -> str:
    return sha256(key).hexdigest()[:16]


def _derive_escrow_key(wrapping_key: bytes) -> bytes:
    return hmac.new(wrapping_key, b"ara-memory archive key escrow v1", sha256).digest()


def _escrow_aad(escrow: dict[str, Any]) -> bytes:
    aad_payload = {
        key: escrow[key]
        for key in ("algorithm", "archive_key_id", "created_at", "nonce", "wrapping_key_id")
        if key in escrow
    }
    return json.dumps(aad_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_escrow_header(escrow: dict[str, Any], *, wrapping_key: bytes) -> None:
    if escrow.get("algorithm") != ARCHIVE_KEY_ESCROW_ALGORITHM:
        raise ValueError("Archive key escrow algorithm is unsupported.")
    for field in ("archive_key_id", "created_at", "nonce", "wrapping_key_id", "ciphertext", "tag"):
        if not isinstance(escrow.get(field), str) or not escrow[field]:
            raise ValueError(f"Archive key escrow field is invalid: {field}")
    if escrow.get("wrapping_key_id") != _key_id(wrapping_key):
        raise ValueError("Archive key escrow wrapping key id mismatch.")
    nonce = _b64decode_field(escrow, "nonce")
    tag = _b64decode_field(escrow, "tag")
    if len(nonce) != GCM_NONCE_BYTES:
        raise ValueError("Archive key escrow nonce size is invalid.")
    if len(tag) != GCM_TAG_BYTES:
        raise ValueError("Archive key escrow tag size is invalid.")


def _b64decode_field(payload: dict[str, Any], field: str) -> bytes:
    try:
        return base64.b64decode(payload[field], validate=True)
    except (binascii.Error, KeyError, TypeError) as exc:
        raise ValueError(f"Base64 field is invalid: {field}") from exc


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_crypto() -> None:
    if _CRYPTOGRAPHY_IMPORT_ERROR is not None:
        raise RuntimeError("cryptography is required for archive object encryption.") from _CRYPTOGRAPHY_IMPORT_ERROR
