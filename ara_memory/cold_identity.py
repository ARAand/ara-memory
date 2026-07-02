from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Iterable


COLD_IDENTITY_VERSION = 1
COLD_IDENTITY_SETS = (
    "cold_capsules",
    "cold_source_events",
    "protected_source_events",
    "prunable_source_events",
)


def build_cold_identity(
    *,
    scope: str | None,
    cold_capsule_ids: Iterable[str],
    cold_source_event_ids: Iterable[str],
    protected_event_ids: Iterable[str],
    prunable_event_ids: Iterable[str],
) -> dict[str, Any]:
    sections = {
        "cold_capsules": _section(cold_capsule_ids),
        "cold_source_events": _section(cold_source_event_ids),
        "protected_source_events": _section(protected_event_ids),
        "prunable_source_events": _section(prunable_event_ids),
    }
    payload: dict[str, Any] = {
        "version": COLD_IDENTITY_VERSION,
        "scope": scope,
        **sections,
    }
    payload["combined_sha256"] = _payload_digest(
        {
            "version": COLD_IDENTITY_VERSION,
            "scope": scope,
            "sets": sections,
        }
    )
    return payload


def compare_cold_identities(current: dict[str, Any] | None, cycle: dict[str, Any] | None) -> dict[str, Any]:
    if not current:
        return {"available": False, "matches": False, "reason": "missing_current_identity", "mismatched_sets": []}
    if not cycle:
        return {"available": False, "matches": False, "reason": "missing_cycle_identity", "mismatched_sets": []}
    current_version = current.get("version")
    cycle_version = cycle.get("version")
    if current_version != cycle_version:
        return {
            "available": True,
            "matches": False,
            "reason": "version_mismatch",
            "current_version": current_version,
            "cycle_version": cycle_version,
            "mismatched_sets": [],
        }
    mismatched_sets = [
        name
        for name in COLD_IDENTITY_SETS
        if _section_signature(current.get(name)) != _section_signature(cycle.get(name))
    ]
    current_digest = current.get("combined_sha256")
    cycle_digest = cycle.get("combined_sha256")
    digest_matches = bool(current_digest) and current_digest == cycle_digest
    matches = not mismatched_sets and digest_matches
    return {
        "available": True,
        "matches": matches,
        "reason": "match" if matches else "fingerprint_mismatch",
        "mismatched_sets": mismatched_sets,
        "current_sha256": current_digest,
        "cycle_sha256": cycle_digest,
    }


def _section(ids: Iterable[str]) -> dict[str, Any]:
    normalized = _unique_sorted(ids)
    return {
        "count": len(normalized),
        "sha256": _ids_digest(normalized),
    }


def _section_signature(section: Any) -> tuple[int | None, str | None]:
    if not isinstance(section, dict):
        return None, None
    count = section.get("count")
    try:
        normalized_count = int(count)
    except (TypeError, ValueError):
        normalized_count = None
    sha = section.get("sha256")
    return normalized_count, str(sha) if sha else None


def _unique_sorted(ids: Iterable[str]) -> list[str]:
    return sorted({str(item).strip() for item in ids if str(item).strip()})


def _ids_digest(ids: list[str]) -> str:
    digest = sha256()
    for item in ids:
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()
