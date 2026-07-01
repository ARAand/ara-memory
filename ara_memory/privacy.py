from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ara_memory.risk import direct_identifier_matches, redact_sensitive_text, sensitive_text_matches


HIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "cot",
    "hidden_reasoning",
    "internal_reasoning",
    "reasoning",
    "reasoning_summary",
    "scratchpad",
}

PATH_VALUE_KEYS = {
    "archive_path",
    "capture_cwd",
    "cwd",
    "file",
    "original_path",
    "path",
    "spool_original_path",
    "spool_snapshot_path",
}
MAX_METADATA_DEPTH = 64

HIDDEN_REASONING_LINE = re.compile(
    r"(?im)^(\s*[\"']?(?:chain[-_\s]?of[-_\s]?thought|cot|hidden[-_\s]?reasoning|"
    r"internal[-_\s]?reasoning|reasoning[-_\s]?summary|scratchpad|reasoning)[\"']?\s*[:=]\s*).*$"
)


@dataclass(slots=True)
class PrivacyGuardResult:
    text: str
    metadata: dict[str, Any]
    reasons: list[str]
    text_redacted: bool
    metadata_redacted: bool
    source: str | None = None
    scope: str | None = None
    source_redacted: bool = False
    scope_redacted: bool = False
    allow_raw_private: bool = False

    @property
    def redacted(self) -> bool:
        return self.text_redacted or self.metadata_redacted or self.source_redacted or self.scope_redacted

    def metadata_note(self) -> dict[str, Any]:
        action = "allowed_raw_private" if self.allow_raw_private and self.reasons else "redacted" if self.redacted else "none"
        return {
            "action": action,
            "reasons": self.reasons,
            "text_redacted": self.text_redacted,
            "metadata_redacted": self.metadata_redacted,
            "source_redacted": self.source_redacted,
            "scope_redacted": self.scope_redacted,
        }


@dataclass(slots=True)
class TurnPrivacyGuardResult:
    value: dict[str, Any]
    reasons: list[str]
    redacted: bool
    allow_raw_private: bool = False

    def metadata_note(self) -> dict[str, Any]:
        action = "allowed_raw_private" if self.allow_raw_private and self.reasons else "redacted" if self.redacted else "none"
        return {
            "action": action,
            "reasons": self.reasons,
            "turn_redacted": self.redacted,
        }


def guard_event_payload(
    *,
    text: str,
    metadata: dict[str, Any] | None = None,
    source: str | None = None,
    scope: str | None = None,
    allow_raw_private: bool = False,
) -> PrivacyGuardResult:
    metadata = dict(metadata or {})
    text_reasons = _text_privacy_reasons(text)
    metadata_result = _guard_metadata(metadata, allow_raw_private=allow_raw_private)
    safe_source, source_reasons, source_redacted = _guard_label(
        source,
        label="source",
        allow_raw_private=allow_raw_private,
    )
    safe_scope, scope_reasons, scope_redacted = _guard_label(
        scope,
        label="scope",
        allow_raw_private=allow_raw_private,
    )
    reasons = _dedupe([*text_reasons, *metadata_result.reasons, *source_reasons, *scope_reasons])
    if allow_raw_private:
        return PrivacyGuardResult(
            text=text,
            metadata=metadata,
            reasons=reasons,
            text_redacted=False,
            metadata_redacted=False,
            source=source,
            scope=scope,
            allow_raw_private=True,
        )
    redacted_text = redact_private_text(text)
    text_redacted = redacted_text != text
    return PrivacyGuardResult(
        text=redacted_text,
        metadata=metadata_result.value,
        reasons=reasons,
        text_redacted=text_redacted,
        metadata_redacted=metadata_result.redacted,
        source=safe_source,
        scope=safe_scope,
        source_redacted=source_redacted,
        scope_redacted=scope_redacted,
    )


def guard_turn_payload(
    turn: dict[str, Any],
    *,
    allow_raw_private: bool = False,
) -> TurnPrivacyGuardResult:
    value, reasons, redacted = _guard_metadata_value(
        dict(turn),
        key_path="turn",
        allow_raw_private=allow_raw_private,
        preserve_path_values=False,
        depth=0,
        seen=set(),
    )
    if not isinstance(value, dict):
        return TurnPrivacyGuardResult({}, reasons, True, allow_raw_private=allow_raw_private)
    return TurnPrivacyGuardResult(value, reasons, redacted, allow_raw_private=allow_raw_private)


def privacy_safe_label(value: str, *, label: str, allow_raw_private: bool = False) -> str:
    safe_value, _, _ = _guard_label(value, label=label, allow_raw_private=allow_raw_private)
    return safe_value or value


def redact_private_text(text: str) -> str:
    redacted = redact_sensitive_text(text)
    return HIDDEN_REASONING_LINE.sub(lambda match: f"{match.group(1)}[redacted hidden reasoning]", redacted)


@dataclass(slots=True)
class _MetadataResult:
    value: dict[str, Any]
    reasons: list[str]
    redacted: bool


def _guard_metadata(metadata: dict[str, Any], *, allow_raw_private: bool) -> _MetadataResult:
    value, reasons, redacted = _guard_metadata_value(
        metadata,
        key_path="metadata",
        allow_raw_private=allow_raw_private,
        preserve_path_values=False,
        depth=0,
        seen=set(),
    )
    if not isinstance(value, dict):
        return _MetadataResult({}, reasons, True)
    return _MetadataResult(value, reasons, redacted)


def _guard_metadata_value(
    value: Any,
    *,
    key_path: str,
    allow_raw_private: bool,
    preserve_path_values: bool,
    depth: int,
    seen: set[int],
) -> tuple[Any, list[str], bool]:
    if depth > MAX_METADATA_DEPTH:
        return (
            "[redacted deeply nested metadata]",
            [f"metadata nesting exceeded: {_safe_reason_path(key_path)}"],
            True,
        )
    key_name = _last_key_name(key_path)
    if key_name in HIDDEN_REASONING_KEYS:
        reason_path = _safe_reason_path(key_path)
        return "[redacted hidden reasoning]", [f"hidden reasoning metadata: {reason_path}"], not allow_raw_private

    if isinstance(value, dict):
        object_id = id(value)
        if object_id in seen:
            return (
                "[redacted cyclic metadata]",
                [f"cyclic metadata: {_safe_reason_path(key_path)}"],
                True,
            )
        seen.add(object_id)
        out: dict[str, Any] = {}
        reasons: list[str] = []
        redacted = False
        for index, (key, item) in enumerate(value.items()):
            key_text = str(key)
            safe_key, key_reasons, key_redacted = _guard_metadata_key(
                key_text,
                index=index,
                allow_raw_private=allow_raw_private,
            )
            child_key = f"{key_path}.{safe_key}" if key_path else safe_key
            guarded, child_reasons, child_redacted = _guard_metadata_value(
                item,
                key_path=child_key,
                allow_raw_private=allow_raw_private,
                preserve_path_values=preserve_path_values,
                depth=depth + 1,
                seen=seen,
            )
            out[key_text if allow_raw_private else safe_key] = item if allow_raw_private and not child_redacted else guarded
            reasons.extend(key_reasons)
            reasons.extend(child_reasons)
            redacted = redacted or key_redacted or child_redacted
        seen.remove(object_id)
        return out, _dedupe(reasons), redacted

    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in seen:
            return (
                "[redacted cyclic metadata]",
                [f"cyclic metadata: {_safe_reason_path(key_path)}"],
                True,
            )
        seen.add(object_id)
        out = []
        reasons: list[str] = []
        redacted = False
        for index, item in enumerate(value):
            guarded, child_reasons, child_redacted = _guard_metadata_value(
                item,
                key_path=f"{key_path}[{index}]",
                allow_raw_private=allow_raw_private,
                preserve_path_values=preserve_path_values,
                depth=depth + 1,
                seen=seen,
            )
            out.append(item if allow_raw_private and not child_redacted else guarded)
            reasons.extend(child_reasons)
            redacted = redacted or child_redacted
        seen.remove(object_id)
        return out, _dedupe(reasons), redacted

    if isinstance(value, str):
        reason_path = _safe_reason_path(key_path)
        reasons = [f"metadata {reason}: {reason_path}" for reason in _text_privacy_reasons(value)]
        if preserve_path_values and key_name in PATH_VALUE_KEYS:
            return value, reasons, False
        redacted = redact_private_text(value)
        return (value if allow_raw_private else redacted), reasons, redacted != value and not allow_raw_private

    return value, [], False


def _guard_metadata_key(key: str, *, index: int, allow_raw_private: bool) -> tuple[str, list[str], bool]:
    reasons = [f"metadata key {reason}: key[{index}]" for reason in _text_privacy_reasons(key)]
    if allow_raw_private or not reasons:
        return key, reasons, False
    digest = sha256(key.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"redacted_key_{index}_{digest}", reasons, True


def _guard_label(
    value: str | None,
    *,
    label: str,
    allow_raw_private: bool,
) -> tuple[str | None, list[str], bool]:
    if value is None:
        return None, [], False
    reasons = [f"{label} {reason}" for reason in _text_privacy_reasons(value)]
    if allow_raw_private or not reasons:
        return value, reasons, False
    digest = sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"private-{label}-{digest}", reasons, True


def _text_privacy_reasons(text: str) -> list[str]:
    reasons = [f"sensitive data pattern: {reason}" for reason in sensitive_text_matches(text)]
    reasons.extend(f"direct identifier: {reason}" for reason in direct_identifier_matches(text))
    if HIDDEN_REASONING_LINE.search(text):
        reasons.append("hidden reasoning text")
    return _dedupe(reasons)


def _last_key_name(key_path: str) -> str:
    key_name = key_path.rsplit(".", 1)[-1]
    if "[" in key_name:
        key_name = key_name.split("[", 1)[0]
    return key_name.lower()


def _safe_reason_path(key_path: str) -> str:
    parts = []
    for part in key_path.split("."):
        reasons = _text_privacy_reasons(part)
        parts.append("[redacted-key]" if reasons else part)
    return ".".join(parts)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
