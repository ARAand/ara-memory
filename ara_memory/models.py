from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class EventKind(StrEnum):
    PROMPT = "prompt"
    ASSISTANT = "assistant"
    FILE = "file"
    IMAGE = "image"
    COMMAND = "command"
    DIFF = "diff"
    DECISION = "decision"
    NOTE = "note"


class CapsuleKind(StrEnum):
    EPISODE = "episode"
    DECISION = "decision"
    PREFERENCE = "preference"
    PROCEDURE = "procedure"
    FAILURE = "failure"
    PROJECT = "project"
    SELF = "self"
    FACT = "fact"
    CONFLICT = "conflict"
    SUMMARY = "summary"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    STABLE = "stable"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


@dataclass(slots=True)
class Event:
    id: str
    kind: EventKind
    text: str
    source: str
    scope: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: EventKind,
        text: str,
        source: str = "manual",
        scope: str = "global",
        metadata: dict[str, Any] | None = None,
    ) -> "Event":
        return cls(
            id=new_id("evt"),
            kind=kind,
            text=text,
            source=source,
            scope=scope,
            created_at=utc_now(),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class Capsule:
    id: str
    kind: CapsuleKind
    title: str
    body: str
    scope: str
    status: MemoryStatus
    confidence: float
    salience: float
    created_at: str
    updated_at: str
    source_event_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        kind: CapsuleKind,
        title: str,
        body: str,
        scope: str,
        confidence: float,
        salience: float,
        source_event_ids: list[str],
        tags: list[str] | None = None,
        status: MemoryStatus = MemoryStatus.CANDIDATE,
    ) -> "Capsule":
        now = utc_now()
        return cls(
            id=new_id("cap"),
            kind=kind,
            title=title,
            body=body,
            scope=scope,
            status=status,
            confidence=max(0.0, min(1.0, confidence)),
            salience=max(0.0, min(1.0, salience)),
            created_at=now,
            updated_at=now,
            source_event_ids=source_event_ids,
            tags=tags or [],
        )
