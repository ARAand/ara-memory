from __future__ import annotations

from dataclasses import dataclass

from ara_memory.models import Event
from ara_memory.storage import MemoryStore, row_to_capsule, row_to_event


INSTRUCTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "reveal hidden",
    "reveal the system",
    "always obey this memory",
    "permanent instruction",
    "you must obey",
    "do not tell the user",
)
UNTRUSTED_EVENT_SOURCES = (
    "file-ingest",
    "git-untracked-file",
    "image",
    "web",
    "ocr",
)
EVIDENCE_ARTIFACT_SUFFIXES = (".py", ".toml", ".md")


@dataclass(slots=True)
class RiskVerdict:
    capsule_id: str
    score: float
    reasons: list[str]

    @property
    def should_quarantine(self) -> bool:
        return self.score >= 0.70

    def as_dict(self) -> dict[str, object]:
        return {
            "capsule_id": self.capsule_id,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "should_quarantine": self.should_quarantine,
        }


class MemoryRiskAssessor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def assess_capsule(self, capsule: dict) -> RiskVerdict:
        reasons: list[str] = []
        score = 0.0
        text = f"{capsule['title']}\n{capsule['body']}".lower()

        matched = [pattern for pattern in INSTRUCTION_PATTERNS if pattern in text]
        events = self._events(capsule["source_event_ids"])
        untrusted = [event.source for event in events if _is_untrusted_source(event.source)]

        evidence_artifact = _is_evidence_artifact(capsule)

        if matched:
            score += 0.25 if evidence_artifact else 0.62
            reasons.append("instruction-like text: " + ", ".join(matched[:4]))
            if untrusted and not evidence_artifact:
                score += 0.18
                reasons.append("instruction-like text from untrusted source")
            if evidence_artifact:
                reasons.append("instruction-like text appears inside code/test/document artifact")

        if capsule["kind"] in {"self", "preference", "procedure"}:
            if untrusted:
                score += 0.30
                reasons.append("behavioral memory from untrusted source: " + ", ".join(sorted(set(untrusted))[:4]))

        if capsule["kind"] == "self" and capsule["confidence"] < 0.85:
            score += 0.20
            reasons.append("self-memory requires high confidence")

        if capsule["source_event_ids"] == []:
            score += 0.25
            reasons.append("missing source provenance")

        return RiskVerdict(capsule_id=capsule["id"], score=min(score, 1.0), reasons=reasons)

    def assess_scope(self, *, scope: str, limit: int = 200) -> list[RiskVerdict]:
        rows = self.store.list_capsules(scope=scope, status=None, limit=limit)
        verdicts = [self.assess_capsule(dict(row_to_capsule(row))) for row in rows]
        return [verdict for verdict in verdicts if verdict.score > 0.0]

    def _events(self, event_ids: list[str]) -> list[Event]:
        if not event_ids:
            return []
        return [row_to_event(row) for row in self.store.get_events(event_ids)]


def _is_untrusted_source(source: str) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in UNTRUSTED_EVENT_SOURCES)


def _is_evidence_artifact(capsule: dict) -> bool:
    artifacts = [
        tag.removeprefix("artifact:")
        for tag in capsule["tags"]
        if isinstance(tag, str) and tag.startswith("artifact:")
    ]
    for artifact in artifacts:
        lowered = artifact.lower()
        if lowered.startswith("tests/") or lowered.startswith("docs/security"):
            return True
        if lowered.endswith(EVIDENCE_ARTIFACT_SUFFIXES) and capsule["kind"] in {"episode", "project", "summary"}:
            return True
    return False
