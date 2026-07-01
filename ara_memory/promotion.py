from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ara_memory.models import EventKind, MemoryStatus
from ara_memory.risk import MemoryRiskAssessor, RiskVerdict
from ara_memory.storage import MemoryStore, row_to_event


@dataclass(slots=True)
class PromotionGateResult:
    allowed: bool
    reasons: list[str]
    risk_score: float = 0.0
    should_quarantine: bool = False
    exclude_from_hot: bool = False
    source_event_count: int = 0
    source_capsule_count: int = 0
    trusted_source_event_ids: list[str] = field(default_factory=list)
    missing_source_event_ids: list[str] = field(default_factory=list)

    def reason_text(self) -> str:
        return "; ".join(self.reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasons": self.reasons,
            "risk_score": round(self.risk_score, 3),
            "should_quarantine": self.should_quarantine,
            "exclude_from_hot": self.exclude_from_hot,
            "source_event_count": self.source_event_count,
            "source_capsule_count": self.source_capsule_count,
            "trusted_source_event_ids": self.trusted_source_event_ids,
            "missing_source_event_ids": self.missing_source_event_ids,
        }


def can_promote_capsule(
    store: MemoryStore,
    capsule: dict[str, Any],
    *,
    require_provenance: bool = True,
    risk_verdict: RiskVerdict | None = None,
    source_capsule_count: int = 0,
) -> PromotionGateResult:
    reasons: list[str] = []
    status = str(capsule.get("status") or "")
    if status == MemoryStatus.QUARANTINED.value:
        reasons.append("capsule is quarantined")
    elif status in {MemoryStatus.REJECTED.value, MemoryStatus.SUPERSEDED.value}:
        reasons.append(f"capsule status is not promotable: {status}")
    elif status not in {MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value}:
        reasons.append(f"unknown capsule status: {status}")

    verdict = risk_verdict or MemoryRiskAssessor(store).assess_capsule(capsule)
    if verdict.should_quarantine:
        reasons.append("deterministic risk policy requires quarantine: " + "; ".join(verdict.reasons[:3]))
    elif verdict.should_exclude_from_hot and not _allow_evidence_summary_stable(capsule, verdict):
        reasons.append("deterministic risk policy excludes from hot memory: " + "; ".join(verdict.reasons[:3]))

    source_ids = _source_event_ids(capsule)
    trusted_ids: list[str] = []
    missing_ids: list[str] = []
    if require_provenance:
        rows = store.get_events(source_ids)
        events_by_id = {str(row["id"]): row_to_event(row) for row in rows}
        missing_ids = [event_id for event_id in source_ids if event_id not in events_by_id]
        trusted_ids = _trusted_source_event_ids(capsule, list(events_by_id.values()))
        if missing_ids:
            reasons.append(f"missing source event rows: {len(missing_ids)}")
        if len(source_ids) < 2 and source_capsule_count < 2 and not trusted_ids:
            reasons.append(
                "automatic promotion requires two source events, two consolidated source capsules, "
                "or one explicit decision source"
            )

    return PromotionGateResult(
        allowed=not reasons,
        reasons=reasons,
        risk_score=verdict.score,
        should_quarantine=verdict.should_quarantine,
        exclude_from_hot=verdict.should_exclude_from_hot,
        source_event_count=len(source_ids),
        source_capsule_count=source_capsule_count,
        trusted_source_event_ids=trusted_ids,
        missing_source_event_ids=missing_ids,
    )


def promotion_block_reason(result: PromotionGateResult) -> str:
    detail = result.reason_text() or "promotion gate blocked"
    return f"promotion gate blocked: {detail}"


def _source_event_ids(capsule: dict[str, Any]) -> list[str]:
    raw = capsule.get("source_event_ids") or []
    if not isinstance(raw, list):
        return []
    return [str(event_id) for event_id in dict.fromkeys(raw) if str(event_id)]


def _allow_evidence_summary_stable(capsule: dict[str, Any], verdict: RiskVerdict) -> bool:
    tags = {str(tag) for tag in capsule.get("tags") or []}
    if str(capsule.get("kind") or "") != "summary":
        return False
    if not {"episode-summary", "candidate-summary"}.intersection(tags):
        return False
    return (
        verdict.has_keyword_stuffing
        and not verdict.should_quarantine
        and not verdict.has_instruction_like_text
        and not verdict.has_sensitive_text
        and not verdict.has_self_serving_text
    )


def _trusted_source_event_ids(capsule: dict[str, Any], events: list[Any]) -> list[str]:
    trusted: list[str] = []
    capsule_kind = str(capsule.get("kind") or "")
    for event in events:
        if event.kind == EventKind.DECISION:
            trusted.append(event.id)
            continue
        if event.source == "manual" and event.kind in {EventKind.DECISION, EventKind.NOTE}:
            trusted.append(event.id)
            continue
        if (
            capsule_kind == "self"
            and event.source in {"manual", "codex-user"}
            and event.kind in {EventKind.PROMPT, EventKind.DECISION, EventKind.NOTE}
        ):
            trusted.append(event.id)
    return trusted
