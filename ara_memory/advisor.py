from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Protocol

from ara_memory.promotion import can_promote_capsule, promotion_block_reason
from ara_memory.risk import MemoryRiskAssessor, redact_sensitive_text
from ara_memory.storage import MemoryStore, row_to_capsule


PROMOTE_KINDS = {"self", "goal", "procedure", "decision", "failure", "project", "fact"}


@dataclass(slots=True)
class MemoryRecommendation:
    capsule_id: str
    action: str
    actor: str
    reason: str
    risk_score: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "capsule_id": self.capsule_id,
            "action": self.action,
            "actor": self.actor,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 3),
        }


class MemoryAdvisor(Protocol):
    def review(self, candidates: list[dict]) -> list[MemoryRecommendation]:
        ...


class DeterministicMemoryAdvisor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.risk = MemoryRiskAssessor(store)

    def review(self, candidates: list[dict]) -> list[MemoryRecommendation]:
        recommendations: list[MemoryRecommendation] = []
        for cap in candidates:
            verdict = self.risk.assess_capsule(cap)
            if verdict.should_quarantine:
                recommendations.append(
                    MemoryRecommendation(
                        capsule_id=cap["id"],
                        action="quarantine",
                        actor="memory-auditor",
                        reason="; ".join(verdict.reasons),
                        risk_score=verdict.score,
                    )
                )
                continue
            if verdict.should_exclude_from_hot:
                recommendations.append(
                    MemoryRecommendation(
                        capsule_id=cap["id"],
                        action="keep",
                        actor="memory-auditor",
                        reason="excluded from hot memory by deterministic risk policy: " + "; ".join(verdict.reasons),
                        risk_score=verdict.score,
                    )
                )
                continue
            if _should_promote(cap):
                gate = can_promote_capsule(self.store, cap, require_provenance=True, risk_verdict=verdict)
                if not gate.allowed:
                    recommendations.append(
                        MemoryRecommendation(
                            capsule_id=cap["id"],
                            action="keep",
                            actor="memory-auditor",
                            reason=promotion_block_reason(gate),
                            risk_score=verdict.score,
                        )
                    )
                    continue
                recommendations.append(
                    MemoryRecommendation(
                        capsule_id=cap["id"],
                        action="promote",
                        actor="memory-curator",
                        reason="high salience/confidence candidate",
                        risk_score=verdict.score,
                    )
                )
                continue
            recommendations.append(
                MemoryRecommendation(
                    capsule_id=cap["id"],
                    action="keep",
                    actor="memory-curator",
                    reason="insufficient confidence, salience, or kind for automatic promotion",
                    risk_score=verdict.score,
                )
            )
        return recommendations


class ExternalCommandMemoryAdvisor:
    def __init__(
        self,
        *,
        command: str,
        store: MemoryStore,
        fallback: MemoryAdvisor,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.command = command
        self.store = store
        self.fallback = fallback
        self.timeout_seconds = timeout_seconds

    def review(self, candidates: list[dict]) -> list[MemoryRecommendation]:
        fallback_recs = {rec.capsule_id: rec for rec in self.fallback.review(candidates)}
        if not candidates:
            return []
        payload = {
            "version": 1,
            "task": "memory_review",
            "allowed_actions": ["promote", "quarantine", "keep"],
            "candidates": [_candidate_payload(cap, fallback_recs[cap["id"]]) for cap in candidates],
        }
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                check=True,
                shell=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            parsed = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return list(fallback_recs.values())

        recommendations = _parse_external_recommendations(
            parsed,
            fallback_recs,
            store=self.store,
            candidates={cap["id"]: cap for cap in candidates},
        )
        ordered_ids = [cap["id"] for cap in candidates]
        return [recommendations[capsule_id] for capsule_id in ordered_ids]


class MemoryReviewEngine:
    def __init__(self, store: MemoryStore, advisor: MemoryAdvisor | None = None) -> None:
        self.store = store
        self.advisor = advisor or DeterministicMemoryAdvisor(store)

    def review_scope(self, *, scope: str, limit: int = 500) -> list[MemoryRecommendation]:
        from ara_memory.models import MemoryStatus

        rows = self.store.list_capsules(scope=scope, status=MemoryStatus.CANDIDATE, limit=limit)
        return self.advisor.review([row_to_capsule(row) for row in rows])


def build_memory_advisor(store: MemoryStore) -> MemoryAdvisor:
    deterministic = DeterministicMemoryAdvisor(store)
    provider = os.environ.get("ARA_MEMORY_ADVISOR", "deterministic").strip().lower()
    if provider in {"", "deterministic", "local"}:
        return deterministic
    if provider in {"external", "external-command", "command"}:
        command = os.environ.get("ARA_MEMORY_ADVISOR_COMMAND", "").strip()
        if not command:
            return deterministic
        timeout_ms = _env_int("ARA_MEMORY_ADVISOR_TIMEOUT_MS", default=20000)
        return ExternalCommandMemoryAdvisor(
            command=command,
            store=store,
            fallback=deterministic,
            timeout_seconds=max(1.0, timeout_ms / 1000),
        )
    return deterministic


def _should_promote(cap: dict) -> bool:
    if cap["kind"] not in PROMOTE_KINDS:
        return False
    if cap["kind"] == "preference":
        return False
    return cap["confidence"] >= 0.70 and cap["salience"] >= 0.70


def _candidate_payload(cap: dict, fallback: MemoryRecommendation | None = None) -> dict[str, object]:
    redacted = bool(fallback and _is_deterministically_blocked(fallback))
    title = cap["title"]
    body = cap["body"][:1600]
    tags = cap["tags"][:24]
    source_event_ids = cap["source_event_ids"][:24]
    if redacted:
        title = redact_sensitive_text(str(title))
        body = "[redacted by deterministic memory auditor]"
        tags = [redact_sensitive_text(str(tag)) for tag in tags]
        source_event_ids = []
    return {
        "id": cap["id"],
        "kind": cap["kind"],
        "title": title,
        "body": body,
        "scope": cap["scope"],
        "confidence": cap["confidence"],
        "salience": cap["salience"],
        "tags": tags,
        "source_event_ids": source_event_ids,
        "risk_redacted": redacted,
    }


def _parse_external_recommendations(
    parsed: object,
    fallback_recs: dict[str, MemoryRecommendation],
    *,
    store: MemoryStore | None = None,
    candidates: dict[str, dict] | None = None,
) -> dict[str, MemoryRecommendation]:
    if not isinstance(parsed, dict):
        return fallback_recs
    raw_recs = parsed.get("recommendations")
    if not isinstance(raw_recs, list):
        return fallback_recs

    out = dict(fallback_recs)
    for raw in raw_recs:
        if not isinstance(raw, dict):
            continue
        capsule_id = raw.get("capsule_id")
        if not isinstance(capsule_id, str) or capsule_id not in fallback_recs:
            continue
        action = raw.get("action")
        if action not in {"promote", "quarantine", "keep"}:
            continue
        fallback = fallback_recs[capsule_id]
        if fallback.action == "quarantine" and action != "quarantine":
            continue
        if action == "promote" and _is_deterministically_blocked(fallback):
            continue
        if action == "promote" and store is not None and candidates is not None:
            gate = can_promote_capsule(store, candidates[capsule_id], require_provenance=True)
            if not gate.allowed:
                continue
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = f"external advisor recommended {action}"
        risk_score = raw.get("risk_score")
        if not isinstance(risk_score, (int, float)):
            risk_score = fallback.risk_score
        out[capsule_id] = MemoryRecommendation(
            capsule_id=capsule_id,
            action=action,
            actor="external-memory-advisor",
            reason=reason[:800],
            risk_score=max(0.0, min(1.0, float(risk_score))),
        )
    return out


def _is_deterministically_blocked(fallback: MemoryRecommendation) -> bool:
    return (
        fallback.action == "quarantine"
        or fallback.reason.startswith("excluded from hot memory by deterministic risk policy")
        or fallback.reason.startswith("promotion gate blocked")
    )


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
