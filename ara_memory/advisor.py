from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Protocol

from ara_memory.risk import MemoryRiskAssessor
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
            if _should_promote(cap):
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
        fallback: MemoryAdvisor,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.command = command
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
            "candidates": [_candidate_payload(cap) for cap in candidates],
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

        recommendations = _parse_external_recommendations(parsed, fallback_recs)
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


def _candidate_payload(cap: dict) -> dict[str, object]:
    return {
        "id": cap["id"],
        "kind": cap["kind"],
        "title": cap["title"],
        "body": cap["body"][:1600],
        "scope": cap["scope"],
        "confidence": cap["confidence"],
        "salience": cap["salience"],
        "tags": cap["tags"][:24],
        "source_event_ids": cap["source_event_ids"][:24],
    }


def _parse_external_recommendations(
    parsed: object,
    fallback_recs: dict[str, MemoryRecommendation],
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


def _env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
