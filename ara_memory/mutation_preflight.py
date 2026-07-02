from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ara_memory.models import CapsuleKind, MemoryStatus
from ara_memory.risk import MemoryRiskAssessor
from ara_memory.storage import MemoryStore, row_to_capsule


MUTATION_ACTIONS = {"rewrite", "delete"}


@dataclass(slots=True)
class MutationPreflightReport:
    capsule_id: str
    scope: str | None
    action: str
    passed: bool
    status: str
    warnings: list[str]
    blocked_reasons: list[str]
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    rollback: dict[str, Any] | None
    mutation_digest: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "scope": self.scope,
            "action": self.action,
            "passed": self.passed,
            "status": self.status,
            "warnings": self.warnings,
            "blocked_reasons": self.blocked_reasons,
            "before": self.before,
            "after": self.after,
            "rollback": self.rollback,
            "mutation_digest": self.mutation_digest,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Mutation Preflight: {self.action}",
            f"capsule_id={self.capsule_id}",
            f"status={self.status}, passed={self.passed}",
        ]
        if self.blocked_reasons:
            lines.append("## Blocked")
            lines.extend(f"- {item}" for item in self.blocked_reasons)
        if self.warnings:
            lines.append("## Warnings")
            lines.extend(f"- {item}" for item in self.warnings)
        if self.mutation_digest:
            lines.append(f"mutation_digest={self.mutation_digest}")
        return "\n".join(lines)


class MutationPreflight:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.risk = MemoryRiskAssessor(store)

    def run(
        self,
        *,
        capsule_id: str,
        action: str,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> MutationPreflightReport:
        self.store.init()
        action = str(action)
        if action not in MUTATION_ACTIONS:
            return _blocked(capsule_id, None, action, [f"unsupported mutation action: {action}"])
        row = self.store.get_capsule(capsule_id)
        if row is None:
            return _blocked(capsule_id, None, action, ["capsule not found"])
        capsule = row_to_capsule(row)
        before = _snapshot(capsule)
        warnings: list[str] = []
        blocked: list[str] = []
        if capsule["kind"] in {CapsuleKind.SELF.value, CapsuleKind.GOAL.value}:
            warnings.append("core identity/goal memory requires explicit higher-level approval")

        if action == "rewrite":
            after = _rewrite_after(capsule, title=title, body=body, tags=tags)
            if _projection_digest(before) == _projection_digest(after):
                blocked.append("rewrite preflight has no projection changes")
            risk = self.risk.assess_capsule(_capsule_from_snapshot(after))
            if risk.should_quarantine:
                blocked.append("rewrite projection would trigger deterministic quarantine risk")
            elif risk.has_sensitive_text:
                warnings.append("rewrite projection still contains sensitive-looking text")
        else:
            after = dict(before)
            after["status"] = MemoryStatus.REJECTED.value
            after["mutation_policy"] = "soft-delete only; physical deletion is not authorized by this preflight"
            if capsule["status"] == MemoryStatus.REJECTED.value:
                blocked.append("capsule is already rejected")
            if capsule["status"] == MemoryStatus.STABLE.value:
                blocked.append("stable memory delete requires a future shadow-delete approval, not this preflight")
            warnings.append("delete preflight maps to reversible rejected status; source events are preserved")

        rollback = {
            "target_capsule_id": capsule_id,
            "restore": before,
            "requires_current_digest": _snapshot_digest(after),
            "policy": "rollback may restore only this capsule projection/status if the live capsule still matches after digest",
        }
        mutation_digest = _json_digest({"action": action, "before": before, "after": after, "rollback": rollback})
        status = "fail" if blocked else ("watch" if warnings else "pass")
        return MutationPreflightReport(
            capsule_id=capsule_id,
            scope=capsule["scope"],
            action=action,
            passed=not blocked,
            status=status,
            warnings=warnings,
            blocked_reasons=blocked,
            before=before,
            after=after,
            rollback=rollback,
            mutation_digest=mutation_digest,
        )


def _rewrite_after(
    capsule: dict[str, Any],
    *,
    title: str | None,
    body: str | None,
    tags: list[str] | None,
) -> dict[str, Any]:
    after = _snapshot(capsule)
    if title is not None:
        after["title"] = title
    if body is not None:
        after["body"] = body
    if tags is not None:
        after["tags"] = list(tags)
    return after


def _snapshot(capsule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": capsule["id"],
        "kind": capsule["kind"],
        "title": capsule["title"],
        "body": capsule["body"],
        "scope": capsule["scope"],
        "status": capsule["status"],
        "confidence": capsule["confidence"],
        "salience": capsule["salience"],
        "source_event_ids": list(capsule["source_event_ids"]),
        "tags": list(capsule["tags"]),
        "projection_digest": _projection_digest(capsule),
    }


def _capsule_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": snapshot["id"],
        "kind": snapshot["kind"],
        "title": snapshot["title"],
        "body": snapshot["body"],
        "scope": snapshot["scope"],
        "status": snapshot["status"],
        "confidence": snapshot["confidence"],
        "salience": snapshot["salience"],
        "source_event_ids": list(snapshot["source_event_ids"]),
        "tags": list(snapshot["tags"]),
    }


def _projection_digest(capsule: dict[str, Any]) -> str:
    return _json_digest(
        {
            "title": capsule["title"],
            "body": capsule["body"],
            "tags": list(capsule["tags"]),
            "status": capsule["status"],
        }
    )


def _snapshot_digest(snapshot: dict[str, Any]) -> str:
    return _json_digest({key: value for key, value in snapshot.items() if key != "projection_digest"})


def mutation_plan_digest(payload: dict[str, Any]) -> str:
    return _json_digest(payload)


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    return _snapshot_digest(snapshot)


def _json_digest(payload: dict[str, Any]) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _blocked(
    capsule_id: str,
    scope: str | None,
    action: str,
    reasons: list[str],
) -> MutationPreflightReport:
    return MutationPreflightReport(
        capsule_id=capsule_id,
        scope=scope,
        action=action,
        passed=False,
        status="fail",
        warnings=[],
        blocked_reasons=reasons,
        before=None,
        after=None,
        rollback=None,
        mutation_digest=None,
    )
