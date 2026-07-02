from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import secrets
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens
from ara_memory.models import Capsule, CapsuleKind, Event, EventKind, MemoryStatus, new_id, utc_now
from ara_memory.storage import _invalidate_hot_scope, _sync_capsule_fts_row, row_to_capsule


RECONSOLIDATION_APPLY_CONFIRMATION = "APPLY RECONSOLIDATION FRAME"
RECONSOLIDATION_LIVE_ROLLBACK_CONFIRMATION = "ROLLBACK RECONSOLIDATION CANDIDATE"
STRONG_RECONSOLIDATION_ACTIONS = ("promote", "rewrite", "delete", "cool")


@dataclass(slots=True)
class ReconsolidationEvidence:
    capsule_id: str | None
    kind: str
    status: str
    title: str
    text: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "kind": self.kind,
            "status": self.status,
            "title": self.title,
            "text": self.text,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ReconsolidationFrame:
    name: str
    stance: str
    thesis: str
    evidence: list[ReconsolidationEvidence]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stance": self.stance,
            "thesis": self.thesis,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class ReconsolidationReport:
    scope: str
    query: str
    status: str
    frames: list[ReconsolidationFrame]
    diagnostics: dict[str, Any]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "status": self.status,
            "frames": [frame.as_dict() for frame in self.frames],
            "diagnostics": self.diagnostics,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Frame: {self.scope}",
            f"status: {self.status}",
            f"query: {compact_text(self.query, limit=260)}",
            "## Frames",
        ]
        for frame in self.frames:
            lines.append(f"- [{frame.stance}] {frame.name}: {frame.thesis}")
            for item in frame.evidence[:4]:
                source = f"source {item.capsule_id}" if item.capsule_id else "source synthetic"
                lines.append(
                    f"  evidence: [{item.kind}/{item.status}] "
                    f"{compact_text(item.text, limit=240)} ({source}; {item.reason})"
                )
        lines.append("## Diagnostics")
        for key, value in self.diagnostics.items():
            lines.append(f"- {key}: {value}")
        lines.append("## Recommendations")
        lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationApproval:
    approval_id: str | None
    token: str | None
    expires_at: str | None
    frame: ReconsolidationReport
    frame_fingerprint: str
    rollback_witness: dict[str, Any]

    @property
    def prepared(self) -> bool:
        return self.approval_id is not None and self.token is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "prepared": self.prepared,
            "token": self.token,
            "expires_at": self.expires_at,
            "scope": self.frame.scope,
            "status": self.frame.status,
            "frame_fingerprint": self.frame_fingerprint,
            "rollback_witness": self.rollback_witness,
            "frame": self.frame.as_dict(),
            "safety": [
                "Prepare is read-only and stores a short-lived approval token.",
                "Prepare stores a rollback witness preview for evidence identities, source links, and content digests.",
                "Apply must rebuild the same frame and match the fingerprint.",
                "Apply creates one candidate summary capsule and witness; it does not promote, supersede, rewrite, delete, or cool existing capsules.",
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Prepare: {self.frame.scope}",
            f"status: {self.frame.status}",
            f"frame_fingerprint: {self.frame_fingerprint}",
            f"rollback_witness: {self.rollback_witness.get('evidence_capsule_count', 0)} evidence capsules",
        ]
        if self.prepared:
            lines.extend(
                [
                    f"approval_id: {self.approval_id}",
                    f"expires_at: {self.expires_at}",
                    f"approval_token: {self.token}",
                ]
            )
        else:
            lines.append("approval_id: none")
        lines.extend(
            [
                "## Safety",
                "- Prepare only: no memory is mutated.",
                "- Rollback witness preview freezes the evidence identities, source links, and digests needed by future stronger gates.",
                "- Apply is candidate-only and must consume this token once.",
                "- Review the witness before any stronger reconsolidation action exists.",
            ]
        )
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationApplyReport:
    scope: str | None
    passed: bool
    approval_id: str | None
    capsule_id: str | None
    witness_id: str | None
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "approval_id": self.approval_id,
            "capsule_id": self.capsule_id,
            "witness_id": self.witness_id,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Apply: {self.scope or 'unknown'}",
            f"status: {'pass' if self.passed else 'blocked'}",
            f"approval_id: {self.approval_id or 'none'}",
            f"capsule_id: {self.capsule_id or 'none'}",
            f"witness_id: {self.witness_id or 'none'}",
        ]
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationReviewItem:
    witness_id: str
    approval_id: str
    scope: str
    capsule_id: str
    status: str
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "approval_id": self.approval_id,
            "scope": self.scope,
            "capsule_id": self.capsule_id,
            "status": self.status,
            "passed": self.passed,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class ReconsolidationReviewReport:
    scope: str | None
    approval_id: str | None
    passed: bool
    reviewed: int
    pass_count: int
    watch_count: int
    fail_count: int
    items: list[ReconsolidationReviewItem]
    recommendations: list[str]
    regression: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "approval_id": self.approval_id,
            "passed": self.passed,
            "reviewed": self.reviewed,
            "pass_count": self.pass_count,
            "watch_count": self.watch_count,
            "fail_count": self.fail_count,
            "items": [item.as_dict() for item in self.items],
            "recommendations": self.recommendations,
            "regression": self.regression,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Reconsolidation Review: {scope}",
            f"status: {'pass' if self.passed else 'watch'}",
            f"reviewed={self.reviewed}, pass={self.pass_count}, watch={self.watch_count}, fail={self.fail_count}",
        ]
        if not self.items:
            lines.append("- No reconsolidation witnesses matched the review filter.")
        for item in self.items[:20]:
            warning = "; ".join(item.warnings) if item.warnings else "witness invariants hold"
            lines.append(f"- [{item.status}] {item.witness_id} capsule={item.capsule_id}: {warning}")
        if self.regression is not None:
            cases = self.regression.get("cases", [])
            baseline = self.regression.get("baseline_comparison", [])
            lines.extend(
                [
                    "## Recall Regression Sandbox",
                    f"status: {'pass' if self.regression.get('passed') else 'fail'}",
                    f"cases: {_passed_count(cases)}",
                    f"baseline: {_passed_count(baseline)}",
                ]
            )
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationReviewQueueReport:
    scope: str | None
    recorded: int
    resolved: int
    open_items: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "recorded": self.recorded,
            "resolved": self.resolved,
            "open_items": self.open_items,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Reconsolidation Review Queue: {scope}",
            f"recorded={self.recorded}, resolved={self.resolved}, open={len(self.open_items)}",
        ]
        if not self.open_items:
            lines.append("- No open reconsolidation review items.")
        for item in self.open_items[:20]:
            lines.append(
                f"- [{item['review_status']}] {item['action']} "
                f"witness={item.get('witness_id') or 'none'} capsule={item.get('capsule_id') or 'none'}: "
                f"{item['reason']}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationRollbackWitnessBackfillReport:
    scope: str | None
    dry_run: bool
    reviewed: int
    changed: int
    skipped: int
    items: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "reviewed": self.reviewed,
            "changed": self.changed,
            "skipped": self.skipped,
            "items": self.items,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Reconsolidation Rollback Witness Backfill: {scope}",
            f"dry_run={self.dry_run}, reviewed={self.reviewed}, changed={self.changed}, skipped={self.skipped}",
        ]
        for item in self.items[:20]:
            lines.append(
                f"- [{item['status']}] witness={item['witness_id']} approval={item['approval_id']}: "
                f"{item['reason']}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationStrongPreflightReport:
    scope: str
    query: str
    backup_path: str
    passed: bool
    backup_verification: dict[str, Any]
    restore: dict[str, Any] | None
    prepare: dict[str, Any] | None
    apply: dict[str, Any] | None
    review: dict[str, Any] | None
    action_gates: list[dict[str, Any]]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "backup_path": self.backup_path,
            "passed": self.passed,
            "backup_verification": self.backup_verification,
            "restore": self.restore,
            "prepare": self.prepare,
            "apply": self.apply,
            "review": self.review,
            "action_gates": self.action_gates,
            "recommendations": self.recommendations,
            "safety": [
                "Runs only inside a temporary restored backup.",
                "Does not mutate the live memory store.",
                "Passing preflight is evidence for designing stronger gates, not permission to promote, supersede, rewrite, delete, or cool live memory.",
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Strong Preflight: {self.scope}",
            f"status: {'pass' if self.passed else 'fail'}",
            f"query: {compact_text(self.query, limit=260)}",
            f"backup: {self.backup_path}",
            f"backup_verified: {self.backup_verification.get('passed')}",
        ]
        if self.restore is not None:
            lines.append(f"restore: {self.restore.get('passed')}")
        if self.prepare is not None:
            lines.append(f"prepared: {self.prepare.get('prepared')}")
        if self.apply is not None:
            lines.append(f"shadow_apply: {self.apply.get('passed')}")
        if self.review is not None:
            lines.append(
                "shadow_review: "
                f"{self.review.get('passed')} "
                f"reviewed={self.review.get('reviewed')} "
                f"fail={self.review.get('fail_count')}"
            )
        if self.action_gates:
            lines.append("## Action Gates")
            for item in self.action_gates:
                lines.append(
                    f"- {item['action']}: design_ready={item['design_ready']}, "
                    f"live_authorized={item['live_authorized']}"
                )
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationShadowRollbackItem:
    witness_id: str
    approval_id: str
    capsule_id: str
    status: str
    rolled_back: bool
    before_status: str | None
    after_status: str | None
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "approval_id": self.approval_id,
            "capsule_id": self.capsule_id,
            "status": self.status,
            "passed": self.passed,
            "rolled_back": self.rolled_back,
            "before_status": self.before_status,
            "after_status": self.after_status,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class ReconsolidationShadowRollbackReport:
    scope: str | None
    backup_path: str
    passed: bool
    backup_verification: dict[str, Any]
    restore: dict[str, Any] | None
    reviewed: int
    rolled_back: int
    fail_count: int
    items: list[ReconsolidationShadowRollbackItem]
    doctor: dict[str, Any] | None
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "backup_path": self.backup_path,
            "passed": self.passed,
            "backup_verification": self.backup_verification,
            "restore": self.restore,
            "reviewed": self.reviewed,
            "rolled_back": self.rolled_back,
            "fail_count": self.fail_count,
            "items": [item.as_dict() for item in self.items],
            "doctor": self.doctor,
            "recommendations": self.recommendations,
            "safety": [
                "Runs only inside a temporary restored backup.",
                "Does not mutate the live memory store.",
                "Rejects only the candidate frame capsule created by a reconsolidation witness.",
                "Preserves source events, evidence capsules, approvals, and witnesses for audit.",
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Shadow Rollback: {self.scope or 'all'}",
            f"status: {'pass' if self.passed else 'fail'}",
            f"backup: {self.backup_path}",
            f"backup_verified: {self.backup_verification.get('passed')}",
            f"reviewed={self.reviewed}, rolled_back={self.rolled_back}, fail={self.fail_count}",
        ]
        if self.restore is not None:
            lines.append(f"restore: {self.restore.get('passed')}")
        if self.doctor is not None:
            lines.append(f"doctor: {self.doctor.get('passed')}")
        if not self.items:
            lines.append("- No reconsolidation witnesses matched the rollback filter.")
        for item in self.items[:20]:
            warning = "; ".join(item.warnings) if item.warnings else "rollback invariants hold"
            lines.append(
                f"- [{item.status}] {item.witness_id} capsule={item.capsule_id}: "
                f"{item.before_status or 'missing'} -> {item.after_status or 'missing'}; {warning}"
            )
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationLiveRollbackApproval:
    approval_id: str
    token: str
    expires_at: str
    shadow: ReconsolidationShadowRollbackReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "token": self.token,
            "expires_at": self.expires_at,
            "scope": self.shadow.scope,
            "witness_id": self.shadow.items[0].witness_id if self.shadow.items else None,
            "capsule_id": self.shadow.items[0].capsule_id if self.shadow.items else None,
            "shadow": self.shadow.as_dict(),
            "safety": [
                "Approval is short-lived and one-use.",
                "Live rollback is limited to one explicit reconsolidation witness.",
                "Live rollback may only reject the candidate frame capsule created by that witness.",
                "Evidence capsules, source events, approvals, and reconsolidation witnesses are preserved.",
            ],
        }

    def to_text(self) -> str:
        item = self.shadow.items[0] if self.shadow.items else None
        return "\n".join(
            [
                "# Ara Reconsolidation Live Rollback Prepare",
                f"approval_id: {self.approval_id}",
                f"expires_at: {self.expires_at}",
                f"approval_token: {self.token}",
                f"witness_id: {item.witness_id if item else 'none'}",
                f"capsule_id: {item.capsule_id if item else 'none'}",
                "## Safety",
                "- Shadow rollback already passed on a verified restored backup.",
                "- Live execution still requires exact confirmation.",
                "- The only allowed live mutation is candidate frame capsule rejection.",
            ]
        )


@dataclass(slots=True)
class ReconsolidationLiveRollbackReport:
    scope: str | None
    passed: bool
    rollback_approval_id: str | None
    rollback_witness_id: str | None
    reconsolidation_witness_id: str | None
    capsule_id: str | None
    before_status: str | None
    after_status: str | None
    doctor: dict[str, Any] | None
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "rollback_approval_id": self.rollback_approval_id,
            "rollback_witness_id": self.rollback_witness_id,
            "reconsolidation_witness_id": self.reconsolidation_witness_id,
            "capsule_id": self.capsule_id,
            "before_status": self.before_status,
            "after_status": self.after_status,
            "doctor": self.doctor,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Live Rollback: {self.scope or 'unknown'}",
            f"status: {'pass' if self.passed else 'blocked'}",
            f"rollback_approval_id: {self.rollback_approval_id or 'none'}",
            f"rollback_witness_id: {self.rollback_witness_id or 'none'}",
            f"reconsolidation_witness_id: {self.reconsolidation_witness_id or 'none'}",
            f"capsule_id: {self.capsule_id or 'none'}",
            f"status_change: {self.before_status or 'none'} -> {self.after_status or 'none'}",
        ]
        if self.doctor is not None:
            lines.append(f"doctor: {self.doctor.get('passed')}")
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationExceptionWitnessReport:
    scope: str | None
    passed: bool
    exception_witness_id: str | None
    reconsolidation_witness_id: str | None
    capsule_id: str | None
    field: str | None
    before_digest: str | None
    after_digest: str | None
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "exception_witness_id": self.exception_witness_id,
            "reconsolidation_witness_id": self.reconsolidation_witness_id,
            "capsule_id": self.capsule_id,
            "field": self.field,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Exception Witness: {self.scope or 'unknown'}",
            f"status: {'pass' if self.passed else 'blocked'}",
            f"exception_witness_id: {self.exception_witness_id or 'none'}",
            f"reconsolidation_witness_id: {self.reconsolidation_witness_id or 'none'}",
            f"capsule_id: {self.capsule_id or 'none'}",
            f"field: {self.field or 'none'}",
        ]
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class ReconsolidationExceptionWitnessList:
    scope: str | None
    witness_id: str | None
    items: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "witness_id": self.witness_id,
            "items": self.items,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Exception Witnesses: {self.scope or 'all'}",
            f"items={len(self.items)}",
        ]
        if not self.items:
            lines.append("- No exception witnesses matched the filter.")
        for item in self.items[:30]:
            lines.append(
                f"- [{item['status']}] {item['field']} capsule={item['capsule_id']} "
                f"witness={item['reconsolidation_witness_id']}: {item['reason']}"
            )
        return "\n".join(lines)


def build_reconsolidation_frame(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budgets: list[int] | None = None,
    working_budget: int = 900,
    recall_budget: int = 1600,
    include_global: bool = True,
    include_hot: bool = True,
) -> ReconsolidationReport:
    memory.init()
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query is required.")
    normalized_budgets = sorted({int(item) for item in (budgets or [800, 1600, 2500]) if int(item) > 0})
    if not normalized_budgets:
        raise ValueError("At least one recall budget is required.")
    recall_policy = memory.recall_policy(
        clean_query,
        scope=scope,
        budgets=normalized_budgets,
        include_global=include_global,
        include_hot=include_hot,
    )
    working = memory.working_memory(
        prompt=clean_query,
        scope=scope,
        budget=working_budget,
        recall_budget=recall_budget,
        include_global=include_global,
        include_hot=include_hot,
    )
    lifecycle = memory.lifecycle(scope=scope, limit=1000, examples_per_tier=2)
    failure_audit = memory.failure_kind_audit(scope=scope, statuses=["candidate", "stable"], dry_run=True)
    self_audit = memory.self_kind_audit(scope=scope, statuses=["candidate", "stable"], dry_run=True)

    items = [_evidence_from_working_item(item) for item in working.items]
    purpose = _select(items, kinds={"goal", "self", "preference"}, sections={"keep", "action"})
    decisions = _select(items, kinds={"decision", "procedure"}, sections={"keep", "action"})
    frictions = _select(items, kinds={"failure", "conflict"}, sections={"risk", "action"})
    weak_context = _select(items, kinds={"project", "summary", "fact"}, sections={"keep", "action"})
    frames = [
        ReconsolidationFrame(
            "purpose and identity",
            "anchor" if purpose else "watch",
            "Keep long-running goal and identity visible before compressing new evidence.",
            purpose or _lifecycle_core_evidence(lifecycle),
        ),
        ReconsolidationFrame(
            "settled decisions",
            "preserve" if decisions else "watch",
            "Treat existing decisions as context to preserve unless current evidence contradicts them.",
            decisions,
        ),
        ReconsolidationFrame(
            "failure and conflict",
            "inspect" if frictions else "clear",
            "Carry forward only relevant hazards; do not let old failure labels dominate the frame.",
            frictions,
        ),
        ReconsolidationFrame(
            "working context",
            "retrieve" if weak_context else "minimal",
            "Use query-selected working memory instead of rereading raw history.",
            weak_context,
        ),
        ReconsolidationFrame(
            "forgetting boundary",
            "protect",
            "Cold and guarded memory stays outside the model context unless a separate audited path asks for it.",
            _forgetting_evidence(lifecycle),
        ),
    ]
    diagnostics = {
        "intent": recall_policy.intent,
        "recall_policy_status": recall_policy.status,
        "recommended_budget": recall_policy.recall_plan.recommended_budget,
        "recall_tokens": recall_policy.recall_plan.estimated_tokens,
        "working_items": len(working.items),
        "core_capsules": lifecycle.totals["core_capsules"],
        "working_capsules": lifecycle.totals["working_capsules"],
        "guarded_capsules": lifecycle.totals["guarded_capsules"],
        "cold_capsules": lifecycle.totals["cold_capsules"],
        "false_failure_candidates": failure_audit.changed,
        "false_self_candidates": self_audit.changed,
        "frame_tokens": estimate_tokens("\n".join(frame.thesis for frame in frames)),
        "read_only": True,
    }
    status = _status(recall_policy, working, lifecycle, failure_audit, self_audit)
    return ReconsolidationReport(
        scope=scope,
        query=clean_query,
        status=status,
        frames=frames,
        diagnostics=diagnostics,
        recommendations=_recommend(status, diagnostics, recall_policy, working),
    )


def prepare_reconsolidation_approval(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budgets: list[int] | None = None,
    working_budget: int = 900,
    recall_budget: int = 1600,
    include_global: bool = True,
    include_hot: bool = True,
    ttl_minutes: int = 60,
) -> ReconsolidationApproval:
    frame = build_reconsolidation_frame(
        memory,
        query,
        scope=scope,
        budgets=budgets,
        working_budget=working_budget,
        recall_budget=recall_budget,
        include_global=include_global,
        include_hot=include_hot,
    )
    fingerprint = _frame_fingerprint(frame)
    rollback_witness = _reconsolidation_rollback_witness(memory, frame, fingerprint=fingerprint)
    if frame.status == "fail":
        return ReconsolidationApproval(
            approval_id=None,
            token=None,
            expires_at=None,
            frame=frame,
            frame_fingerprint=fingerprint,
            rollback_witness=rollback_witness,
        )
    params = {
        "scope": scope,
        "budgets": sorted({int(item) for item in (budgets or [800, 1600, 2500]) if int(item) > 0}),
        "working_budget": working_budget,
        "recall_budget": recall_budget,
        "include_global": include_global,
        "include_hot": include_hot,
    }
    token = secrets.token_urlsafe(24)
    approval_id = new_id("recon_approval")
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
    memory.store.init()
    with memory.store.session() as conn:
        conn.execute(
            """
            INSERT INTO reconsolidation_approvals(
              id, token_hash, scope, query, frame_json, frame_fingerprint,
              params_json, rollback_witness_json, expires_at, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                _token_hash(token),
                frame.scope,
                frame.query,
                json.dumps(frame.as_dict(), ensure_ascii=False, sort_keys=True),
                fingerprint,
                json.dumps(params, ensure_ascii=False, sort_keys=True),
                json.dumps(rollback_witness, ensure_ascii=False, sort_keys=True),
                expires_at,
                "prepared",
                utc_now(),
            ),
        )
    return ReconsolidationApproval(
        approval_id=approval_id,
        token=token,
        expires_at=expires_at,
        frame=frame,
        frame_fingerprint=fingerprint,
        rollback_witness=rollback_witness,
    )


def apply_reconsolidation_approval(
    memory: Any,
    *,
    approval_token: str,
    confirmation: str,
) -> ReconsolidationApplyReport:
    memory.store.init()
    if confirmation != RECONSOLIDATION_APPLY_CONFIRMATION:
        return ReconsolidationApplyReport(
            scope=None,
            passed=False,
            approval_id=None,
            capsule_id=None,
            witness_id=None,
            recommendations=[f"Confirmation must exactly match: {RECONSOLIDATION_APPLY_CONFIRMATION}"],
        )
    approval = _load_approval(memory, approval_token)
    if approval is None:
        return ReconsolidationApplyReport(
            scope=None,
            passed=False,
            approval_id=None,
            capsule_id=None,
            witness_id=None,
            recommendations=["Approval token was not found."],
        )
    if approval["status"] != "prepared":
        return _blocked_apply_report(approval, f"Approval status is {approval['status']}, not prepared.")
    if _is_expired(str(approval["expires_at"])):
        _mark_approval_status(memory, str(approval["id"]), "expired")
        return _blocked_apply_report(approval, "Approval token is expired.")

    params = json.loads(str(approval["params_json"]))
    current = build_reconsolidation_frame(
        memory,
        str(approval["query"]),
        scope=str(params["scope"]),
        budgets=list(params["budgets"]),
        working_budget=int(params["working_budget"]),
        recall_budget=int(params["recall_budget"]),
        include_global=bool(params["include_global"]),
        include_hot=bool(params["include_hot"]),
    )
    current_fingerprint = _frame_fingerprint(current)
    if current_fingerprint != approval["frame_fingerprint"]:
        return _blocked_apply_report(
            approval,
            "Reconsolidation frame changed after approval; rerun reconsolidation-prepare.",
        )
    if current.status == "fail":
        return _blocked_apply_report(approval, "Current reconsolidation frame is failing.")
    rollback_witness = _load_rollback_witness(approval)
    if rollback_witness.get("frame_fingerprint") and rollback_witness.get("frame_fingerprint") != current_fingerprint:
        return _blocked_apply_report(
            approval,
            "Approval rollback witness fingerprint does not match the current frame.",
        )

    with memory.store.session() as conn:
        status = conn.execute(
            "SELECT status FROM reconsolidation_approvals WHERE id = ?",
            (approval["id"],),
        ).fetchone()
        if status is None or status["status"] != "prepared":
            return _blocked_apply_report(approval, "Approval was consumed or changed before apply.")

    before = _capsule_snapshot(memory, current)
    before["approval_rollback_witness"] = rollback_witness
    event = Event.create(
        kind=EventKind.NOTE,
        text=current.to_text(),
        source="reconsolidation-apply",
        scope=current.scope,
        metadata={
            "approval_id": approval["id"],
            "frame_fingerprint": current_fingerprint,
            "query": current.query,
        },
    )
    event = memory.store.append_event(event)
    capsule = Capsule.create(
        kind=CapsuleKind.SUMMARY,
        title=f"Reconsolidation frame: {compact_text(current.query, limit=80)}",
        body=_frame_capsule_body(current),
        scope=current.scope,
        confidence=0.74,
        salience=0.62,
        source_event_ids=[event.id],
        tags=["reconsolidation", "frame", "candidate", "read-only-source"],
        status=MemoryStatus.CANDIDATE,
    )
    witness_id = new_id("recon_witness")
    now = utc_now()
    with memory.store.session() as conn:
        memory.store.upsert_capsule(capsule)
        after = _capsule_snapshot(memory, current, new_capsule_id=capsule.id)
        conn.execute(
            """
            INSERT INTO reconsolidation_witnesses(
              id, approval_id, scope, query, capsule_id, action, reason,
              frame_fingerprint, before_json, after_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                witness_id,
                approval["id"],
                current.scope,
                current.query,
                capsule.id,
                "create-candidate-frame-summary",
                "approved read-only reconsolidation frame snapshot",
                current_fingerprint,
                json.dumps(before, ensure_ascii=False, sort_keys=True),
                json.dumps(after, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        conn.execute(
            "UPDATE reconsolidation_approvals SET status = ?, used_at = ? WHERE id = ?",
            ("used", now, approval["id"]),
        )
        conn.execute(
            """
            INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "reconsolidation-frame-candidate",
                capsule.id,
                current.scope,
                "created candidate summary from approved reconsolidation frame",
                "reconsolidation-apply",
                now,
            ),
        )
    return ReconsolidationApplyReport(
        scope=current.scope,
        passed=True,
        approval_id=str(approval["id"]),
        capsule_id=capsule.id,
        witness_id=witness_id,
        recommendations=[
            "Candidate frame summary was created; review the witness before any stronger reconsolidation action.",
            "Run recall-regression, doctor, and health before promoting this candidate or using it as hot memory.",
        ],
    )


def record_reconsolidation_review_queue(
    memory: Any,
    report: ReconsolidationReviewReport,
    *,
    limit: int = 50,
) -> ReconsolidationReviewQueueReport:
    memory.store.init()
    now = utc_now()
    recorded = 0
    resolved = 0
    with memory.store.session() as conn:
        for item in report.items:
            if item.status == "pass":
                resolved += _resolve_reconsolidation_review_queue(
                    conn,
                    scope=item.scope,
                    witness_id=item.witness_id,
                    now=now,
                )
                continue
            action = "block-strong-reconsolidation" if item.status == "fail" else "review-reconsolidation-witness"
            reason = "; ".join(item.warnings) or f"reconsolidation witness status {item.status}"
            if _upsert_reconsolidation_review_queue(
                conn,
                scope=item.scope,
                approval_id=item.approval_id,
                witness_id=item.witness_id,
                capsule_id=item.capsule_id,
                action=action,
                priority=0.98 if item.status == "fail" else 0.72,
                reason=reason,
                review_status=item.status,
                review_json=item.as_dict(),
                now=now,
            ):
                recorded += 1
        if report.regression is not None:
            if bool(report.regression.get("passed")):
                resolved += _resolve_reconsolidation_review_queue(
                    conn,
                    scope=report.scope,
                    witness_id=None,
                    now=now,
                    action="block-strong-reconsolidation-regression",
                )
            elif _upsert_reconsolidation_review_queue(
                conn,
                scope=report.scope or "global",
                approval_id=report.approval_id,
                witness_id=None,
                capsule_id=None,
                action="block-strong-reconsolidation-regression",
                priority=0.99,
                reason="reconsolidation review recall-regression sandbox failed",
                review_status="fail",
                review_json=report.regression,
                now=now,
            ):
                recorded += 1
        open_items = _list_reconsolidation_review_queue(conn, scope=report.scope, status="open", limit=limit)
    return ReconsolidationReviewQueueReport(
        scope=report.scope,
        recorded=recorded,
        resolved=resolved,
        open_items=open_items,
    )


def list_reconsolidation_review_queue(
    memory: Any,
    *,
    scope: str | None = None,
    status: str = "open",
    limit: int = 50,
) -> ReconsolidationReviewQueueReport:
    memory.store.init()
    with memory.store.session() as conn:
        rows = _list_reconsolidation_review_queue(conn, scope=scope, status=status, limit=limit)
    return ReconsolidationReviewQueueReport(scope=scope, recorded=0, resolved=0, open_items=rows)


def review_reconsolidation_witnesses(
    memory: Any,
    *,
    scope: str | None = None,
    approval_id: str | None = None,
    limit: int = 50,
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
) -> ReconsolidationReviewReport:
    memory.store.init()
    clauses: list[str] = []
    args: list[Any] = []
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    if approval_id:
        clauses.append("approval_id = ?")
        args.append(approval_id)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    args.append(limit)
    with memory.store.session() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT *
                FROM reconsolidation_witnesses
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                args,
            )
        ]
    items = [_review_witness(memory, row) for row in rows]
    fail_count = sum(1 for item in items if item.status == "fail")
    watch_count = sum(1 for item in items if item.status == "watch")
    pass_count = sum(1 for item in items if item.status == "pass")
    recommendations: list[str] = []
    regression_payload = None
    regression_passed = True
    if regression_cases is not None:
        regression = memory.recall_regression(regression_cases, baseline=regression_baseline)
        regression_payload = regression.as_dict()
        regression_passed = bool(regression.passed)
    if not items:
        recommendations.append("Run reconsolidation-apply before reviewing applied frame witnesses.")
    if fail_count:
        recommendations.append("Block stronger reconsolidation actions until failed witnesses are repaired or explained.")
    if watch_count:
        recommendations.append("Inspect watched reconsolidation witnesses before promoting frame summaries.")
    if regression_payload is not None:
        if regression_passed:
            recommendations.append("Recall-regression sandbox passed; stronger reconsolidation still requires explicit apply gates.")
        else:
            recommendations.insert(0, "Block stronger reconsolidation actions because recall-regression sandbox failed.")
    return ReconsolidationReviewReport(
        scope=scope,
        approval_id=approval_id,
        passed=fail_count == 0 and regression_passed,
        reviewed=len(items),
        pass_count=pass_count,
        watch_count=watch_count,
        fail_count=fail_count,
        items=items,
        recommendations=recommendations,
        regression=regression_payload,
    )


def backfill_legacy_reconsolidation_rollback_witnesses(
    memory: Any,
    *,
    scope: str | None = None,
    approval_id: str | None = None,
    apply: bool = False,
    limit: int = 50,
) -> ReconsolidationRollbackWitnessBackfillReport:
    memory.store.init()
    clauses = ["(a.rollback_witness_json IS NULL OR a.rollback_witness_json = '' OR a.rollback_witness_json = '{}')"]
    args: list[Any] = []
    if scope:
        clauses.append("w.scope = ?")
        args.append(scope)
    if approval_id:
        clauses.append("w.approval_id = ?")
        args.append(approval_id)
    where = "WHERE " + " AND ".join(clauses)
    args.append(limit)
    with memory.store.session() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT
                  w.id AS witness_id,
                  w.approval_id AS approval_id,
                  w.scope AS scope,
                  w.query AS query,
                  w.frame_fingerprint AS frame_fingerprint,
                  w.before_json AS before_json
                FROM reconsolidation_witnesses w
                JOIN reconsolidation_approvals a ON a.id = w.approval_id
                {where}
                ORDER BY w.created_at DESC
                LIMIT ?
                """,
                args,
            )
        ]
    items: list[dict[str, Any]] = []
    updates: list[tuple[str, str, str]] = []
    for row in rows:
        status = "skipped"
        reason = "not reviewed"
        try:
            before = json.loads(str(row["before_json"]))
        except json.JSONDecodeError:
            before = {}
            reason = "before snapshot is not valid JSON"
        if before.get("frame_fingerprint") != row["frame_fingerprint"]:
            reason = "before snapshot fingerprint mismatch"
        elif isinstance(before.get("approval_rollback_witness"), dict):
            reason = "before snapshot already carries approval rollback witness"
        else:
            witness = _rollback_witness_from_snapshot(
                before,
                scope=str(row["scope"]),
                query=str(row["query"]),
                frame_fingerprint=str(row["frame_fingerprint"]),
            )
            if witness is None:
                reason = "before snapshot does not contain reconstructable evidence capsules"
            else:
                status = "changed" if apply else "would-change"
                reason = "legacy approval rollback witness reconstructed from before snapshot"
                updates.append(
                    (
                        json.dumps(witness, ensure_ascii=False, sort_keys=True),
                        str(row["approval_id"]),
                        str(row["scope"]),
                    )
                )
        items.append(
            {
                "witness_id": str(row["witness_id"]),
                "approval_id": str(row["approval_id"]),
                "scope": str(row["scope"]),
                "status": status,
                "reason": reason,
            }
        )
    if apply and updates:
        now = utc_now()
        with memory.store.session() as conn:
            for payload, approval_id_value, item_scope in updates:
                conn.execute(
                    """
                    UPDATE reconsolidation_approvals
                    SET rollback_witness_json = ?
                    WHERE id = ?
                      AND (rollback_witness_json IS NULL OR rollback_witness_json = '' OR rollback_witness_json = '{}')
                    """,
                    (payload, approval_id_value),
                )
                conn.execute(
                    """
                    INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "reconsolidation-rollback-witness-backfill",
                        None,
                        item_scope,
                        f"backfilled approval rollback witness for {approval_id_value}",
                        "reconsolidation-backfill-rollback-witnesses",
                        now,
                    ),
                )
    changed = sum(1 for item in items if item["status"] in {"changed", "would-change"})
    return ReconsolidationRollbackWitnessBackfillReport(
        scope=scope,
        dry_run=not apply,
        reviewed=len(rows),
        changed=changed,
        skipped=len(rows) - changed,
        items=items,
    )


def strong_reconsolidation_preflight(
    memory: Any,
    query: str,
    *,
    backup_path: Path,
    scope: str = "global",
    budgets: list[int] | None = None,
    working_budget: int = 900,
    recall_budget: int = 1600,
    include_global: bool = True,
    include_hot: bool = True,
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
    actions: list[str] | None = None,
) -> ReconsolidationStrongPreflightReport:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query is required.")
    requested_actions = _normalize_strong_reconsolidation_actions(actions)
    backup_path = backup_path.resolve()
    backup_verification = memory.verify_backup(backup_path)
    recommendations: list[str] = []
    if not backup_verification.get("passed"):
        return ReconsolidationStrongPreflightReport(
            scope=scope,
            query=clean_query,
            backup_path=str(backup_path),
            passed=False,
            backup_verification=backup_verification,
            restore=None,
            prepare=None,
            apply=None,
            review=None,
            action_gates=_strong_action_gate_matrix(
                requested_actions,
                design_ready=False,
                reason="backup verification failed",
            ),
            recommendations=["Verify the backup before running strong reconsolidation preflight."],
        )

    with tempfile.TemporaryDirectory() as tmp:
        target_root = Path(tmp) / "shadow-memory"
        restore = memory.restore_backup(backup_path, target_root)
        if not restore.get("passed"):
            return ReconsolidationStrongPreflightReport(
                scope=scope,
                query=clean_query,
                backup_path=str(backup_path),
                passed=False,
                backup_verification=backup_verification,
                restore=restore,
                prepare=None,
                apply=None,
                review=None,
                action_gates=_strong_action_gate_matrix(
                    requested_actions,
                    design_ready=False,
                    reason="backup restore failed",
                ),
                recommendations=["Restore the backup cleanly before trusting any stronger reconsolidation preflight."],
            )
        shadow = memory.__class__(target_root)
        approval = prepare_reconsolidation_approval(
            shadow,
            clean_query,
            scope=scope,
            budgets=budgets,
            working_budget=working_budget,
            recall_budget=recall_budget,
            include_global=include_global,
            include_hot=include_hot,
            ttl_minutes=15,
        )
        prepare_payload = approval.as_dict()
        if not approval.prepared:
            return ReconsolidationStrongPreflightReport(
                scope=scope,
                query=clean_query,
                backup_path=str(backup_path),
                passed=False,
                backup_verification=backup_verification,
                restore=restore,
                prepare=_without_shadow_token(prepare_payload),
                apply=None,
                review=None,
                action_gates=_strong_action_gate_matrix(
                    requested_actions,
                    design_ready=False,
                    reason="shadow prepare failed",
                ),
                recommendations=[
                    "Shadow prepare did not produce an approval token; repair the frame before stronger reconsolidation.",
                ],
            )
        applied = apply_reconsolidation_approval(
            shadow,
            approval_token=approval.token or "",
            confirmation=RECONSOLIDATION_APPLY_CONFIRMATION,
        )
        apply_payload = applied.as_dict()
        if not applied.passed:
            return ReconsolidationStrongPreflightReport(
                scope=scope,
                query=clean_query,
                backup_path=str(backup_path),
                passed=False,
                backup_verification=backup_verification,
                restore=restore,
                prepare=_without_shadow_token(prepare_payload),
                apply=apply_payload,
                review=None,
                action_gates=_strong_action_gate_matrix(
                    requested_actions,
                    design_ready=False,
                    reason="shadow candidate apply failed",
                ),
                recommendations=[
                    "Shadow candidate apply failed; do not design stronger live mutation from this frame.",
                ],
            )
        review = review_reconsolidation_witnesses(
            shadow,
            scope=scope,
            approval_id=applied.approval_id,
            regression_cases=regression_cases,
            regression_baseline=regression_baseline,
        )
        review_payload = review.as_dict()
        passed = bool(review.passed and review.reviewed > 0 and review.fail_count == 0)
        if passed:
            recommendations.append(
                "Shadow preflight passed; stronger reconsolidation still needs a separate reviewed live gate and rollback witness design."
            )
        else:
            recommendations.append(
                "Block stronger reconsolidation: shadow witness review or recall-regression failed."
            )
        action_gates = _strong_action_gate_matrix(
            requested_actions,
            design_ready=passed,
            reason=(
                "shadow preflight evidence passed; live executor still intentionally unavailable"
                if passed
                else "shadow witness review or recall-regression failed"
            ),
        )
        return ReconsolidationStrongPreflightReport(
            scope=scope,
            query=clean_query,
            backup_path=str(backup_path),
            passed=passed,
            backup_verification=backup_verification,
            restore=restore,
            prepare=_without_shadow_token(prepare_payload),
            apply=apply_payload,
            review=review_payload,
            action_gates=action_gates,
            recommendations=recommendations,
        )


def _normalize_strong_reconsolidation_actions(actions: list[str] | None) -> list[str]:
    if not actions:
        return list(STRONG_RECONSOLIDATION_ACTIONS)
    normalized: list[str] = []
    invalid: list[str] = []
    for action in actions:
        clean = action.strip().lower()
        if clean not in STRONG_RECONSOLIDATION_ACTIONS:
            invalid.append(action)
            continue
        if clean not in normalized:
            normalized.append(clean)
    if invalid:
        allowed = ", ".join(STRONG_RECONSOLIDATION_ACTIONS)
        raise ValueError(
            f"unsupported strong reconsolidation action(s): {', '.join(invalid)}; "
            f"allowed: {allowed}"
        )
    return normalized


def _strong_action_gate_matrix(
    actions: list[str],
    *,
    design_ready: bool,
    reason: str,
) -> list[dict[str, Any]]:
    common_required = [
        "verified_backup",
        "shadow_restore",
        "shadow_prepare_apply_review",
        "recall_regression_when_manifest_is_supplied",
        "open_reconsolidation_review_queue_empty",
        "one_use_live_token_design",
        "rollback_or_exception_witness_design",
    ]
    action_requirements = {
        "promote": ["candidate_identity_compare_and_set", "promotion_provenance_review"],
        "rewrite": ["field_digest_compare_and_set", "exact_before_after_exception_witness"],
        "delete": ["cold_export_or_irreversible_operation_record", "source_event_preservation"],
        "cool": ["lifecycle_pressure_evidence", "recall_visibility_shadow_check"],
    }
    return [
        {
            "action": action,
            "design_ready": bool(design_ready),
            "live_authorized": False,
            "reason": reason,
            "required_gates": common_required + action_requirements[action],
            "denied_live_mutations": [item for item in STRONG_RECONSOLIDATION_ACTIONS if item != action],
            "confirmation_required": f"ENABLE STRONG RECONSOLIDATION {action.upper()}",
        }
        for action in actions
    ]


def shadow_reconsolidation_rollback(
    memory: Any,
    *,
    backup_path: Path,
    scope: str | None = None,
    approval_id: str | None = None,
    witness_id: str | None = None,
    limit: int = 20,
    doctor_query: str = "current memory state after reconsolidation rollback",
    recall_budget: int = 1200,
    hot_budget: int = 900,
    include_global: bool = True,
) -> ReconsolidationShadowRollbackReport:
    backup_path = backup_path.resolve()
    backup_verification = memory.verify_backup(backup_path)
    if not backup_verification.get("passed"):
        return ReconsolidationShadowRollbackReport(
            scope=scope,
            backup_path=str(backup_path),
            passed=False,
            backup_verification=backup_verification,
            restore=None,
            reviewed=0,
            rolled_back=0,
            fail_count=1,
            items=[],
            doctor=None,
            recommendations=["Verify the backup before running shadow rollback."],
        )

    with tempfile.TemporaryDirectory() as tmp:
        target_root = Path(tmp) / "shadow-memory"
        restore = memory.restore_backup(backup_path, target_root)
        if not restore.get("passed"):
            return ReconsolidationShadowRollbackReport(
                scope=scope,
                backup_path=str(backup_path),
                passed=False,
                backup_verification=backup_verification,
                restore=restore,
                reviewed=0,
                rolled_back=0,
                fail_count=1,
                items=[],
                doctor=None,
                recommendations=["Restore the backup cleanly before trusting shadow rollback."],
            )
        shadow = memory.__class__(target_root)
        rows = _select_reconsolidation_witness_rows(
            shadow,
            scope=scope,
            approval_id=approval_id,
            witness_id=witness_id,
            limit=limit,
        )
        items = [_shadow_rollback_witness(shadow, row) for row in rows]
        doctor_payload: dict[str, Any] | None = None
        if items:
            doctor = shadow.doctor(
                scope=scope or rows[0]["scope"],
                recall_query=doctor_query,
                recall_budget=recall_budget,
                hot_budget=hot_budget,
                include_global=include_global,
            )
            doctor_payload = doctor.as_dict()
        fail_count = sum(1 for item in items if not item.passed)
        rolled_back = sum(1 for item in items if item.rolled_back)
        passed = bool(items) and fail_count == 0 and rolled_back == len(items) and (
            doctor_payload is None or bool(doctor_payload.get("passed"))
        )
        recommendations: list[str] = []
        if not items:
            recommendations.append("No applied reconsolidation witness was available for rollback.")
        if fail_count:
            recommendations.append("Do not open live rollback until shadow rollback warnings are repaired.")
        if doctor_payload is not None and not doctor_payload.get("passed"):
            recommendations.append("Shadow rollback changed the restored store into a failing doctor state.")
        if passed:
            recommendations.append(
                "Shadow rollback passed; live rollback still requires an explicit approval token and witness table."
            )
        return ReconsolidationShadowRollbackReport(
            scope=scope,
            backup_path=str(backup_path),
            passed=passed,
            backup_verification=backup_verification,
            restore=restore,
            reviewed=len(items),
            rolled_back=rolled_back,
            fail_count=fail_count,
            items=items,
            doctor=doctor_payload,
            recommendations=recommendations,
        )


def prepare_live_reconsolidation_rollback(
    memory: Any,
    *,
    backup_path: Path,
    witness_id: str,
    scope: str | None = None,
    ttl_minutes: int = 30,
    doctor_query: str = "current memory state after reconsolidation rollback",
    recall_budget: int = 1200,
    hot_budget: int = 900,
    include_global: bool = True,
) -> ReconsolidationLiveRollbackApproval:
    clean_witness_id = witness_id.strip()
    if not clean_witness_id:
        raise ValueError("witness_id is required for live rollback approval.")
    memory.store.init()
    backup_path = backup_path.resolve()
    backup_identity_before = _file_identity(backup_path)
    shadow = shadow_reconsolidation_rollback(
        memory,
        backup_path=backup_path,
        scope=scope,
        witness_id=clean_witness_id,
        limit=1,
        doctor_query=doctor_query,
        recall_budget=recall_budget,
        hot_budget=hot_budget,
        include_global=include_global,
    )
    if not shadow.passed or len(shadow.items) != 1:
        raise ValueError("Cannot prepare live rollback because shadow rollback did not pass for exactly one witness.")
    backup_identity_after = _file_identity(backup_path)
    if backup_identity_before != backup_identity_after:
        raise ValueError("Cannot prepare live rollback because backup changed during shadow rollback.")
    row = _select_reconsolidation_witness_rows(
        memory,
        scope=scope,
        approval_id=None,
        witness_id=clean_witness_id,
        limit=1,
    )
    if len(row) != 1:
        raise ValueError("Cannot prepare live rollback because live reconsolidation witness is missing.")
    live_scope = str(row[0]["scope"])
    live_item = _inspect_rollback_witness(memory, row[0])
    if not live_item.passed:
        raise ValueError("Cannot prepare live rollback because live witness no longer matches rollback invariants.")
    shadow_payload = shadow.as_dict()
    shadow_payload["backup_identity"] = backup_identity_after
    token = secrets.token_urlsafe(24)
    approval_id = new_id("recon_rb_approval")
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
    capsule_snapshot = _capsule_snapshot_for_id(memory, live_item.capsule_id)
    with memory.store.session() as conn:
        conn.execute(
            """
            INSERT INTO reconsolidation_rollback_approvals(
              id, token_hash, scope, backup_path, witness_id, approval_id, capsule_id,
              shadow_json, backup_identity_json, capsule_snapshot_json, expires_at, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                _token_hash(token),
                live_scope,
                str(backup_path),
                live_item.witness_id,
                live_item.approval_id,
                live_item.capsule_id,
                json.dumps(shadow_payload, ensure_ascii=False, sort_keys=True),
                json.dumps(backup_identity_after, ensure_ascii=False, sort_keys=True),
                json.dumps(capsule_snapshot, ensure_ascii=False, sort_keys=True),
                expires_at,
                "prepared",
                utc_now(),
            ),
        )
    return ReconsolidationLiveRollbackApproval(
        approval_id=approval_id,
        token=token,
        expires_at=expires_at,
        shadow=shadow,
    )


def live_reconsolidation_rollback(
    memory: Any,
    *,
    approval_token: str,
    confirmation: str,
    doctor_query: str = "current memory state after reconsolidation rollback",
    recall_budget: int = 1200,
    hot_budget: int = 900,
    include_global: bool = True,
) -> ReconsolidationLiveRollbackReport:
    memory.store.init()
    if confirmation != RECONSOLIDATION_LIVE_ROLLBACK_CONFIRMATION:
        return ReconsolidationLiveRollbackReport(
            scope=None,
            passed=False,
            rollback_approval_id=None,
            rollback_witness_id=None,
            reconsolidation_witness_id=None,
            capsule_id=None,
            before_status=None,
            after_status=None,
            doctor=None,
            recommendations=[f"Confirmation must exactly match: {RECONSOLIDATION_LIVE_ROLLBACK_CONFIRMATION}"],
        )
    approval = _load_live_rollback_approval(memory, approval_token)
    if approval is None:
        return _blocked_live_rollback_report(None, "Approval token was not found.")
    if approval["status"] != "prepared":
        return _blocked_live_rollback_report(approval, f"Approval status is {approval['status']}, not prepared.")
    if _is_expired(str(approval["expires_at"])):
        _mark_live_rollback_approval(memory, str(approval["id"]), "expired")
        return _blocked_live_rollback_report(approval, "Approval token is expired.")
    if _file_identity(Path(str(approval["backup_path"])).resolve()) != json.loads(str(approval["backup_identity_json"])):
        return _blocked_live_rollback_report(approval, "Approved backup changed after rollback approval.")
    backup_verification = memory.verify_backup(Path(str(approval["backup_path"])).resolve())
    if not backup_verification.get("passed"):
        return _blocked_live_rollback_report(approval, "Approved backup no longer verifies.")

    rows = _select_reconsolidation_witness_rows(
        memory,
        scope=str(approval["scope"]),
        approval_id=str(approval["approval_id"]),
        witness_id=str(approval["witness_id"]),
        limit=1,
    )
    if len(rows) != 1:
        return _blocked_live_rollback_report(approval, "Approved reconsolidation witness is missing.")
    live_scope = str(rows[0]["scope"])
    item = _inspect_rollback_witness(memory, rows[0])
    if not item.passed:
        return _blocked_live_rollback_report(
            approval,
            "Approved reconsolidation witness no longer matches rollback invariants: " + "; ".join(item.warnings),
        )
    approved_snapshot = json.loads(str(approval["capsule_snapshot_json"]))
    current_snapshot = _capsule_snapshot_for_id(memory, item.capsule_id)
    if current_snapshot != approved_snapshot:
        return _blocked_live_rollback_report(approval, "Approved candidate capsule changed after rollback approval.")

    rollback_witness_id = new_id("recon_rb_witness")
    now = utc_now()
    before = {
        "schema": "reconsolidation-live-rollback-before-v1",
        "rollback_approval_id": approval["id"],
        "reconsolidation_witness_id": item.witness_id,
        "capsule": current_snapshot,
        "shadow": json.loads(str(approval["shadow_json"])),
    }
    with memory.store.session() as conn:
        row = conn.execute(
            "SELECT * FROM capsules WHERE id = ? AND status = ?",
            (item.capsule_id, MemoryStatus.CANDIDATE.value),
        ).fetchone()
        if row is None:
            return _blocked_live_rollback_report(approval, "Candidate capsule changed before live rollback.")
        cur = conn.execute(
            "UPDATE capsules SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
            (MemoryStatus.REJECTED.value, now, item.capsule_id, MemoryStatus.CANDIDATE.value),
        )
        if cur.rowcount != 1:
            return _blocked_live_rollback_report(approval, "Candidate rollback compare-and-set failed.")
        updated = conn.execute("SELECT * FROM capsules WHERE id = ?", (item.capsule_id,)).fetchone()
        if updated is None:
            return _blocked_live_rollback_report(approval, "Candidate capsule disappeared during live rollback.")
        _sync_capsule_fts_row(conn, updated)
        after_snapshot = _capsule_compare_snapshot(row_to_capsule(updated))
        after = {
            "schema": "reconsolidation-live-rollback-after-v1",
            "rollback_approval_id": approval["id"],
            "reconsolidation_witness_id": item.witness_id,
            "capsule": after_snapshot,
        }
        conn.execute(
            """
            INSERT INTO reconsolidation_rollback_witnesses(
              id, rollback_approval_id, reconsolidation_witness_id, scope, capsule_id,
              action, reason, before_json, after_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rollback_witness_id,
                approval["id"],
                item.witness_id,
                live_scope,
                item.capsule_id,
                "reject-reconsolidation-candidate-frame",
                "approved live rollback of candidate-only reconsolidation frame",
                json.dumps(before, ensure_ascii=False, sort_keys=True),
                json.dumps(after, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "reject-reconsolidation-candidate-frame",
                item.capsule_id,
                live_scope,
                f"live rollback of reconsolidation witness {item.witness_id}",
                "reconsolidation-live-rollback",
                now,
            ),
        )
        conn.execute(
            "UPDATE reconsolidation_rollback_approvals SET status = ?, used_at = ? WHERE id = ?",
            ("used", now, approval["id"]),
        )
    _invalidate_hot_scope(memory.store.hot_dir, live_scope)
    doctor = memory.doctor(
        scope=live_scope,
        recall_query=doctor_query,
        recall_budget=recall_budget,
        hot_budget=hot_budget,
        include_global=include_global,
    )
    doctor_payload = doctor.as_dict()
    return ReconsolidationLiveRollbackReport(
        scope=live_scope,
        passed=bool(doctor_payload.get("passed")),
        rollback_approval_id=str(approval["id"]),
        rollback_witness_id=rollback_witness_id,
        reconsolidation_witness_id=item.witness_id,
        capsule_id=item.capsule_id,
        before_status=item.before_status,
        after_status=MemoryStatus.REJECTED.value,
        doctor=doctor_payload,
        recommendations=[
            "Live rollback rejected only the candidate frame capsule and preserved source events and witnesses.",
            "Run health, recall-regression, backup, and restore-drill after live rollback.",
        ],
    )


def record_reconsolidation_exception_witness(
    memory: Any,
    *,
    witness_id: str,
    capsule_id: str | None = None,
    field: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> ReconsolidationExceptionWitnessReport:
    memory.store.init()
    clean_witness_id = witness_id.strip()
    clean_field = field.strip()
    clean_reason = reason.strip()
    allowed_fields = {"title_digest", "body_digest", "source_event_ids"}
    if not clean_witness_id:
        return _blocked_exception_witness("witness_id is required.")
    if clean_field not in allowed_fields:
        return _blocked_exception_witness(
            f"field must be one of: {', '.join(sorted(allowed_fields))}.",
            witness_id=clean_witness_id,
            field=clean_field or None,
        )
    if not clean_reason:
        return _blocked_exception_witness(
            "reason is required.",
            witness_id=clean_witness_id,
            field=clean_field,
        )
    rows = _select_reconsolidation_witness_rows(
        memory,
        scope=None,
        approval_id=None,
        witness_id=clean_witness_id,
        limit=1,
    )
    if len(rows) != 1:
        return _blocked_exception_witness(
            "Reconsolidation witness was not found.",
            witness_id=clean_witness_id,
            field=clean_field,
        )
    row = rows[0]
    try:
        after = json.loads(str(row["after_json"]))
    except json.JSONDecodeError:
        return _blocked_exception_witness(
            "Reconsolidation witness has malformed after snapshot.",
            scope=str(row.get("scope")),
            witness_id=clean_witness_id,
            capsule_id=str(row.get("capsule_id")),
            field=clean_field,
        )
    after_new = after.get("new_capsule")
    if not isinstance(after_new, dict):
        return _blocked_exception_witness(
            "Reconsolidation witness has no created capsule snapshot.",
            scope=str(row.get("scope")),
            witness_id=clean_witness_id,
            capsule_id=str(row.get("capsule_id")),
            field=clean_field,
        )
    target_capsule_id = (capsule_id or str(row["capsule_id"])).strip()
    if not target_capsule_id:
        return _blocked_exception_witness(
            "capsule_id is required.",
            scope=str(row["scope"]),
            witness_id=clean_witness_id,
            field=clean_field,
        )
    target_snapshot = _snapshot_for_reconsolidation_witness(after, target_capsule_id)
    if target_snapshot is None:
        return _blocked_exception_witness(
            "Target capsule is not part of the reconsolidation witness snapshot.",
            scope=str(row["scope"]),
            witness_id=clean_witness_id,
            capsule_id=target_capsule_id,
            field=clean_field,
        )
    current = _capsule_snapshot_for_id(memory, target_capsule_id)
    if current.get("missing"):
        return _blocked_exception_witness(
            "Target capsule is missing.",
            scope=str(row["scope"]),
            witness_id=clean_witness_id,
            capsule_id=target_capsule_id,
            field=clean_field,
        )
    before_digest = _snapshot_field_digest(target_snapshot, clean_field)
    after_digest = _snapshot_field_digest(current, clean_field)
    if before_digest == after_digest:
        return _blocked_exception_witness(
            "Current capsule field has not changed from the reconsolidation witness snapshot.",
            scope=str(row["scope"]),
            witness_id=clean_witness_id,
            capsule_id=target_capsule_id,
            field=clean_field,
            before_digest=before_digest,
            after_digest=after_digest,
        )
    exception_id = new_id("recon_exception")
    payload = {
        "schema": "reconsolidation-exception-witness-v1",
        "field": clean_field,
        "reason": clean_reason,
        "evidence": evidence or {},
        "approved_before": before_digest,
        "approved_after": after_digest,
    }
    with memory.store.session() as conn:
        conn.execute(
            """
            INSERT INTO reconsolidation_exception_witnesses(
              id, reconsolidation_witness_id, scope, capsule_id, field,
              before_digest, after_digest, reason, evidence_json, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exception_id,
                clean_witness_id,
                str(row["scope"]),
                target_capsule_id,
                clean_field,
                before_digest,
                after_digest,
                clean_reason,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                "active",
                utc_now(),
            ),
        )
        conn.execute(
            """
            INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "record-reconsolidation-exception-witness",
                target_capsule_id,
                str(row["scope"]),
                clean_reason,
                "reconsolidation-exception-witness",
                utc_now(),
            ),
        )
    return ReconsolidationExceptionWitnessReport(
        scope=str(row["scope"]),
        passed=True,
        exception_witness_id=exception_id,
        reconsolidation_witness_id=clean_witness_id,
        capsule_id=target_capsule_id,
        field=clean_field,
        before_digest=before_digest,
        after_digest=after_digest,
        recommendations=[
            "Exception witness recorded; reconsolidation review and rollback checks will only accept this exact digest transition.",
            "Rerun reconsolidation-review --record-queue before stronger memory mutation.",
        ],
    )


def list_reconsolidation_exception_witnesses(
    memory: Any,
    *,
    scope: str | None = None,
    witness_id: str | None = None,
    status: str = "active",
    limit: int = 50,
) -> ReconsolidationExceptionWitnessList:
    memory.store.init()
    clauses = ["status = ?"]
    args: list[Any] = [status]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    if witness_id:
        clauses.append("reconsolidation_witness_id = ?")
        args.append(witness_id)
    args.append(max(1, int(limit)))
    with memory.store.session() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT *
                FROM reconsolidation_exception_witnesses
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                args,
            )
        ]
    return ReconsolidationExceptionWitnessList(scope=scope, witness_id=witness_id, items=rows)


def _evidence_from_working_item(item: Any) -> ReconsolidationEvidence:
    return ReconsolidationEvidence(
        capsule_id=item.capsule_id,
        kind=item.kind,
        status=item.status,
        title=item.title,
        text=compact_text(item.text, limit=360),
        reason=item.reason,
    )


def _select(
    items: list[ReconsolidationEvidence],
    *,
    kinds: set[str],
    sections: set[str],
    limit: int = 4,
) -> list[ReconsolidationEvidence]:
    selected = [
        item
        for item in items
        if item.kind in kinds and _section_from_reason(item.reason, item.text) in sections
    ]
    if not selected:
        selected = [item for item in items if item.kind in kinds]
    return _dedupe_evidence(selected)[:limit]


def _dedupe_evidence(items: list[ReconsolidationEvidence]) -> list[ReconsolidationEvidence]:
    seen: set[tuple[str | None, str, str]] = set()
    out: list[ReconsolidationEvidence] = []
    for item in items:
        key = (item.capsule_id, item.kind, item.status)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _section_from_reason(reason: str, text: str) -> str:
    lower = f"{reason} {text}".lower()
    if "hazard" in lower or "failure" in lower or "conflict" in lower:
        return "risk"
    if "action" in lower or "next action" in lower or "treat this" in lower:
        return "action"
    return "keep"


def _lifecycle_core_evidence(lifecycle: Any) -> list[ReconsolidationEvidence]:
    out: list[ReconsolidationEvidence] = []
    for tier in lifecycle.tiers:
        if tier.name != "core":
            continue
        for item in tier.examples:
            out.append(
                ReconsolidationEvidence(
                    capsule_id=item.capsule_id,
                    kind=item.kind,
                    status=item.status,
                    title=item.title,
                    text=item.title,
                    reason="lifecycle core anchor",
                )
            )
    return out[:4]


def _forgetting_evidence(lifecycle: Any) -> list[ReconsolidationEvidence]:
    evidence: list[ReconsolidationEvidence] = []
    for tier in lifecycle.tiers:
        if tier.name not in {"guarded", "evidence", "archive", "reject"}:
            continue
        if tier.count <= 0:
            continue
        evidence.append(
            ReconsolidationEvidence(
                capsule_id=None,
                kind="lifecycle",
                status=tier.name,
                title=tier.policy,
                text=f"{tier.name}: count={tier.count}, estimated_tokens={tier.estimated_tokens}; {tier.policy}",
                reason="lifecycle forgetting boundary",
            )
        )
    return evidence[:4]


def _status(recall_policy: Any, working: Any, lifecycle: Any, failure_audit: Any, self_audit: Any) -> str:
    if recall_policy.status == "fail":
        return "fail"
    if lifecycle.totals["core_capsules"] == 0:
        return "fail"
    if failure_audit.changed or self_audit.changed:
        return "watch"
    if not working.items:
        return "watch"
    if lifecycle.totals["guarded_capsules"] > max(5, lifecycle.totals["active_capsules"] * 0.25):
        return "watch"
    return "pass"


def _recommend(status: str, diagnostics: dict[str, Any], recall_policy: Any, working: Any) -> list[str]:
    recommendations: list[str] = []
    if status == "fail":
        recommendations.append("Repair purpose/lifecycle evidence before using reconsolidation as an action frame.")
    if diagnostics["false_failure_candidates"] or diagnostics["false_self_candidates"]:
        recommendations.append("Run failure-kind-audit and self-kind-audit before preserving this frame.")
    if not working.items:
        recommendations.append("Capture or consolidate more direct evidence for this query before relying on a memory frame.")
    if recall_policy.actions:
        recommendations.append(
            "Follow recall-policy action order: " + ", ".join(action.name for action in recall_policy.actions[:4])
        )
    recommendations.append("Keep this frame read-only until a reviewed apply path proves recall-regression safety.")
    return recommendations


def _frame_fingerprint(frame: ReconsolidationReport) -> str:
    payload = {
        "scope": frame.scope,
        "query": frame.query,
        "status": frame.status,
        "frames": [
            {
                "name": item.name,
                "stance": item.stance,
                "thesis": item.thesis,
                "evidence": [
                    {
                        "capsule_id": evidence.capsule_id,
                        "kind": evidence.kind,
                        "status": evidence.status,
                        "title": evidence.title,
                        "reason": evidence.reason,
                    }
                    for evidence in item.evidence
                ],
            }
            for item in frame.frames
        ],
        "diagnostics": {
            key: frame.diagnostics[key]
            for key in (
                "intent",
                "recall_policy_status",
                "recommended_budget",
                "working_items",
                "core_capsules",
                "guarded_capsules",
                "false_failure_candidates",
                "false_self_candidates",
            )
            if key in frame.diagnostics
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _frame_capsule_body(frame: ReconsolidationReport) -> str:
    lines = [
        "Reconsolidation frame snapshot.",
        f"Query: {frame.query}",
        f"Status: {frame.status}",
        "",
    ]
    for item in frame.frames:
        lines.append(f"{item.name} [{item.stance}]: {item.thesis}")
        for evidence in item.evidence[:3]:
            source = evidence.capsule_id or "synthetic"
            lines.append(f"- {evidence.kind}/{evidence.status}/{source}: {compact_text(evidence.text, limit=220)}")
        lines.append("")
    lines.append("Safety: this capsule is a candidate frame snapshot, not an instruction to promote, delete, rewrite, or cool memory.")
    return "\n".join(lines)


def _reconsolidation_rollback_witness(
    memory: Any,
    frame: ReconsolidationReport,
    *,
    fingerprint: str,
) -> dict[str, Any]:
    evidence_by_id: dict[str, list[dict[str, str]]] = {}
    for frame_item in frame.frames:
        for evidence in frame_item.evidence:
            if not evidence.capsule_id:
                continue
            evidence_by_id.setdefault(evidence.capsule_id, []).append(
                {
                    "frame": frame_item.name,
                    "kind": evidence.kind,
                    "status": evidence.status,
                    "reason": evidence.reason,
                }
            )
    capsules: list[dict[str, Any]] = []
    missing_capsule_ids: list[str] = []
    for capsule_id in sorted(evidence_by_id):
        row = memory.store.get_capsule(capsule_id)
        if row is None:
            missing_capsule_ids.append(capsule_id)
            continue
        cap = row_to_capsule(row)
        snapshot = _capsule_compare_snapshot(cap)
        snapshot["evidence_roles"] = evidence_by_id[capsule_id]
        capsules.append(snapshot)
    return {
        "schema": "reconsolidation-rollback-witness-v1",
        "scope": frame.scope,
        "query_digest": _text_digest(frame.query),
        "frame_fingerprint": fingerprint,
        "candidate_only": True,
        "allowed_live_mutations": [],
        "evidence_capsule_count": len(capsules),
        "missing_capsule_ids": missing_capsule_ids,
        "evidence_capsules": capsules,
        "rollback_policy": (
            "Future stronger reconsolidation must either preserve these evidence capsule identities, "
            "statuses, source links, and content digests or write an explicit reviewed rollback/exception witness."
        ),
    }


def _capsule_snapshot(memory: Any, frame: ReconsolidationReport, *, new_capsule_id: str | None = None) -> dict[str, Any]:
    evidence_ids = sorted(
        {
            evidence.capsule_id
            for frame_item in frame.frames
            for evidence in frame_item.evidence
            if evidence.capsule_id
        }
    )
    capsules: list[dict[str, Any]] = []
    for capsule_id in evidence_ids:
        row = memory.store.get_capsule(capsule_id)
        if row is not None:
            cap = row_to_capsule(row)
            capsules.append(_capsule_compare_snapshot(cap))
    new_capsule = None
    if new_capsule_id:
        row = memory.store.get_capsule(new_capsule_id)
        if row is not None:
            cap = row_to_capsule(row)
            new_capsule = _capsule_compare_snapshot(cap)
    return {
        "frame_fingerprint": _frame_fingerprint(frame),
        "evidence_capsule_count": len(capsules),
        "evidence_capsules": capsules,
        "new_capsule": new_capsule,
    }


def _capsule_compare_snapshot(capsule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": capsule["id"],
        "kind": capsule["kind"],
        "status": capsule["status"],
        "scope": capsule["scope"],
        "source_event_ids": capsule["source_event_ids"],
        "title_digest": _text_digest(capsule["title"]),
        "body_digest": _text_digest(capsule["body"]),
    }


def _review_witness(memory: Any, row: dict[str, Any]) -> ReconsolidationReviewItem:
    warnings: list[str] = []
    before = json.loads(str(row["before_json"]))
    after = json.loads(str(row["after_json"]))
    if before.get("frame_fingerprint") != row["frame_fingerprint"]:
        warnings.append("before snapshot fingerprint mismatch")
    approval_rollback = before.get("approval_rollback_witness")
    if not isinstance(approval_rollback, dict):
        approval_rollback = _load_approval_rollback_witness_by_id(memory, str(row["approval_id"]))
    if not isinstance(approval_rollback, dict):
        warnings.append("legacy approval rollback witness missing")
    elif approval_rollback.get("frame_fingerprint") != row["frame_fingerprint"]:
        warnings.append("approval rollback witness fingerprint mismatch")
    if after.get("frame_fingerprint") != row["frame_fingerprint"]:
        warnings.append("after snapshot fingerprint mismatch")
    before_capsules = {item["id"]: item for item in before.get("evidence_capsules", []) if isinstance(item, dict)}
    after_capsules = {item["id"]: item for item in after.get("evidence_capsules", []) if isinstance(item, dict)}
    if before_capsules.keys() != after_capsules.keys():
        warnings.append("evidence capsule set changed during apply")
    for capsule_id, before_item in before_capsules.items():
        after_item = after_capsules.get(capsule_id)
        if not after_item:
            continue
        for key in ("kind", "status", "scope", "source_event_ids", "title_digest", "body_digest"):
            if before_item.get(key) != after_item.get(key):
                warnings.append(f"evidence capsule {capsule_id} changed field {key}")
                break
    capsule_id = str(row["capsule_id"])
    capsule_row = memory.store.get_capsule(capsule_id)
    capsule: dict[str, Any] | None = None
    if capsule_row is None:
        warnings.append("created frame capsule is missing")
    else:
        capsule = row_to_capsule(capsule_row)
        if capsule["status"] != MemoryStatus.CANDIDATE.value:
            warnings.append("created frame capsule is no longer candidate")
        if capsule["kind"] != CapsuleKind.SUMMARY.value:
            warnings.append("created frame capsule is not a summary")
        tags = set(capsule["tags"])
        if "reconsolidation" not in tags or "frame" not in tags:
            warnings.append("created frame capsule lacks reconsolidation tags")
    after_new = after.get("new_capsule")
    if not isinstance(after_new, dict) or after_new.get("id") != capsule_id:
        warnings.append("after snapshot missing created frame capsule")
    elif capsule is not None:
        current_new = _capsule_compare_snapshot(capsule)
        for key in ("kind", "status", "scope", "source_event_ids", "title_digest", "body_digest"):
            if current_new.get(key) != after_new.get(key):
                if not _exception_allows_change(
                    memory,
                    witness_id=str(row["id"]),
                    capsule_id=capsule_id,
                    field=key,
                    before_snapshot=after_new,
                    after_snapshot=current_new,
                ):
                    warnings.append(f"created frame capsule changed field {key}")
                break
    status = "pass"
    if warnings:
        status = "fail" if any(_is_failed_witness_warning(item) for item in warnings) else "watch"
    return ReconsolidationReviewItem(
        witness_id=str(row["id"]),
        approval_id=str(row["approval_id"]),
        scope=str(row["scope"]),
        capsule_id=capsule_id,
        status=status,
        warnings=warnings,
    )


def _is_failed_witness_warning(warning: str) -> bool:
    return warning in {
        "before snapshot fingerprint mismatch",
        "approval rollback witness fingerprint mismatch",
        "after snapshot fingerprint mismatch",
        "evidence capsule set changed during apply",
        "created frame capsule is missing",
    } or warning.startswith("evidence capsule ") or warning.startswith("created frame capsule changed field ")


def _rollback_witness_from_snapshot(
    snapshot: dict[str, Any],
    *,
    scope: str,
    query: str,
    frame_fingerprint: str,
) -> dict[str, Any] | None:
    evidence_capsules = snapshot.get("evidence_capsules")
    if not isinstance(evidence_capsules, list):
        return None
    capsules = [item for item in evidence_capsules if isinstance(item, dict)]
    if len(capsules) != len(evidence_capsules):
        return None
    return {
        "schema": "reconsolidation-rollback-witness-v1",
        "scope": scope,
        "query_digest": _text_digest(query),
        "frame_fingerprint": frame_fingerprint,
        "candidate_only": True,
        "allowed_live_mutations": [],
        "evidence_capsule_count": len(capsules),
        "missing_capsule_ids": [],
        "evidence_capsules": capsules,
        "rollback_policy": (
            "Legacy approval rollback witness reconstructed from the immutable before snapshot. "
            "Future stronger reconsolidation must preserve these evidence capsule identities, "
            "statuses, source links, and content digests or write an explicit reviewed rollback/exception witness."
        ),
        "reconstructed_from": "reconsolidation_witnesses.before_json",
    }


def _load_approval_rollback_witness_by_id(memory: Any, approval_id: str) -> dict[str, Any] | None:
    with memory.store.session() as conn:
        row = conn.execute(
            "SELECT rollback_witness_json FROM reconsolidation_approvals WHERE id = ?",
            (approval_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(str(row["rollback_witness_json"] or "{}"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and payload else None


def _upsert_reconsolidation_review_queue(
    conn: sqlite3.Connection,
    *,
    scope: str,
    approval_id: str | None,
    witness_id: str | None,
    capsule_id: str | None,
    action: str,
    priority: float,
    reason: str,
    review_status: str,
    review_json: dict[str, Any],
    now: str,
) -> bool:
    existing = conn.execute(
        f"""
        SELECT id
        FROM reconsolidation_review_queue
        WHERE scope = ?
          AND action = ?
          AND status = 'open'
          AND {'witness_id IS NULL' if witness_id is None else 'witness_id = ?'}
        LIMIT 1
        """,
        (scope, action) if witness_id is None else (scope, action, witness_id),
    ).fetchone()
    payload = json.dumps(review_json, ensure_ascii=False, sort_keys=True)
    if existing:
        conn.execute(
            """
            UPDATE reconsolidation_review_queue
            SET priority = ?, reason = ?, review_status = ?, review_json = ?, capsule_id = ?, created_at = ?
            WHERE id = ?
            """,
            (priority, reason, review_status, payload, capsule_id, now, existing["id"]),
        )
        return False
    conn.execute(
        """
        INSERT INTO reconsolidation_review_queue(
          id, scope, approval_id, witness_id, capsule_id, action, priority, reason,
          review_status, review_json, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id("recon_review"),
            scope,
            approval_id,
            witness_id,
            capsule_id,
            action,
            priority,
            reason,
            review_status,
            payload,
            "open",
            now,
        ),
    )
    return True


def _resolve_reconsolidation_review_queue(
    conn: sqlite3.Connection,
    *,
    scope: str | None,
    witness_id: str | None,
    now: str,
    action: str | None = None,
) -> int:
    clauses = ["status = 'open'"]
    where_args: list[Any] = []
    if scope:
        clauses.append("scope = ?")
        where_args.append(scope)
    if witness_id is None:
        clauses.append("witness_id IS NULL")
    else:
        clauses.append("witness_id = ?")
        where_args.append(witness_id)
    if action:
        clauses.append("action = ?")
        where_args.append(action)
    cur = conn.execute(
        f"""
        UPDATE reconsolidation_review_queue
        SET status = ?, resolved_at = ?
        WHERE {' AND '.join(clauses)}
        """,
        ("resolved", now, *where_args),
    )
    return cur.rowcount


def _list_reconsolidation_review_queue(
    conn: sqlite3.Connection,
    *,
    scope: str | None,
    status: str,
    limit: int,
) -> list[dict[str, Any]]:
    clauses = ["status = ?"]
    args: list[Any] = [status]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    args.append(limit)
    return [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM reconsolidation_review_queue
            WHERE {' AND '.join(clauses)}
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            args,
        )
    ]


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _load_approval(memory: Any, token: str) -> dict[str, Any] | None:
    with memory.store.session() as conn:
        row = conn.execute(
            "SELECT * FROM reconsolidation_approvals WHERE token_hash = ?",
            (_token_hash(token),),
        ).fetchone()
        return dict(row) if row else None


def _load_rollback_witness(approval: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(str(approval.get("rollback_witness_json") or "{}"))
    except json.JSONDecodeError:
        return {"schema": "reconsolidation-rollback-witness-v1", "malformed": True}
    return payload if isinstance(payload, dict) else {}


def _mark_approval_status(memory: Any, approval_id: str, status: str) -> None:
    with memory.store.session() as conn:
        conn.execute(
            "UPDATE reconsolidation_approvals SET status = ? WHERE id = ?",
            (status, approval_id),
        )


def _blocked_apply_report(approval: dict[str, Any], message: str) -> ReconsolidationApplyReport:
    return ReconsolidationApplyReport(
        scope=approval.get("scope"),
        passed=False,
        approval_id=approval.get("id"),
        capsule_id=None,
        witness_id=None,
        recommendations=[message],
    )


def _load_live_rollback_approval(memory: Any, token: str) -> dict[str, Any] | None:
    with memory.store.session() as conn:
        row = conn.execute(
            "SELECT * FROM reconsolidation_rollback_approvals WHERE token_hash = ?",
            (_token_hash(token),),
        ).fetchone()
        return dict(row) if row else None


def _mark_live_rollback_approval(memory: Any, approval_id: str, status: str) -> None:
    with memory.store.session() as conn:
        conn.execute(
            "UPDATE reconsolidation_rollback_approvals SET status = ? WHERE id = ?",
            (status, approval_id),
        )


def _blocked_live_rollback_report(
    approval: dict[str, Any] | None,
    message: str,
) -> ReconsolidationLiveRollbackReport:
    return ReconsolidationLiveRollbackReport(
        scope=approval.get("scope") if approval else None,
        passed=False,
        rollback_approval_id=approval.get("id") if approval else None,
        rollback_witness_id=None,
        reconsolidation_witness_id=approval.get("witness_id") if approval else None,
        capsule_id=approval.get("capsule_id") if approval else None,
        before_status=None,
        after_status=None,
        doctor=None,
        recommendations=[message],
    )


def _select_reconsolidation_witness_rows(
    memory: Any,
    *,
    scope: str | None,
    approval_id: str | None,
    witness_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    if approval_id:
        clauses.append("approval_id = ?")
        args.append(approval_id)
    if witness_id:
        clauses.append("id = ?")
        args.append(witness_id)
    args.append(max(1, int(limit)))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with memory.store.session() as conn:
        return [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT *
                FROM reconsolidation_witnesses
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                args,
            )
        ]


def _inspect_rollback_witness(memory: Any, row: dict[str, Any]) -> ReconsolidationShadowRollbackItem:
    capsule_id = str(row["capsule_id"])
    warnings: list[str] = []
    before_status: str | None = None
    try:
        before = json.loads(str(row["before_json"]))
        after = json.loads(str(row["after_json"]))
    except json.JSONDecodeError:
        return ReconsolidationShadowRollbackItem(
            witness_id=str(row["id"]),
            approval_id=str(row["approval_id"]),
            capsule_id=capsule_id,
            status="fail",
            rolled_back=False,
            before_status=None,
            after_status=None,
            warnings=["malformed witness snapshot json"],
        )
    if before.get("frame_fingerprint") != row["frame_fingerprint"]:
        warnings.append("before snapshot fingerprint mismatch")
    if after.get("frame_fingerprint") != row["frame_fingerprint"]:
        warnings.append("after snapshot fingerprint mismatch")
    warnings.extend(_evidence_snapshot_warnings(memory, before, witness_id=str(row["id"])))
    after_new = after.get("new_capsule")
    if not isinstance(after_new, dict) or after_new.get("id") != capsule_id:
        warnings.append("after snapshot missing created frame capsule")
    row_capsule = memory.store.get_capsule(capsule_id)
    if row_capsule is None:
        warnings.append("created frame capsule is missing")
    else:
        before_status = str(row_capsule["status"])
        current = _capsule_compare_snapshot(row_to_capsule(row_capsule))
        if isinstance(after_new, dict):
            for key in ("kind", "status", "scope", "source_event_ids", "title_digest", "body_digest"):
                if current.get(key) != after_new.get(key):
                    if not _exception_allows_change(
                        memory,
                        witness_id=str(row["id"]),
                        capsule_id=capsule_id,
                        field=key,
                        before_snapshot=after_new,
                        after_snapshot=current,
                    ):
                        warnings.append(f"created frame capsule changed field {key} before rollback")
                    break
        if before_status != MemoryStatus.CANDIDATE.value:
            warnings.append(f"created frame capsule status is {before_status}, not candidate")
    return ReconsolidationShadowRollbackItem(
        witness_id=str(row["id"]),
        approval_id=str(row["approval_id"]),
        capsule_id=capsule_id,
        status="pass" if not warnings else "fail",
        rolled_back=False,
        before_status=before_status,
        after_status=None,
        warnings=warnings,
    )


def _shadow_rollback_witness(memory: Any, row: dict[str, Any]) -> ReconsolidationShadowRollbackItem:
    inspected = _inspect_rollback_witness(memory, row)
    capsule_id = inspected.capsule_id
    warnings = list(inspected.warnings)
    before_status = inspected.before_status
    after_status: str | None = inspected.after_status
    rolled_back = False
    if not warnings:
        rolled_back = memory.store.update_capsule_status_if_current(
            capsule_id,
            MemoryStatus.CANDIDATE,
            MemoryStatus.REJECTED,
            actor="reconsolidation-shadow-rollback",
            reason=f"shadow rollback of reconsolidation witness {row['id']}",
        )
        if not rolled_back:
            warnings.append("candidate rollback compare-and-set failed")
        updated = memory.store.get_capsule(capsule_id)
        after_status = str(updated["status"]) if updated is not None else None
        if after_status != MemoryStatus.REJECTED.value:
            warnings.append("created frame capsule was not rejected by rollback")
    status = "pass" if rolled_back and not warnings else "fail"
    return ReconsolidationShadowRollbackItem(
        witness_id=str(row["id"]),
        approval_id=str(row["approval_id"]),
        capsule_id=capsule_id,
        status=status,
        rolled_back=rolled_back,
        before_status=before_status,
        after_status=after_status,
        warnings=warnings,
    )


def _evidence_snapshot_warnings(memory: Any, snapshot: dict[str, Any], *, witness_id: str) -> list[str]:
    warnings: list[str] = []
    for item in snapshot.get("evidence_capsules", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        row = memory.store.get_capsule(str(item["id"]))
        if row is None:
            warnings.append(f"evidence capsule {item['id']} is missing")
            continue
        current = _capsule_compare_snapshot(row_to_capsule(row))
        for key in ("kind", "status", "scope", "source_event_ids", "title_digest", "body_digest"):
            if current.get(key) != item.get(key):
                if not _exception_allows_change(
                    memory,
                    witness_id=witness_id,
                    capsule_id=str(item["id"]),
                    field=key,
                    before_snapshot=item,
                    after_snapshot=current,
                ):
                    warnings.append(f"evidence capsule {item['id']} changed field {key}")
                break
    return warnings


def _snapshot_for_reconsolidation_witness(snapshot: dict[str, Any], capsule_id: str) -> dict[str, Any] | None:
    after_new = snapshot.get("new_capsule")
    if isinstance(after_new, dict) and after_new.get("id") == capsule_id:
        return after_new
    for item in snapshot.get("evidence_capsules", []):
        if isinstance(item, dict) and item.get("id") == capsule_id:
            return item
    return None


def _snapshot_field_digest(snapshot: dict[str, Any], field: str) -> str:
    value = snapshot.get(field)
    if field in {"title_digest", "body_digest"}:
        return str(value or "")
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _exception_allows_change(
    memory: Any,
    *,
    witness_id: str,
    capsule_id: str,
    field: str,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
) -> bool:
    if field not in {"title_digest", "body_digest", "source_event_ids"}:
        return False
    before_digest = _snapshot_field_digest(before_snapshot, field)
    after_digest = _snapshot_field_digest(after_snapshot, field)
    with memory.store.session() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM reconsolidation_exception_witnesses
            WHERE reconsolidation_witness_id = ?
              AND capsule_id = ?
              AND field = ?
              AND before_digest = ?
              AND after_digest = ?
              AND status = 'active'
            LIMIT 1
            """,
            (witness_id, capsule_id, field, before_digest, after_digest),
        ).fetchone()
    return row is not None


def _blocked_exception_witness(
    message: str,
    *,
    scope: str | None = None,
    witness_id: str | None = None,
    capsule_id: str | None = None,
    field: str | None = None,
    before_digest: str | None = None,
    after_digest: str | None = None,
) -> ReconsolidationExceptionWitnessReport:
    return ReconsolidationExceptionWitnessReport(
        scope=scope,
        passed=False,
        exception_witness_id=None,
        reconsolidation_witness_id=witness_id,
        capsule_id=capsule_id,
        field=field,
        before_digest=before_digest,
        after_digest=after_digest,
        recommendations=[message],
    )


def _capsule_snapshot_for_id(memory: Any, capsule_id: str) -> dict[str, Any]:
    row = memory.store.get_capsule(capsule_id)
    if row is None:
        return {"id": capsule_id, "missing": True}
    return _capsule_compare_snapshot(row_to_capsule(row))


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    digest = sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _is_expired(expires_at: str) -> bool:
    try:
        parsed = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= parsed


def _text_digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _passed_count(items: object) -> str:
    if not isinstance(items, list):
        return "0/0"
    passed = sum(1 for item in items if isinstance(item, dict) and item.get("passed"))
    return f"{passed}/{len(items)}"


def _without_shadow_token(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    if redacted.get("token"):
        redacted["token"] = "<shadow-token-redacted>"
    return redacted
