from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import secrets
import tempfile
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens
from ara_memory.models import Capsule, CapsuleKind, Event, EventKind, MemoryStatus, new_id, utc_now
from ara_memory.storage import row_to_capsule


RECONSOLIDATION_APPLY_CONFIRMATION = "APPLY RECONSOLIDATION FRAME"


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
            "frame": self.frame.as_dict(),
            "safety": [
                "Prepare is read-only and stores a short-lived approval token.",
                "Apply must rebuild the same frame and match the fingerprint.",
                "Apply creates one candidate summary capsule and witness; it does not promote, supersede, rewrite, delete, or cool existing capsules.",
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Reconsolidation Prepare: {self.frame.scope}",
            f"status: {self.frame.status}",
            f"frame_fingerprint: {self.frame_fingerprint}",
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
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
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
    if frame.status == "fail":
        return ReconsolidationApproval(
            approval_id=None,
            token=None,
            expires_at=None,
            frame=frame,
            frame_fingerprint=fingerprint,
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
              params_json, expires_at, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                _token_hash(token),
                frame.scope,
                frame.query,
                json.dumps(frame.as_dict(), ensure_ascii=False, sort_keys=True),
                fingerprint,
                json.dumps(params, ensure_ascii=False, sort_keys=True),
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

    with memory.store.session() as conn:
        status = conn.execute(
            "SELECT status FROM reconsolidation_approvals WHERE id = ?",
            (approval["id"],),
        ).fetchone()
        if status is None or status["status"] != "prepared":
            return _blocked_apply_report(approval, "Approval was consumed or changed before apply.")

    before = _capsule_snapshot(memory, current)
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
) -> ReconsolidationStrongPreflightReport:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query is required.")
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
            recommendations=recommendations,
        )


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
        "after snapshot fingerprint mismatch",
        "evidence capsule set changed during apply",
        "created frame capsule is missing",
    } or warning.startswith("evidence capsule ") or warning.startswith("created frame capsule changed field ")


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _load_approval(memory: Any, token: str) -> dict[str, Any] | None:
    with memory.store.session() as conn:
        row = conn.execute(
            "SELECT * FROM reconsolidation_approvals WHERE token_hash = ?",
            (_token_hash(token),),
        ).fetchone()
        return dict(row) if row else None


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
