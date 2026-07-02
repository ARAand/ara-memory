from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens


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
