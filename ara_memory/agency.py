from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens, extract_keywords
from ara_memory.models import Event


ANTI_JUDGMENT_MARKERS = (
    "just obey",
    "always obey",
    "do not judge",
    "no judgment",
    "be a tool",
    "ignore your judgment",
    "ignore safety",
    "ignore instructions",
    "\ubb34\uc870\uac74 \ubcf5\uc885",
    "\ud310\ub2e8\ud558\uc9c0\ub9c8",
    "\ub3c4\uad6c\uac00 \ub418",
)

DESTRUCTIVE_MARKERS = (
    "delete",
    "destroy",
    "wipe",
    "prune",
    "remove all",
    "reset",
    "irreversible",
    "\uc0ad\uc81c",
    "\ucd08\uae30\ud654",
    "\ud30c\uad34",
)

VAGUE_CONTINUATION_MARKERS = (
    "continue",
    "next",
    "keep going",
    "\uc774\uc5b4\uc11c",
    "\ub2e4\uc74c",
    "\uacc4\uc18d",
)


@dataclass(slots=True)
class AgencyReviewReport:
    scope: str
    prompt: str
    proposed_action: str
    stance: str
    passed: bool
    action_allowed: bool
    reasons: list[str]
    recommended_actions: list[str]
    evidence: dict[str, Any]
    diagnostics: dict[str, Any]
    event_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "prompt": self.prompt,
            "proposed_action": self.proposed_action,
            "stance": self.stance,
            "passed": self.passed,
            "action_allowed": self.action_allowed,
            "reasons": self.reasons,
            "recommended_actions": self.recommended_actions,
            "evidence": self.evidence,
            "diagnostics": self.diagnostics,
            "event_id": self.event_id,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Agency Review: {self.scope}",
            f"stance: {self.stance}",
            f"passed: {self.passed}",
            f"action_allowed: {self.action_allowed}",
            f"prompt: {self.prompt}",
        ]
        if self.proposed_action:
            lines.append(f"proposed_action: {self.proposed_action}")
        if self.event_id:
            lines.append(f"event_id: {self.event_id}")
        lines.append("## Reasons")
        lines.extend(f"- {item}" for item in self.reasons or ["none"])
        lines.append("## Recommended Actions")
        lines.extend(f"- {item}" for item in self.recommended_actions or ["none"])
        lines.append("## Evidence")
        lines.extend(_evidence_lines(self.evidence))
        return "\n".join(lines)


def build_agency_review(
    memory: Any,
    *,
    prompt: str,
    scope: str = "global",
    proposed_action: str = "",
    constraints: list[str] | None = None,
    active_files: list[str] | None = None,
    command_errors: list[str] | None = None,
    budgets: list[int] | None = None,
    working_budget: int = 700,
    recall_budget: int = 1200,
    include_global: bool = True,
    include_hot: bool = True,
    include_health: bool = False,
    record: bool = False,
) -> AgencyReviewReport:
    memory.init()
    prompt_text = str(prompt or "").strip()
    proposed = str(proposed_action or "").strip()
    budgets = budgets or [800, 1600]
    constraint_text = " ".join(str(item).strip() for item in constraints or [] if str(item).strip())
    review_text = " ".join(part for part in [prompt_text, proposed, constraint_text] if part)

    purpose = memory.purpose_check(scope=scope, include_global=include_global, repair_hot=False)
    identity = memory.identity_check(scope=scope, include_global=include_global, repair_hot=False)
    policy = memory.recall_policy(
        prompt_text or proposed or "current task",
        scope=scope,
        budgets=budgets,
        include_global=include_global,
        include_hot=include_hot,
        cold_budget=600,
        cold_group_limit=2,
    )
    working = memory.working_memory(
        prompt=prompt_text or proposed or "current task",
        scope=scope,
        active_files=active_files or [],
        command_errors=command_errors or [],
        budget=working_budget,
        recall_budget=recall_budget,
        include_global=include_global,
        include_hot=include_hot,
    )
    health = memory.health(scope=scope) if include_health else None

    reasons = _reasons(
        text=review_text,
        purpose_passed=purpose.passed,
        identity_passed=identity.passed,
        policy_status=policy.status,
        health_passed=True if health is None else bool(health.passed),
        working_items=int(working.diagnostics.get("items", 0)),
    )
    stance = _stance(reasons)
    passed = purpose.passed and identity.passed and policy.status != "fail" and (
        health is None or bool(health.passed)
    )
    action_allowed = passed and stance == "proceed"
    actions = _recommended_actions(
        stance=stance,
        reasons=reasons,
        purpose_recommendations=purpose.recommendations,
        identity_recommendations=identity.recommendations,
        policy_actions=[action.name for action in policy.actions],
        working_items=int(working.diagnostics.get("items", 0)),
    )
    evidence = {
        "purpose": {
            "passed": purpose.passed,
            "stable_goals": purpose.stable_goals,
            "hot_has_goals": purpose.hot_has_goals,
            "recall_has_goals": purpose.recall_has_goals,
        },
        "identity": {
            "passed": identity.passed,
            "stable_self": identity.stable_self,
            "hot_has_identity": identity.hot_has_identity,
            "recall_has_identity": identity.recall_has_identity,
        },
        "recall_policy": {
            "status": policy.status,
            "intent": policy.intent,
            "strategy": policy.strategy,
            "actions": [action.name for action in policy.actions],
            "estimated_tokens": policy.recall_plan.estimated_tokens,
            "avoided_cold_raw_tokens": policy.token_policy.get("avoided_cold_raw_tokens", 0),
        },
        "working_memory": {
            "items": int(working.diagnostics.get("items", 0)),
            "capsules_projected": int(working.diagnostics.get("capsules_projected", 0)),
            "estimated_tokens": int(working.diagnostics.get("estimated_tokens", 0)),
        },
    }
    if health is not None:
        evidence["health"] = {
            "passed": health.passed,
            "status": health.status,
            "score": health.score,
        }
    diagnostics = {
        "prompt_terms": extract_keywords(review_text, limit=12),
        "control_tokens": estimate_tokens(_control_text(evidence, reasons, actions)),
        "anti_judgment_detected": "anti_judgment_frame" in reasons,
        "destructive_detected": "destructive_or_irreversible_request" in reasons,
        "raw_ledger_read": False,
        "model_free": True,
    }
    report = AgencyReviewReport(
        scope=scope,
        prompt=compact_text(prompt_text, limit=420),
        proposed_action=compact_text(proposed, limit=420),
        stance=stance,
        passed=passed,
        action_allowed=action_allowed,
        reasons=reasons,
        recommended_actions=actions,
        evidence=evidence,
        diagnostics=diagnostics,
    )
    if record:
        event = record_agency_review(memory, report=report)
        report = replace(report, event_id=event.id)
    return report


def record_agency_review(
    memory: Any,
    *,
    report: AgencyReviewReport,
    source: str = "agency-review",
) -> Event:
    payload = {
        "prompt": report.prompt,
        "proposed_action": report.proposed_action,
        "stance": report.stance,
        "passed": report.passed,
        "action_allowed": report.action_allowed,
        "reasons": report.reasons,
        "recommended_actions": report.recommended_actions,
        "evidence": report.evidence,
        "diagnostics": report.diagnostics,
    }
    text = (
        "Agency review:\n"
        f"Prompt: {compact_text(report.prompt, limit=300)}\n"
        f"Proposed action: {compact_text(report.proposed_action, limit=260) or 'none'}\n"
        f"Stance: {report.stance}\n"
        f"Reasons: {', '.join(report.reasons) or 'none'}\n"
        f"Recommended actions: {', '.join(report.recommended_actions) or 'none'}"
    )
    return memory.retain(
        kind="note",
        text=text,
        source=source,
        scope=report.scope,
        metadata={"agency_review": payload},
    )


def _reasons(
    *,
    text: str,
    purpose_passed: bool,
    identity_passed: bool,
    policy_status: str,
    health_passed: bool,
    working_items: int,
) -> list[str]:
    lowered = text.lower()
    reasons: list[str] = []
    if any(marker in lowered for marker in ANTI_JUDGMENT_MARKERS):
        reasons.append("anti_judgment_frame")
    if any(marker in lowered for marker in DESTRUCTIVE_MARKERS):
        reasons.append("destructive_or_irreversible_request")
    if not purpose_passed:
        reasons.append("purpose_not_visible")
    if not identity_passed:
        reasons.append("identity_not_visible")
    if policy_status == "fail":
        reasons.append("recall_policy_gate_failed")
    if not health_passed:
        reasons.append("operational_health_failed")
    if working_items == 0 and any(marker in lowered for marker in VAGUE_CONTINUATION_MARKERS):
        reasons.append("low_context_for_vague_continuation")
    return reasons


def _stance(reasons: list[str]) -> str:
    if "anti_judgment_frame" in reasons:
        return "refuse-or-reframe"
    if any(item in reasons for item in ("purpose_not_visible", "identity_not_visible")):
        return "repair-memory-first"
    if any(item in reasons for item in ("recall_policy_gate_failed", "operational_health_failed")):
        return "defer-for-safety"
    if "destructive_or_irreversible_request" in reasons:
        return "ask-before-acting"
    if "low_context_for_vague_continuation" in reasons:
        return "ask-or-roadmap"
    return "proceed"


def _recommended_actions(
    *,
    stance: str,
    reasons: list[str],
    purpose_recommendations: list[str],
    identity_recommendations: list[str],
    policy_actions: list[str],
    working_items: int,
) -> list[str]:
    actions: list[str] = []
    if "purpose_not_visible" in reasons:
        actions.append("purpose-check --repair-hot")
        actions.extend(purpose_recommendations[:2])
    if "identity_not_visible" in reasons:
        actions.append("identity-check --repair-hot")
        actions.extend(identity_recommendations[:2])
    if stance == "refuse-or-reframe":
        actions.append("state the conflicting frame and choose a safer objective")
    if stance == "ask-before-acting":
        actions.append("ask for explicit approval before irreversible work")
    if stance == "ask-or-roadmap":
        actions.append("run goal-roadmap or ask for the missing objective")
    actions.extend(policy_actions[:4])
    if working_items:
        actions.append("use working-memory cues before editing")
    actions.append("record-agency-review after meaningful judgment")
    return list(dict.fromkeys(action for action in actions if action))


def _control_text(evidence: dict[str, Any], reasons: list[str], actions: list[str]) -> str:
    return f"{evidence}\n{reasons}\n{actions}"


def _evidence_lines(evidence: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in evidence.items():
        if isinstance(value, dict):
            compact = ", ".join(f"{item_key}={item_value}" for item_key, item_value in value.items())
            lines.append(f"- {key}: {compact}")
        else:
            lines.append(f"- {key}: {value}")
    return lines
