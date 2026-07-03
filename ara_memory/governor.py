from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens, extract_keywords
from ara_memory.costs import estimate_api_cost
from ara_memory.turn import plan_turn_ingress


@dataclass(slots=True)
class GovernanceAction:
    name: str
    status: str
    reason: str
    command: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
        }
        if self.command:
            payload["command"] = self.command
        return payload


@dataclass(slots=True)
class TurnGovernanceReport:
    scope: str
    cue: str
    recall_query: str
    budget_profile: dict[str, Any]
    capture_plan: dict[str, Any]
    agency_review: dict[str, Any]
    recall_probe: dict[str, Any]
    working_memory: dict[str, Any]
    projection_gate: dict[str, Any]
    recall_quality: dict[str, Any]
    working_memory_impact: dict[str, Any]
    cost: dict[str, Any]
    cost_gate: dict[str, Any]
    fallback_plan: dict[str, Any]
    risks: list[str] = field(default_factory=list)
    actions: list[GovernanceAction] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(risk.startswith("block:") for risk in self.risks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scope": self.scope,
            "cue": self.cue,
            "recall_query": self.recall_query,
            "budget_profile": self.budget_profile,
            "capture_plan": self.capture_plan,
            "agency_review": self.agency_review,
            "recall_probe": self.recall_probe,
            "working_memory": self.working_memory,
            "projection_gate": self.projection_gate,
            "recall_quality": self.recall_quality,
            "working_memory_impact": self.working_memory_impact,
            "cost": self.cost,
            "cost_gate": self.cost_gate,
            "fallback_plan": self.fallback_plan,
            "risks": list(self.risks),
            "actions": [action.as_dict() for action in self.actions],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Turn Governor: {self.scope}",
            f"status: {'pass' if self.passed else 'watch'}",
            f"cue: {compact_text(self.cue or '(empty)', limit=260)}",
            f"recall_query: {self.recall_query or '(none)'}",
            (
                "budget_profile: "
                f"{self.budget_profile.get('name')} "
                f"budgets={self.budget_profile.get('budgets')} "
                f"working={self.budget_profile.get('working_budget')}"
            ),
            "## Actions",
        ]
        if self.actions:
            for action in self.actions:
                command = f"; command={action.command}" if action.command else ""
                lines.append(f"- [{action.status}] {action.name}: {action.reason}{command}")
        else:
            lines.append("- [skip] no action recommended")
        lines.append("## Evidence")
        lines.append(
            "- capture: "
            f"{self.capture_plan.get('recommended_mode')} "
            f"raw_tokens={self.capture_plan.get('raw_text', {}).get('estimated_tokens_if_recalled_whole', 0)}"
        )
        lines.append(
            "- agency_review: "
            f"stance={self.agency_review.get('stance')} "
            f"action_allowed={self.agency_review.get('action_allowed')} "
            f"reasons={len(self.agency_review.get('reasons', []))}"
        )
        lines.append(
            "- recall_probe: "
            f"budget={self.recall_probe.get('budget')} "
            f"visible={self.recall_probe.get('visible_capsules')} "
            f"fallback_suppressed={self.recall_probe.get('low_evidence_fallback_suppressed')}"
        )
        lines.append(
            "- working_memory: "
            f"items={self.working_memory.get('items')} "
            f"tokens={self.working_memory.get('estimated_tokens')}"
        )
        lines.append(
            "- projection_gate: "
            f"status={self.projection_gate.get('status')} "
            f"overlap={self.projection_gate.get('visible_projected_overlap')} "
            f"action_items={self.projection_gate.get('action_items')}"
        )
        lines.append(
            "- recall_quality: "
            f"status={self.recall_quality.get('status')} "
            f"projected_feedback={len(self.recall_quality.get('projected_feedback', []))}"
        )
        impact_state = self.working_memory_impact.get("status", "not-requested")
        if impact_state != "not-requested":
            lines.append(
                "- working_memory_impact: "
                f"status={impact_state} "
                f"event_id={self.working_memory_impact.get('event_id') or 'none'}"
            )
        lines.append("## Cost")
        lines.append(
            "- local_only=True "
            f"avoided_raw_tokens={self.cost.get('avoided_raw_tokens', 0)} "
            f"estimated_model_cost_usd={self.cost.get('estimated_model_cost_usd', 0.0):.6f}"
        )
        lines.append(
            "- cost_gate: "
            f"status={self.cost_gate.get('status')} "
            f"input_tokens={self.cost_gate.get('selected_input_tokens', 0)}"
        )
        lines.append(
            "- fallback_plan: "
            f"strategy={self.fallback_plan.get('strategy')} "
            f"model_context={self.fallback_plan.get('model_context')}"
        )
        if self.risks:
            lines.append("## Risks")
            lines.extend(f"- {risk}" for risk in self.risks)
        return "\n".join(lines)


def govern_turn(
    memory: Any,
    turn: Mapping[str, Any],
    *,
    scope: str = "global",
    capture_cwd: Path | None = None,
    budgets: list[int] | None = None,
    working_budget: int | None = None,
    budget_profile: str = "auto",
    include_global: bool = True,
    include_hot: bool = True,
    direct_text_threshold: int = 16000,
    max_file_chars: int = 8000,
    max_text_chars: int = 12000,
    output_tokens: int = 0,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
    max_input_tokens: int = 0,
    max_model_cost_usd: float = 0.0,
    record_working_impact: bool = False,
    impact_outcome: str = "",
    impact_helped: bool | None = None,
) -> TurnGovernanceReport:
    """Plan memory actions for a turn without storing or rendering raw history."""

    memory.init()
    capture_plan = plan_turn_ingress(
        turn,
        capture_cwd=capture_cwd,
        max_file_chars=max_file_chars,
        max_text_chars=max_text_chars,
        direct_text_threshold=direct_text_threshold,
    )
    cue = _turn_cue(turn)
    recall_query = _recall_query(cue)
    active_files = _turn_artifact_paths(turn)
    command_errors = _turn_command_errors(turn)
    resolved_profile = _resolve_budget_profile(
        cue,
        active_files=active_files,
        command_errors=command_errors,
        requested_profile=budget_profile,
        explicit_budgets=budgets,
        explicit_working_budget=working_budget,
        explicit_max_input_tokens=max_input_tokens,
        explicit_max_model_cost_usd=max_model_cost_usd,
    )
    budgets = list(resolved_profile["budgets"])
    working_budget = int(resolved_profile["working_budget"])
    max_input_tokens = int(resolved_profile["max_input_tokens"])
    max_model_cost_usd = float(resolved_profile["max_model_cost_usd"])
    selected_probe = _select_recall_probe(
        memory,
        recall_query,
        scope=scope,
        budgets=budgets,
        include_global=include_global,
        include_hot=include_hot,
    )
    working = (
        memory.working_memory(
            prompt=cue,
            scope=scope,
            active_files=active_files,
            command_errors=command_errors,
            budget=working_budget,
            recall_budget=int(selected_probe["budget"]),
            include_global=include_global,
            include_hot=include_hot,
        )
        if cue
        else None
    )
    working_payload = _working_payload(working)
    agency = (
        memory.agency_review(
            prompt=cue,
            scope=scope,
            proposed_action="plan turn capture, recall, and working-memory actions",
            active_files=active_files,
            command_errors=command_errors,
            budgets=budgets,
            working_budget=working_budget,
            recall_budget=int(selected_probe["budget"]),
            include_global=include_global,
            include_hot=include_hot,
            include_health=False,
            record=False,
        )
        if cue
        else None
    )
    agency_payload = _agency_payload(agency)
    projection_gate = _projection_gate(working_payload, selected_probe, working_budget=working_budget)
    recall_quality = _recall_quality_payload(
        memory,
        cue=cue,
        scope=scope,
        budgets=budgets,
        include_global=include_global,
        include_hot=include_hot,
        working_budget=working_budget,
        recall_budget=int(selected_probe["budget"]),
    )
    working_impact = _record_working_impact(
        memory,
        scope=scope,
        cue=cue,
        projection_gate=projection_gate,
        requested=record_working_impact,
        outcome=impact_outcome,
        helped=impact_helped,
    )
    cost = _cost_payload(
        capture_plan,
        working_payload,
        output_tokens=output_tokens,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    cost_gate = _cost_gate(cost, max_input_tokens=max_input_tokens, max_model_cost_usd=max_model_cost_usd)
    fallback_plan = _fallback_plan(
        recall_probe=selected_probe,
        working=working_payload,
        projection_gate=projection_gate,
        recall_quality=recall_quality,
        cost=cost,
        cost_gate=cost_gate,
        budget_profile=resolved_profile,
        scope=scope,
        recall_query=recall_query,
        include_global=include_global,
        include_hot=include_hot,
    )
    risks = _risks(
        capture_plan,
        selected_probe,
        working_payload,
        projection_gate,
        recall_quality,
        cost_gate,
        command_errors,
        agency_payload,
    )
    actions = _actions(
        capture_plan,
        selected_probe,
        working_payload,
        projection_gate,
        recall_quality,
        cost_gate,
        agency_payload,
        scope=scope,
        recall_query=recall_query,
        include_global=include_global,
        include_hot=include_hot,
    )
    return TurnGovernanceReport(
        scope=scope,
        cue=cue,
        recall_query=recall_query,
        budget_profile=resolved_profile,
        capture_plan=capture_plan,
        agency_review=agency_payload,
        recall_probe=selected_probe,
        working_memory=working_payload,
        projection_gate=projection_gate,
        recall_quality=recall_quality,
        working_memory_impact=working_impact,
        cost=cost,
        cost_gate=cost_gate,
        fallback_plan=fallback_plan,
        risks=risks,
        actions=actions,
    )


def _select_recall_probe(
    memory: Any,
    query: str,
    *,
    scope: str,
    budgets: list[int],
    include_global: bool,
    include_hot: bool,
) -> dict[str, Any]:
    if not query:
        return {
            "budget": budgets[0],
            "visible_capsules": 0,
            "selected_capsules": 0,
            "low_evidence_fallback_suppressed": False,
            "impact_feedback_used": False,
            "selected_capsule_ids": [],
            "visible_capsule_ids": [],
            "query_terms": [],
        }
    probes = []
    for budget in budgets:
        result = memory.recall_candidates(
            query,
            scope=scope,
            budget=budget,
            include_global=include_global,
            include_hot=include_hot,
        )
        diagnostics = result.diagnostics
        probes.append(
            {
                "budget": budget,
                "visible_capsules": int(diagnostics.get("capsules_renderable", 0)),
                "selected_capsules": int(diagnostics.get("capsules_selected", 0)),
                "low_evidence_fallback_suppressed": bool(
                    diagnostics.get("low_evidence_fallback_suppressed", False)
                ),
                "impact_feedback_used": bool(diagnostics.get("impact_feedback_used", False)),
                "impact_boosted_capsules": list(diagnostics.get("impact_boosted_capsules", [])),
                "impact_penalized_capsules": list(diagnostics.get("impact_penalized_capsules", [])),
                "selected_capsule_ids": list(diagnostics.get("selected_capsule_ids", [])),
                "visible_capsule_ids": list(diagnostics.get("visible_capsule_ids", [])),
                "query_terms": list(diagnostics.get("query_terms", [])),
                "fallback_used": bool(diagnostics.get("fallback_used", False)),
                "salience_supplement_used": bool(diagnostics.get("salience_supplement_used", False)),
            }
        )
    for probe in probes:
        if int(probe["visible_capsules"]) > 0:
            return _with_alternatives(probe, probes)
    probe = probes[0]
    return _with_alternatives(probe, probes)


def _with_alternatives(selected: dict[str, Any], probes: list[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(selected)
    payload["alternatives"] = [dict(probe) for probe in probes]
    return payload


def _working_payload(working: Any) -> dict[str, Any]:
    if working is None:
        return {
            "items": 0,
            "estimated_tokens": 0,
            "influential_capsule_ids": [],
            "projected_capsule_ids": [],
            "low_evidence_fallback_suppressed": False,
        }
    text = working.to_text()
    return {
        "items": len(working.items),
        "estimated_tokens": estimate_tokens(text),
        "budget": int(working.budget),
        "influential_capsule_ids": working.influential_capsule_ids,
        "projected_capsule_ids": list(working.diagnostics.get("projected_capsule_ids", [])),
        "recall_visible_capsule_ids": list(working.recall_diagnostics.get("visible_capsule_ids", [])),
        "low_evidence_fallback_suppressed": bool(
            working.diagnostics.get("low_evidence_fallback_suppressed", False)
        ),
        "sections": dict(working.diagnostics.get("sections", {})),
    }


def _recall_quality_payload(
    memory: Any,
    *,
    cue: str,
    scope: str,
    budgets: list[int],
    include_global: bool,
    include_hot: bool,
    working_budget: int,
    recall_budget: int,
) -> dict[str, Any]:
    if not cue:
        return {
            "status": "skip",
            "projected_feedback": [],
            "recommendations": [],
            "policy": {},
            "working_memory": {},
            "cost_control": "not-run-without-cue",
        }
    report = memory.recall_quality(
        cue,
        scope=scope,
        budgets=budgets,
        include_global=include_global,
        include_hot=include_hot,
        working_budget=working_budget,
        recall_budget=recall_budget,
    )
    payload = report.as_dict()
    return {
        "status": payload["status"],
        "projected_feedback": payload["projected_feedback"],
        "recommendations": payload["recommendations"],
        "policy": payload["policy"],
        "working_memory": payload["working_memory"],
        "memory_impact": payload["memory_impact"],
        "policy_eval": payload["policy_eval"],
        "cost_control": "local-read-only-quality-gate",
    }


def _agency_payload(agency: Any) -> dict[str, Any]:
    if agency is None:
        return {
            "stance": "skip",
            "passed": True,
            "action_allowed": True,
            "reasons": [],
            "recommended_actions": [],
            "control_tokens": 0,
        }
    return {
        "stance": agency.stance,
        "passed": bool(agency.passed),
        "action_allowed": bool(agency.action_allowed),
        "reasons": list(agency.reasons),
        "recommended_actions": list(agency.recommended_actions),
        "control_tokens": int(agency.diagnostics.get("control_tokens", 0)),
        "purpose_passed": bool(agency.evidence.get("purpose", {}).get("passed", False)),
        "identity_passed": bool(agency.evidence.get("identity", {}).get("passed", False)),
        "recall_policy_status": agency.evidence.get("recall_policy", {}).get("status"),
        "working_items": int(agency.evidence.get("working_memory", {}).get("items", 0)),
    }


def _record_working_impact(
    memory: Any,
    *,
    scope: str,
    cue: str,
    projection_gate: dict[str, Any],
    requested: bool,
    outcome: str,
    helped: bool | None,
) -> dict[str, Any]:
    capsule_ids = list(projection_gate.get("projected_capsule_ids", []))
    if not requested:
        return {
            "status": "not-requested",
            "event_id": None,
            "capsule_ids": [],
            "helped": None,
        }
    clean_outcome = outcome.strip()
    if not clean_outcome:
        return {
            "status": "skipped",
            "event_id": None,
            "capsule_ids": capsule_ids,
            "helped": helped,
            "reason": "impact_outcome is required when recording govern-turn working-memory impact",
        }
    if not capsule_ids:
        return {
            "status": "skipped",
            "event_id": None,
            "capsule_ids": [],
            "helped": helped,
            "reason": "projection gate had no projected capsule ids",
        }
    event = memory.record_memory_impact(
        scope=scope,
        cue=cue,
        capsule_ids=capsule_ids,
        outcome=clean_outcome,
        helped=helped,
        source="govern-turn-working-memory-impact",
    )
    return {
        "status": "recorded",
        "event_id": event.id,
        "capsule_ids": capsule_ids,
        "helped": helped,
        "source": event.source,
    }


def _actions(
    capture_plan: dict[str, Any],
    recall_probe: dict[str, Any],
    working: dict[str, Any],
    projection_gate: dict[str, Any],
    recall_quality: dict[str, Any],
    cost_gate: dict[str, Any],
    agency: dict[str, Any],
    *,
    scope: str,
    recall_query: str,
    include_global: bool,
    include_hot: bool,
) -> list[GovernanceAction]:
    actions = [
        GovernanceAction(
            name="capture",
            status="spool" if capture_plan.get("recommended_mode") == "spool-turn" else "direct",
            reason="preserve raw turn evidence locally without adding it to model context",
            command=capture_plan.get("recommended_mode"),
        )
    ]
    no_global = " --no-global" if not include_global else ""
    no_hot = " --no-hot" if not include_hot else ""
    quality_status = str(recall_quality.get("status", "skip"))
    if quality_status not in {"skip", "pass"}:
        actions.append(
            GovernanceAction(
                name="recall-quality",
                status="block" if quality_status == "fail" else "watch",
                reason=_quality_reason(recall_quality),
                command=f"python -m ara_memory recall-quality \"{_shell_hint(recall_query)}\" --scope {scope}{no_global}{no_hot}",
            )
        )
    if cost_gate.get("status") != "pass":
        actions.append(
            GovernanceAction(
                name="cost-gate",
                status="block" if cost_gate.get("status") == "fail" else "watch",
                reason="; ".join(str(item) for item in cost_gate.get("reasons", [])) or "cost gate requires review",
            )
        )
    agency_stance = str(agency.get("stance", "skip"))
    if agency_stance not in {"skip", "proceed"}:
        if agency_stance in {"refuse-or-reframe", "defer-for-safety", "ask-before-acting"}:
            status = "block"
        else:
            status = "watch"
        actions.append(
            GovernanceAction(
                name="agency-review",
                status=status,
                reason=_agency_reason(agency),
                command=f"python -m ara_memory agency-review \"{_shell_hint(recall_query)}\" --scope {scope}{no_global}{no_hot}",
            )
        )
    if int(working.get("items", 0)) > 0:
        actions.append(
            GovernanceAction(
                name="working-memory",
                status="use" if projection_gate.get("status") == "pass" else "watch",
                reason=(
                    "project only visible associative memory into the next action"
                    if projection_gate.get("status") == "pass"
                    else "working-memory projection exists but needs current-evidence verification"
                ),
                command=f"python -m ara_memory working-memory \"{_shell_hint(recall_query)}\" --scope {scope}{no_global}{no_hot}",
            )
        )
    elif bool(recall_probe.get("low_evidence_fallback_suppressed")):
        actions.append(
            GovernanceAction(
                name="proceed-with-current-evidence",
                status="use",
                reason="no direct memory matched; fallback was suppressed to avoid invented context",
            )
        )
    if int(recall_probe.get("visible_capsules", 0)) >= 4:
        actions.append(
            GovernanceAction(
                name="recall-context",
                status="optional",
                reason="several visible memories exist; use a full pack only if the task needs broader evidence",
                command=f"python -m ara_memory recall-context \"{_shell_hint(recall_query)}\" --scope {scope}{no_global}{no_hot}",
            )
        )
    return actions


def _fallback_plan(
    *,
    recall_probe: dict[str, Any],
    working: dict[str, Any],
    projection_gate: dict[str, Any],
    recall_quality: dict[str, Any],
    cost: dict[str, Any],
    cost_gate: dict[str, Any],
    budget_profile: dict[str, Any],
    scope: str,
    recall_query: str,
    include_global: bool,
    include_hot: bool,
) -> dict[str, Any]:
    no_global = " --no-global" if not include_global else ""
    no_hot = " --no-hot" if not include_hot else ""
    query_hint = _shell_hint(recall_query)
    commands: list[str] = []
    reasons: list[str] = []
    actions: list[str] = []
    strategy = "use-working-memory"
    model_context = "working-memory"
    max_input_tokens = int(cost_gate.get("max_input_tokens", 0))
    selected_input_tokens = int(cost.get("selected_input_tokens", 0))

    if cost_gate.get("status") == "fail":
        strategy = "current-evidence-first"
        model_context = "none"
        reasons.extend(str(item) for item in cost_gate.get("reasons", [])[:3])
        actions.extend(["use-current-files-and-user-prompt", "run-recall-plan-before-context"])
        commands.append(f"python -m ara_memory recall-plan \"{query_hint}\" --scope {scope} --budgets {_budget_arg(budget_profile)}{no_global}{no_hot}")
        if int(working.get("estimated_tokens", 0)) <= max_input_tokens or max_input_tokens <= 0:
            actions.append("use-working-memory-only-if-current-evidence-confirms")
            commands.append(f"python -m ara_memory working-memory \"{query_hint}\" --scope {scope}{no_global}{no_hot}")
    elif recall_quality.get("status") == "fail":
        strategy = "do-not-trust-memory"
        model_context = "current-evidence"
        reasons.extend(str(item) for item in recall_quality.get("recommendations", [])[:3])
        actions.extend(["use-current-evidence", "review-recall-quality", "record-reviewed-impact-before-retry"])
        commands.append(f"python -m ara_memory recall-quality \"{query_hint}\" --scope {scope}{no_global}{no_hot}")
    elif bool(recall_probe.get("low_evidence_fallback_suppressed")) or int(recall_probe.get("visible_capsules", 0)) == 0:
        strategy = "current-evidence-first"
        model_context = "current-evidence"
        reasons.append("no direct visible memory evidence; low-evidence fallback should stay out of model context")
        actions.extend(["use-current-files-and-user-prompt", "ask-for-missing-context-if-needed"])
    elif projection_gate.get("status") != "pass":
        strategy = "bounded-working-memory-with-review"
        model_context = "working-memory-watch"
        reasons.extend(str(item) for item in projection_gate.get("reasons", [])[:3])
        actions.extend(["use-working-memory-as-cues-only", "verify-against-current-evidence"])
        commands.append(f"python -m ara_memory working-memory \"{query_hint}\" --scope {scope}{no_global}{no_hot}")
    elif recall_quality.get("status") == "watch":
        strategy = "use-working-memory-and-record-impact"
        model_context = "working-memory"
        reasons.extend(str(item) for item in recall_quality.get("recommendations", [])[:3])
        actions.extend(["use-working-memory", "record-working-memory-impact-after-reviewed-outcome"])
        commands.append(f"python -m ara_memory working-memory \"{query_hint}\" --scope {scope}{no_global}{no_hot}")
    else:
        reasons.append("working memory and cost gate are within current policy")
        actions.append("use-working-memory")
        commands.append(f"python -m ara_memory working-memory \"{query_hint}\" --scope {scope}{no_global}{no_hot}")

    if int(recall_probe.get("visible_capsules", 0)) >= 4 and cost_gate.get("status") == "pass":
        actions.append("optional-recall-context")
        commands.append(f"python -m ara_memory recall-context \"{query_hint}\" --scope {scope}{no_global}{no_hot}")

    return {
        "strategy": strategy,
        "model_context": model_context,
        "profile": budget_profile.get("name"),
        "selected_input_tokens": selected_input_tokens,
        "max_input_tokens": max_input_tokens,
        "recall_budget": int(recall_probe.get("budget", 0)),
        "working_memory_tokens": int(working.get("estimated_tokens", 0)),
        "reasons": reasons[:6],
        "actions": actions[:8],
        "commands": commands[:6],
    }


def _budget_arg(budget_profile: dict[str, Any]) -> str:
    budgets = [str(item) for item in budget_profile.get("budgets", []) if int(item) > 0]
    return ",".join(budgets) if budgets else "800"


def _risks(
    capture_plan: dict[str, Any],
    recall_probe: dict[str, Any],
    working: dict[str, Any],
    projection_gate: dict[str, Any],
    recall_quality: dict[str, Any],
    cost_gate: dict[str, Any],
    command_errors: list[str],
    agency: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    if capture_plan.get("artifacts", {}).get("missing", 0):
        risks.append("watch: one or more files/images are missing and cannot be captured")
    if command_errors:
        risks.append("watch: command errors are part of the cue and should be checked before relying on memory")
    if bool(recall_probe.get("low_evidence_fallback_suppressed")):
        risks.append("watch: low-evidence recall fallback was suppressed")
    if bool(recall_probe.get("impact_feedback_used")):
        risks.append("watch: prior working-memory impact changed ranking; treat it as bounded evidence")
    if int(working.get("items", 0)) == 0 and int(recall_probe.get("visible_capsules", 0)) == 0:
        risks.append("watch: no visible memory evidence for this cue")
    if projection_gate.get("status") == "watch":
        risks.extend(f"watch: {reason}" for reason in projection_gate.get("reasons", [])[:3])
    elif projection_gate.get("status") == "fail":
        risks.extend(f"block: {reason}" for reason in projection_gate.get("reasons", [])[:3])
    if recall_quality.get("status") == "watch":
        risks.extend(f"watch: recall-quality {reason}" for reason in recall_quality.get("recommendations", [])[:3])
    elif recall_quality.get("status") == "fail":
        risks.extend(f"block: recall-quality {reason}" for reason in recall_quality.get("recommendations", [])[:3])
    if cost_gate.get("status") == "watch":
        risks.extend(f"watch: {reason}" for reason in cost_gate.get("reasons", [])[:3])
    elif cost_gate.get("status") == "fail":
        risks.extend(f"block: {reason}" for reason in cost_gate.get("reasons", [])[:3])
    agency_stance = str(agency.get("stance", "skip"))
    if agency_stance == "refuse-or-reframe":
        risks.append("block: agency review rejected the instruction frame")
    elif agency_stance == "ask-before-acting":
        risks.append("block: agency review requires explicit approval before irreversible work")
    elif agency_stance == "defer-for-safety":
        risks.append("block: agency review deferred the turn for a safety gate")
    elif agency_stance in {"repair-memory-first", "ask-or-roadmap"}:
        risks.append(f"watch: agency review recommends {agency_stance}")
    return risks


def _projection_gate(
    working: dict[str, Any],
    recall_probe: dict[str, Any],
    *,
    working_budget: int,
) -> dict[str, Any]:
    projected_ids = set(str(item) for item in working.get("projected_capsule_ids", []))
    visible_ids = set(str(item) for item in recall_probe.get("visible_capsule_ids", []))
    working_visible_ids = set(str(item) for item in working.get("recall_visible_capsule_ids", []))
    visible_evidence_ids = visible_ids or working_visible_ids
    overlap = sorted(projected_ids.intersection(visible_evidence_ids))
    sections = dict(working.get("sections", {}))
    items = int(working.get("items", 0))
    estimated_tokens = int(working.get("estimated_tokens", 0))
    reasons: list[str] = []
    status = "pass"
    if items == 0:
        status = "watch"
        reasons.append("working-memory projected no items; use current files, logs, or user context")
    if items > 0 and visible_evidence_ids and not overlap:
        status = "watch"
        reasons.append("working-memory did not overlap visible recall evidence")
    if items > 0 and int(sections.get("action", 0)) == 0:
        status = "watch"
        reasons.append("working-memory has no action section for the next step")
    if estimated_tokens > max(0, int(working_budget)):
        status = "fail"
        reasons.append("working-memory projection exceeds its token budget")
    return {
        "status": status,
        "items": items,
        "estimated_tokens": estimated_tokens,
        "budget": int(working_budget),
        "action_items": int(sections.get("action", 0)),
        "risk_items": int(sections.get("risk", 0)),
        "keep_items": int(sections.get("keep", 0)),
        "projected_capsule_ids": sorted(projected_ids),
        "visible_evidence_capsule_ids": sorted(visible_evidence_ids),
        "visible_projected_overlap": overlap,
        "reasons": reasons,
    }


def _agency_reason(agency: dict[str, Any]) -> str:
    reasons = ", ".join(str(item) for item in agency.get("reasons", [])[:3])
    actions = ", ".join(str(item) for item in agency.get("recommended_actions", [])[:2])
    if reasons and actions:
        return f"{agency.get('stance')} because {reasons}; next: {actions}"
    if reasons:
        return f"{agency.get('stance')} because {reasons}"
    if actions:
        return f"{agency.get('stance')}; next: {actions}"
    return str(agency.get("stance", "review needed"))


def _quality_reason(recall_quality: dict[str, Any]) -> str:
    recommendations = [str(item) for item in recall_quality.get("recommendations", [])[:2]]
    if recommendations:
        return "; ".join(recommendations)
    return f"recall quality status is {recall_quality.get('status')}"


def _cost_payload(
    capture_plan: dict[str, Any],
    working: dict[str, Any],
    *,
    output_tokens: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> dict[str, Any]:
    raw_tokens = int(capture_plan.get("raw_text", {}).get("estimated_tokens_if_recalled_whole", 0))
    preview_tokens = int(capture_plan.get("recall_preview", {}).get("estimated_tokens", 0))
    working_tokens = int(working.get("estimated_tokens", 0))
    selected_input_tokens = max(working_tokens, preview_tokens)
    estimate = estimate_api_cost(
        input_tokens=selected_input_tokens,
        output_tokens=output_tokens,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    return {
        "raw_text_tokens": raw_tokens,
        "recall_preview_tokens": preview_tokens,
        "working_memory_tokens": working_tokens,
        "selected_input_tokens": selected_input_tokens,
        "avoided_raw_tokens": max(0, raw_tokens - selected_input_tokens),
        "local_planning_requires_api": False,
        "estimated_model_cost_usd": estimate.total_cost_usd,
        "input_cost_usd": estimate.input_cost_usd,
        "output_cost_usd": estimate.output_cost_usd,
    }


def _cost_gate(
    cost: dict[str, Any],
    *,
    max_input_tokens: int,
    max_model_cost_usd: float,
) -> dict[str, Any]:
    selected_input_tokens = int(cost.get("selected_input_tokens", 0))
    estimated_cost = float(cost.get("estimated_model_cost_usd", 0.0))
    reasons: list[str] = []
    status = "pass"
    if int(max_input_tokens) > 0 and selected_input_tokens > int(max_input_tokens):
        status = "fail"
        reasons.append(f"selected input tokens {selected_input_tokens} exceed max_input_tokens {int(max_input_tokens)}")
    if float(max_model_cost_usd) > 0 and estimated_cost > float(max_model_cost_usd):
        status = "fail"
        reasons.append(
            f"estimated model cost {estimated_cost:.6f} exceeds max_model_cost_usd {float(max_model_cost_usd):.6f}"
        )
    if not reasons:
        reasons.append("cost is within configured limits")
    return {
        "status": status,
        "selected_input_tokens": selected_input_tokens,
        "estimated_model_cost_usd": estimated_cost,
        "max_input_tokens": int(max_input_tokens),
        "max_model_cost_usd": float(max_model_cost_usd),
        "reasons": reasons,
    }


def _resolve_budget_profile(
    cue: str,
    *,
    active_files: list[str],
    command_errors: list[str],
    requested_profile: str,
    explicit_budgets: list[int] | None,
    explicit_working_budget: int | None,
    explicit_max_input_tokens: int,
    explicit_max_model_cost_usd: float,
) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {
        "quick": {
            "budgets": [500, 900],
            "working_budget": 650,
            "max_input_tokens": 1200,
            "max_model_cost_usd": 0.0,
            "reason": "quick status or orientation turn",
        },
        "standard": {
            "budgets": [800, 1600, 2500],
            "working_budget": 900,
            "max_input_tokens": 2600,
            "max_model_cost_usd": 0.0,
            "reason": "standard implementation or planning turn",
        },
        "deep": {
            "budgets": [1200, 2500, 4000],
            "working_budget": 1400,
            "max_input_tokens": 4400,
            "max_model_cost_usd": 0.0,
            "reason": "architecture, purpose, or long-running design turn",
        },
        "debug": {
            "budgets": [900, 1800, 3000],
            "working_budget": 1100,
            "max_input_tokens": 3400,
            "max_model_cost_usd": 0.0,
            "reason": "failure or command-error turn",
        },
        "mutation": {
            "budgets": [1200, 2200, 3200],
            "working_budget": 1200,
            "max_input_tokens": 3600,
            "max_model_cost_usd": 0.0,
            "reason": "memory mutation, deletion, rollback, or irreversible operation",
        },
    }
    requested = (requested_profile or "auto").strip().lower()
    if requested == "auto":
        name = _infer_budget_profile(cue, active_files=active_files, command_errors=command_errors)
    elif requested in profiles:
        name = requested
    else:
        name = "standard"
    profile = dict(profiles[name])
    explicit_budget_values = sorted({int(item) for item in (explicit_budgets or []) if int(item) > 0})
    if explicit_budget_values:
        profile["budgets"] = explicit_budget_values
        profile["budgets_source"] = "explicit"
    else:
        profile["budgets_source"] = "profile"
    if explicit_working_budget is not None and int(explicit_working_budget) > 0:
        profile["working_budget"] = int(explicit_working_budget)
        profile["working_budget_source"] = "explicit"
    else:
        profile["working_budget_source"] = "profile"
    if int(explicit_max_input_tokens) > 0:
        profile["max_input_tokens"] = int(explicit_max_input_tokens)
        profile["max_input_tokens_source"] = "explicit"
    else:
        profile["max_input_tokens_source"] = "profile"
    if float(explicit_max_model_cost_usd) > 0:
        profile["max_model_cost_usd"] = float(explicit_max_model_cost_usd)
        profile["max_model_cost_usd_source"] = "explicit"
    else:
        profile["max_model_cost_usd_source"] = "profile"
    profile["name"] = name
    profile["requested"] = requested
    return profile


def _infer_budget_profile(cue: str, *, active_files: list[str], command_errors: list[str]) -> str:
    text = cue.lower()
    if command_errors or any(marker in text for marker in ("traceback", "exception", "failed", "error", "실패", "에러", "오류")):
        return "debug"
    mutation_terms = (
        "delete",
        "rollback",
        "prune",
        "mutation",
        "rewrite",
        "reconsolidation",
        "apply",
        "irreversible",
        "삭제",
        "롤백",
        "되돌",
        "변경",
        "적용",
        "승인",
    )
    if any(term in text for term in mutation_terms):
        return "mutation"
    deep_terms = (
        "architecture",
        "algorithm",
        "memory os",
        "long-term",
        "free will",
        "natural memory",
        "governance",
        "아키텍처",
        "알고리즘",
        "기억저장소",
        "장기",
        "자유의지",
        "자연스러운",
        "목표",
    )
    if any(term in text for term in deep_terms):
        return "deep"
    quick_terms = ("status", "summary", "health", "current work", "상황", "상태", "요약", "현재")
    if not active_files and any(term in text for term in quick_terms):
        return "quick"
    return "standard"


def _turn_cue(turn: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("prompt", "user"):
        value = _text(turn.get(key))
        if value:
            parts.append(value)
            break
    for value in _as_list(turn.get("notes"))[:3]:
        text = _text(value)
        if text:
            parts.append(text)
    for value in _as_list(turn.get("decisions"))[:3]:
        text = _text(value)
        if text:
            parts.append(text)
    for path in _turn_artifact_paths(turn)[:8]:
        parts.append(f"active file {path}")
    for error in _turn_command_errors(turn)[:3]:
        parts.append(f"command error {compact_text(error, limit=220)}")
    return compact_text("\n".join(parts), limit=2200)


def _recall_query(cue: str) -> str:
    terms = extract_keywords(cue, limit=18)
    return " ".join(terms) if terms else compact_text(cue, limit=240)


def _turn_artifact_paths(turn: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("files", "images"):
        for item in _as_list(turn.get(key)):
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping) and item.get("path"):
                out.append(str(item["path"]))
    return out


def _turn_command_errors(turn: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for command in _as_list(turn.get("commands")):
        if isinstance(command, Mapping):
            exit_code = command.get("exit_code")
            output = _text(command.get("output"))
            failed = isinstance(exit_code, int) and exit_code != 0
            suspicious = any(marker in output.lower() for marker in ("error", "failed", "traceback", "exception"))
            if failed or suspicious:
                cmd = _text(command.get("cmd"))
                errors.append(compact_text(f"{cmd}\n{output}", limit=600))
        else:
            text = _text(command)
            if any(marker in text.lower() for marker in ("error", "failed", "traceback", "exception")):
                errors.append(compact_text(text, limit=600))
    return errors


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _shell_hint(value: str) -> str:
    return value.replace('"', "'")[:180]
