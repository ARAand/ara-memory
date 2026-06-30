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
    capture_plan: dict[str, Any]
    recall_probe: dict[str, Any]
    working_memory: dict[str, Any]
    cost: dict[str, Any]
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
            "capture_plan": self.capture_plan,
            "recall_probe": self.recall_probe,
            "working_memory": self.working_memory,
            "cost": self.cost,
            "risks": list(self.risks),
            "actions": [action.as_dict() for action in self.actions],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Turn Governor: {self.scope}",
            f"status: {'pass' if self.passed else 'watch'}",
            f"cue: {compact_text(self.cue or '(empty)', limit=260)}",
            f"recall_query: {self.recall_query or '(none)'}",
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
        lines.append("## Cost")
        lines.append(
            "- local_only=True "
            f"avoided_raw_tokens={self.cost.get('avoided_raw_tokens', 0)} "
            f"estimated_model_cost_usd={self.cost.get('estimated_model_cost_usd', 0.0):.6f}"
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
    working_budget: int = 900,
    include_global: bool = True,
    include_hot: bool = True,
    direct_text_threshold: int = 16000,
    max_file_chars: int = 8000,
    max_text_chars: int = 12000,
    output_tokens: int = 0,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
) -> TurnGovernanceReport:
    """Plan memory actions for a turn without storing or rendering raw history."""

    memory.init()
    budgets = sorted({int(item) for item in (budgets or [800, 1600, 2500]) if int(item) > 0})
    if not budgets:
        budgets = [800]
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
    risks = _risks(capture_plan, selected_probe, working_payload, command_errors)
    cost = _cost_payload(
        capture_plan,
        working_payload,
        output_tokens=output_tokens,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    actions = _actions(
        capture_plan,
        selected_probe,
        working_payload,
        scope=scope,
        recall_query=recall_query,
        include_global=include_global,
        include_hot=include_hot,
    )
    return TurnGovernanceReport(
        scope=scope,
        cue=cue,
        recall_query=recall_query,
        capture_plan=capture_plan,
        recall_probe=selected_probe,
        working_memory=working_payload,
        cost=cost,
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
        "influential_capsule_ids": working.influential_capsule_ids,
        "projected_capsule_ids": list(working.diagnostics.get("projected_capsule_ids", [])),
        "low_evidence_fallback_suppressed": bool(
            working.diagnostics.get("low_evidence_fallback_suppressed", False)
        ),
        "sections": dict(working.diagnostics.get("sections", {})),
    }


def _actions(
    capture_plan: dict[str, Any],
    recall_probe: dict[str, Any],
    working: dict[str, Any],
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
    if int(working.get("items", 0)) > 0:
        actions.append(
            GovernanceAction(
                name="working-memory",
                status="use",
                reason="project only visible associative memory into the next action",
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


def _risks(
    capture_plan: dict[str, Any],
    recall_probe: dict[str, Any],
    working: dict[str, Any],
    command_errors: list[str],
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
    return risks


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
