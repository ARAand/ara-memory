from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.compressors import compact_text, extract_keywords


PURPOSE_MARKERS = {
    "purpose",
    "goal",
    "objective",
    "intent",
    "identity",
    "free will",
    "natural memory",
    "why",
    "목적",
    "목표",
    "의도",
    "정체성",
    "자유의지",
    "왜",
}
COLD_MARKERS = {
    "cold",
    "distant",
    "old",
    "older",
    "archive",
    "history",
    "past",
    "previous",
    "retention",
    "evidence",
    "오래",
    "예전",
    "과거",
    "기록",
    "아카이브",
    "증거",
}
RETENTION_MARKERS = {
    "prune",
    "delete",
    "cleanup",
    "retention",
    "backup",
    "restore",
    "quarantine",
    "reject",
    "irreversible",
    "삭제",
    "정리",
    "백업",
    "복구",
    "격리",
}
WORKING_MARKERS = {
    "current",
    "now",
    "today",
    "file",
    "code",
    "implement",
    "bug",
    "error",
    "test",
    "command",
    "현재",
    "지금",
    "오늘",
    "파일",
    "코드",
    "구현",
    "오류",
    "테스트",
    "명령",
}


@dataclass(slots=True)
class RecallPolicyAction:
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
class RecallPolicyReport:
    query: str
    scope: str
    intent: str
    strategy: str
    status: str
    include_hot: bool
    include_global: bool
    budgets: list[int]
    recall_plan: Any
    lifecycle_summary: dict[str, Any]
    cold_map_summary: dict[str, Any] | None
    token_policy: dict[str, Any]
    actions: list[RecallPolicyAction]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "intent": self.intent,
            "strategy": self.strategy,
            "status": self.status,
            "include_hot": self.include_hot,
            "include_global": self.include_global,
            "budgets": self.budgets,
            "recall_plan": self.recall_plan.as_dict(),
            "lifecycle_summary": self.lifecycle_summary,
            "cold_map_summary": self.cold_map_summary,
            "token_policy": self.token_policy,
            "actions": [action.as_dict() for action in self.actions],
            "diagnostics": self.diagnostics,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Recall Policy: {self.scope}",
            f"status: {self.status}",
            f"intent: {self.intent}",
            f"strategy: {self.strategy}",
            f"query: {compact_text(self.query, limit=260)}",
            f"recommended_budget: {self.recall_plan.recommended_budget}",
            f"include_hot: {self.include_hot}",
            f"include_global: {self.include_global}",
            "## Actions",
        ]
        if not self.actions:
            lines.append("- [skip] no recall action recommended")
        for action in self.actions:
            command = f"; command={action.command}" if action.command else ""
            lines.append(f"- [{action.status}] {action.name}: {action.reason}{command}")
        lines.append("## Evidence")
        lines.append(
            "- recall_plan: "
            f"quality={self.recall_plan.diagnostics.get('quality_score', 0)}, "
            f"visible={self.recall_plan.diagnostics.get('visible_capsules', 0)}, "
            f"tokens={self.recall_plan.estimated_tokens}/{self.recall_plan.recommended_budget}"
        )
        lines.append(
            "- lifecycle: "
            f"core={self.lifecycle_summary.get('core_capsules', 0)}, "
            f"working={self.lifecycle_summary.get('working_capsules', 0)}, "
            f"guarded={self.lifecycle_summary.get('guarded_capsules', 0)}, "
            f"cold={self.lifecycle_summary.get('cold_capsules', 0)}"
        )
        if self.cold_map_summary:
            lines.append(
                "- cold_map: "
                f"matched={self.cold_map_summary.get('matched_capsules', 0)}, "
                f"groups={self.cold_map_summary.get('groups', 0)}, "
                f"map_tokens={self.cold_map_summary.get('estimated_map_tokens', 0)}/"
                f"{self.cold_map_summary.get('budget_tokens', 0)}, "
                f"raw_tokens={self.cold_map_summary.get('matched_raw_tokens', 0)}, "
                f"reduction={self.cold_map_summary.get('token_reduction_ratio', 0.0):.1f}x"
            )
        lines.append("## Token Policy")
        lines.append(
            "- local_only=True "
            f"selected_recall_tokens={self.token_policy['selected_recall_tokens']} "
            f"cold_map_tokens={self.token_policy['cold_map_tokens']} "
            f"avoided_cold_raw_tokens={self.token_policy['avoided_cold_raw_tokens']}"
        )
        if self.diagnostics.get("risks"):
            lines.append("## Risks")
            lines.extend(f"- {risk}" for risk in self.diagnostics["risks"])
        return "\n".join(lines)


def build_recall_policy(
    memory: Any,
    query: str,
    *,
    scope: str = "global",
    budgets: list[int] | None = None,
    include_global: bool = True,
    include_hot: bool = True,
    cold_budget: int = 700,
    cold_group_limit: int = 3,
    output_tokens: int = 0,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
) -> RecallPolicyReport:
    memory.init()
    normalized_budgets = sorted({int(item) for item in (budgets or [800, 1600, 2500]) if int(item) > 0})
    if not normalized_budgets:
        raise ValueError("At least one recall budget is required.")
    intent, intent_scores = classify_recall_intent(query)
    effective_hot = include_hot
    plan = memory.recall_plan(
        query,
        scope=scope,
        budgets=normalized_budgets,
        include_global=include_global,
        include_hot=effective_hot,
        output_tokens=output_tokens,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
    )
    lifecycle = memory.lifecycle(scope=scope, limit=1000, examples_per_tier=0)
    cold_map = None
    if intent in {"distant-memory", "retention-safety"}:
        cold_map = memory.cold_map(
            scope=scope,
            query=query,
            group_limit=cold_group_limit,
            examples_per_group=0,
            budget=cold_budget,
        )
    lifecycle_summary = _lifecycle_summary(lifecycle)
    cold_summary = _cold_summary(cold_map) if cold_map is not None else None
    token_policy = _token_policy(plan, cold_summary)
    risks = _risks(
        intent=intent,
        include_hot=include_hot,
        plan=plan,
        lifecycle_summary=lifecycle_summary,
        cold_map_summary=cold_summary,
    )
    status = _status(intent, plan, lifecycle_summary, cold_summary)
    actions = _actions(
        query=query,
        scope=scope,
        intent=intent,
        include_global=include_global,
        include_hot=include_hot,
        budgets=normalized_budgets,
        plan=plan,
        cold_map_summary=cold_summary,
    )
    return RecallPolicyReport(
        query=query,
        scope=scope,
        intent=intent,
        strategy=_strategy(intent),
        status=status,
        include_hot=include_hot,
        include_global=include_global,
        budgets=normalized_budgets,
        recall_plan=plan,
        lifecycle_summary=lifecycle_summary,
        cold_map_summary=cold_summary,
        token_policy=token_policy,
        actions=actions,
        diagnostics={
            "intent_scores": intent_scores,
            "query_terms": extract_keywords(query, limit=16),
            "risks": risks,
            "cold_map_used": cold_map is not None,
            "no_raw_cold_body_rendering": True,
        },
    )


def classify_recall_intent(query: str) -> tuple[str, dict[str, int]]:
    lowered = str(query).lower()
    scores = {
        "purpose": _marker_score(lowered, PURPOSE_MARKERS),
        "cold": _marker_score(lowered, COLD_MARKERS),
        "retention": _marker_score(lowered, RETENTION_MARKERS),
        "working": _marker_score(lowered, WORKING_MARKERS),
    }
    if scores["retention"] and (scores["cold"] or "memory" in lowered or "기억" in lowered):
        return "retention-safety", scores
    if scores["purpose"] >= max(scores["cold"], scores["working"], 1):
        return "purpose-continuity", scores
    if scores["cold"]:
        return "distant-memory", scores
    if scores["working"]:
        return "working-context", scores
    return "balanced-recall", scores


def _marker_score(text: str, markers: set[str]) -> int:
    return sum(1 for marker in markers if marker in text)


def _strategy(intent: str) -> str:
    return {
        "purpose-continuity": "hot core anchors first; bounded recall pack second; no cold body rendering",
        "distant-memory": "cold-map first; use active recall only after choosing a narrow distant-memory group",
        "retention-safety": "cold-map plus retention gates; never prune from recall output alone",
        "working-context": "working-memory projection first; full recall-context only when broader evidence is needed",
        "balanced-recall": "smallest useful recall-plan budget with hot memory and global policy",
    }[intent]


def _lifecycle_summary(lifecycle: Any) -> dict[str, Any]:
    totals = lifecycle.totals
    policy = lifecycle.token_policy
    return {
        "status": lifecycle.status,
        "core_capsules": int(totals.get("core_capsules", 0)),
        "working_capsules": int(totals.get("working_capsules", 0)),
        "guarded_capsules": int(totals.get("guarded_capsules", 0)),
        "cold_capsules": int(totals.get("cold_capsules", 0)),
        "core_tokens": int(policy.get("core_tokens", 0)),
        "target_hot_tokens": int(policy.get("target_hot_tokens", 0)),
        "raw_to_core_reduction": float(policy.get("raw_to_core_reduction", 0.0)),
    }


def _cold_summary(cold_map: Any) -> dict[str, Any]:
    totals = cold_map.totals
    return {
        "status": cold_map.status,
        "matched_capsules": int(totals.get("matched_capsules", 0)),
        "groups": int(totals.get("groups", 0)),
        "matched_raw_tokens": int(totals.get("matched_raw_tokens", 0)),
        "estimated_map_tokens": int(totals.get("estimated_map_tokens", 0)),
        "budget_tokens": int(totals.get("budget_tokens", 0)),
        "within_budget": bool(totals.get("within_budget", False)),
        "token_reduction_ratio": float(totals.get("token_reduction_ratio", 0.0)),
        "top_groups": [
            {
                "tier": group.tier,
                "kind": group.kind,
                "pattern": group.pattern,
                "count": group.count,
                "source_event_digest": group.source_event_digest,
            }
            for group in cold_map.groups[:3]
        ],
    }


def _token_policy(plan: Any, cold_summary: dict[str, Any] | None) -> dict[str, Any]:
    cold_raw = int((cold_summary or {}).get("matched_raw_tokens", 0))
    cold_map_tokens = int((cold_summary or {}).get("estimated_map_tokens", 0))
    return {
        "selected_recall_tokens": int(plan.estimated_tokens),
        "selected_recall_budget": int(plan.recommended_budget),
        "cold_map_tokens": cold_map_tokens,
        "cold_raw_tokens": cold_raw,
        "avoided_cold_raw_tokens": max(0, cold_raw - cold_map_tokens),
        "api_cost": dict(plan.api_cost),
        "local_planning_requires_api": False,
    }


def _risks(
    *,
    intent: str,
    include_hot: bool,
    plan: Any,
    lifecycle_summary: dict[str, Any],
    cold_map_summary: dict[str, Any] | None,
) -> list[str]:
    risks: list[str] = []
    if not include_hot and intent in {"purpose-continuity", "balanced-recall"}:
        risks.append("watch: hot memory is disabled, so purpose and identity anchors may be absent")
    if int(plan.diagnostics.get("visible_capsules", 0)) == 0:
        risks.append("watch: active recall has no visible capsule evidence for this query")
    if bool(plan.diagnostics.get("low_evidence_fallback_suppressed", False)):
        risks.append("watch: low-evidence salience fallback was suppressed")
    if intent == "purpose-continuity" and lifecycle_summary.get("core_capsules", 0) == 0:
        risks.append("watch: no lifecycle core anchors are available for purpose continuity")
    if cold_map_summary is not None and not cold_map_summary.get("within_budget", False):
        risks.append("watch: cold map exceeds requested budget; narrow the query")
    if intent in {"distant-memory", "retention-safety"} and cold_map_summary is not None:
        if int(cold_map_summary.get("matched_capsules", 0)) == 0:
            risks.append("watch: distant memory did not match; avoid broad cold rereads")
    return risks


def _status(
    intent: str,
    plan: Any,
    lifecycle_summary: dict[str, Any],
    cold_map_summary: dict[str, Any] | None,
) -> str:
    visible = int(plan.diagnostics.get("visible_capsules", 0))
    if intent == "purpose-continuity" and lifecycle_summary.get("core_capsules", 0) == 0 and visible == 0:
        return "watch"
    if intent in {"distant-memory", "retention-safety"}:
        if cold_map_summary and cold_map_summary.get("matched_capsules", 0) > 0:
            return "pass" if cold_map_summary.get("within_budget", False) else "watch"
        return "watch" if visible else "fail"
    if visible == 0 and bool(plan.diagnostics.get("low_evidence_fallback_suppressed", False)):
        return "watch"
    return "pass"


def _actions(
    *,
    query: str,
    scope: str,
    intent: str,
    include_global: bool,
    include_hot: bool,
    budgets: list[int],
    plan: Any,
    cold_map_summary: dict[str, Any] | None,
) -> list[RecallPolicyAction]:
    no_global = " --no-global" if not include_global else ""
    no_hot = " --no-hot" if not include_hot else ""
    budget_arg = ",".join(str(item) for item in budgets)
    query_hint = _shell_hint(query)
    actions: list[RecallPolicyAction] = []
    if intent == "purpose-continuity":
        actions.append(
            RecallPolicyAction(
                "purpose-check",
                "gate",
                "verify stable goal visibility before treating recall as aligned with Ara's long-running purpose",
                f"python -m ara_memory purpose-check --scope {scope} --repair-hot",
            )
        )
    if intent in {"distant-memory", "retention-safety"}:
        actions.append(
            RecallPolicyAction(
                "cold-map",
                "use",
                "locate distant memory with redacted group cues instead of rendering cold bodies",
                f'python -m ara_memory cold-map "{query_hint}" --scope {scope} --budget 700',
            )
        )
    if intent == "retention-safety":
        actions.append(
            RecallPolicyAction(
                "retention-cycle",
                "gate",
                "backup, cold export, prune-plan, and shadow-prune must pass before any live pruning",
                f'python -m ara_memory retention-cycle --scope {scope} --query "{query_hint}"',
            )
        )
    if intent == "working-context":
        actions.append(
            RecallPolicyAction(
                "working-memory",
                "use",
                "project only cue-relevant memory into the next action before rendering a full pack",
                f'python -m ara_memory working-memory "{query_hint}" --scope {scope}{no_global}{no_hot}',
            )
        )
    if int(plan.diagnostics.get("visible_capsules", 0)) > 0:
        actions.append(
            RecallPolicyAction(
                "recall-context",
                "optional" if intent in {"distant-memory", "retention-safety"} else "use",
                "render the bounded active recall pack only when the current task needs this evidence",
                f'python -m ara_memory recall-context "{query_hint}" --scope {scope} --budgets {budget_arg}{no_global}{no_hot}',
            )
        )
    elif not actions:
        actions.append(
            RecallPolicyAction(
                "current-evidence",
                "use",
                "no visible memory evidence matched; inspect current files, logs, or user-provided context",
            )
        )
    if cold_map_summary and int(cold_map_summary.get("matched_raw_tokens", 0)) > 0:
        actions.append(
            RecallPolicyAction(
                "avoid-raw-cold-reread",
                "guard",
                "matched cold bodies are represented by a compact map, not by raw context injection",
            )
        )
    return actions


def _shell_hint(value: str) -> str:
    return str(value).replace('"', "'")[:180]
