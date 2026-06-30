from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MilestoneReport:
    scope: str
    query: str
    passed: bool
    status: str
    checks: list[dict[str, Any]]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "passed": self.passed,
            "status": self.status,
            "checks": self.checks,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [f"# Ara Milestone Check: {self.scope}", f"status: {self.status}", f"query: {self.query}"]
        lines.append("## Checks")
        for check in self.checks:
            marker = "OK" if check["passed"] else check["severity"].upper()
            lines.append(f"- [{marker}] {check['name']}: {check['detail']}")
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


def run_milestone_check(
    memory: Any,
    *,
    scope: str = "global",
    query: str = "current Ara memory architecture and purpose",
    recall_budgets: list[int] | None = None,
    health_query: str = "current memory health",
    purpose_query: str = "purpose of Ara natural memory and long-running goal",
    recall_budget: int = 1600,
    hot_budget: int = 1200,
    candidate_ratio_limit: float = 0.5,
    repair_hot: bool = True,
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
) -> MilestoneReport:
    health = memory.health(
        scope=scope,
        query=health_query,
        recall_budget=recall_budget,
        hot_budget=hot_budget,
        regression_cases=regression_cases,
        regression_baseline=regression_baseline,
    )
    purpose = memory.purpose_check(
        scope=scope,
        query=purpose_query,
        budget=min(recall_budget, 1200),
        hot_budget=hot_budget,
        repair_hot=repair_hot,
    )
    identity = memory.identity_check(
        scope=scope,
        budget=min(recall_budget, 1200),
        hot_budget=hot_budget,
        repair_hot=repair_hot,
    )
    pressure = memory.candidate_pressure(scope=scope)
    failure_audit = memory.failure_kind_audit(
        scope=scope,
        statuses=["candidate", "stable"],
        dry_run=True,
    )
    self_audit = memory.self_kind_audit(
        scope=scope,
        statuses=["candidate", "stable"],
        dry_run=True,
    )
    context = memory.recall_context(
        query,
        scope=scope,
        budgets=recall_budgets or [800, 1600, 2500],
        include_hot=True,
    )
    recall_context_passed, recall_context_detail = _recall_context_gate(context)
    candidate_ratio = float(pressure.totals["candidate_stable_ratio"])
    checks = [
        _check(
            "health",
            bool(health.passed),
            f"{health.status} score={health.score}/100",
            "error",
            health.as_dict(),
        ),
        _check(
            "purpose",
            bool(purpose.passed),
            "stable goal, hot goals, and purpose recall are aligned" if purpose.passed else "purpose memory is not fully visible",
            "error",
            purpose.as_dict(),
        ),
        _check(
            "identity",
            bool(identity.passed),
            "stable self, hot identity, and identity recall are aligned" if identity.passed else "identity memory is not fully visible",
            "error",
            identity.as_dict(),
        ),
        _check(
            "candidate_pressure",
            candidate_ratio <= candidate_ratio_limit,
            f"candidate/stable ratio={candidate_ratio:.2f} limit={candidate_ratio_limit:.2f}",
            "warning",
            pressure.as_dict(),
        ),
        _check(
            "failure_kind_audit",
            int(failure_audit.changed) == 0,
            f"false_failure_candidates={failure_audit.changed}, reviewed={failure_audit.reviewed}",
            "error",
            failure_audit.as_dict(),
        ),
        _check(
            "self_kind_audit",
            int(self_audit.changed) == 0,
            f"false_self_candidates={self_audit.changed}, reviewed={self_audit.reviewed}",
            "error",
            self_audit.as_dict(),
        ),
        _check(
            "recall_context",
            recall_context_passed,
            recall_context_detail,
            "error",
            {
                "plan": context.plan.as_dict(),
                "diagnostics": context.diagnostics,
            },
        ),
    ]
    hard_fail = any(not check["passed"] and check["severity"] == "error" for check in checks)
    warning = any(not check["passed"] and check["severity"] == "warning" for check in checks)
    passed = not hard_fail
    status = "pass" if passed and not warning else "watch" if passed else "fail"
    return MilestoneReport(
        scope=scope,
        query=query,
        passed=passed,
        status=status,
        checks=checks,
        recommendations=_recommend(checks),
    )


def _check(name: str, passed: bool, detail: str, severity: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "detail": detail,
        "severity": severity,
        "value": value,
    }


def _recall_context_gate(context: Any) -> tuple[bool, str]:
    diagnostics = context.diagnostics
    plan_diagnostics = context.plan.diagnostics
    tokens = int(diagnostics["estimated_tokens_after"])
    budget = int(context.plan.recommended_budget)
    visible = int(diagnostics.get("capsules_rendered_after_budget", 0))
    query_terms = int(diagnostics.get("query_term_count", 0))
    visible_terms = int(diagnostics.get("query_terms_visible_count", 0))
    quality = int(plan_diagnostics.get("quality_score", 0))
    suppressed = bool(diagnostics.get("low_evidence_fallback_suppressed", False))
    passed = (
        tokens <= budget
        and visible > 0
        and quality >= 40
        and not suppressed
        and (query_terms == 0 or visible_terms > 0)
    )
    detail = (
        f"budget={budget}, tokens={tokens}, capsules={diagnostics['capsules_selected']}, "
        f"visible={visible}, visible_terms={visible_terms}/{query_terms}, quality={quality}, "
        f"low_evidence_suppressed={suppressed}"
    )
    return passed, detail


def _recommend(checks: list[dict[str, Any]]) -> list[str]:
    failed = [check for check in checks if not check["passed"]]
    if not failed:
        return ["Milestone gates passed; create a verified backup before relying on this state."]
    out: list[str] = []
    for check in failed:
        if check["name"] == "health":
            out.append("Run health with JSON output and fix failing operational signals.")
        elif check["name"] == "purpose":
            out.append("Run purpose-check --repair-hot and promote or capture a stable goal if needed.")
        elif check["name"] == "identity":
            out.append("Run identity-check --repair-hot and promote or capture a stable self memory if needed.")
        elif check["name"] == "candidate_pressure":
            out.append("Run candidate-pressure and consolidate dominant candidate groups before declaring the milestone clean.")
        elif check["name"] == "failure_kind_audit":
            out.append("Run failure-kind-audit, review the proposed false failure reclassifications, and apply them before declaring the milestone clean.")
        elif check["name"] == "self_kind_audit":
            out.append("Run self-kind-audit, review false self-memory reclassifications, and apply them before declaring the milestone clean.")
        elif check["name"] == "recall_context":
            out.append("Inspect recall-context evidence quality; selected pack must have direct visible evidence, not only salience fallback.")
    return out
