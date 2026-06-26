from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_PURPOSE_QUERY = "purpose of Ara natural memory and long-running goal"


@dataclass(slots=True)
class PurposeReport:
    scope: str
    query: str
    passed: bool
    stable_goals: int
    candidate_goals: int
    hot_has_goals: bool
    recall_has_goals: bool
    recall_tokens: int
    selected_goal_ids: list[str]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "passed": self.passed,
            "stable_goals": self.stable_goals,
            "candidate_goals": self.candidate_goals,
            "hot_has_goals": self.hot_has_goals,
            "recall_has_goals": self.recall_has_goals,
            "recall_tokens": self.recall_tokens,
            "selected_goal_ids": self.selected_goal_ids,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        status = "pass" if self.passed else "watch"
        lines = [
            f"# Ara Purpose Check: {self.scope}",
            f"status: {status}",
            f"query: {self.query}",
            f"stable_goals: {self.stable_goals}",
            f"candidate_goals: {self.candidate_goals}",
            f"hot_has_goals: {self.hot_has_goals}",
            f"recall_has_goals: {self.recall_has_goals}",
            f"recall_tokens: {self.recall_tokens}",
            "## Recommendations",
        ]
        lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def run_purpose_check(
    memory: Any,
    *,
    scope: str = "global",
    query: str = DEFAULT_PURPOSE_QUERY,
    budget: int = 1000,
    hot_budget: int = 1200,
    include_global: bool = True,
    repair_hot: bool = False,
) -> PurposeReport:
    memory.init()
    stable_goals = memory.list_capsules(scope=scope, status="stable", kind="goal", limit=20)
    candidate_goals = memory.list_capsules(scope=scope, status="candidate", kind="goal", limit=20)
    if include_global and scope != "global":
        stable_goals.extend(memory.list_capsules(scope="global", status="stable", kind="goal", limit=20))
        candidate_goals.extend(memory.list_capsules(scope="global", status="candidate", kind="goal", limit=20))
    hot = memory.read_hot(scope=scope) or memory.build_hot(scope=scope, budget=hot_budget)
    hot_has_goals = _hot_has_goals(hot.text)
    if repair_hot and stable_goals and not hot_has_goals:
        hot = memory.build_hot(scope=scope, budget=hot_budget)
        hot_has_goals = _hot_has_goals(hot.text)
    recall = memory.recall_result(
        query,
        scope=scope,
        budget=budget,
        include_global=include_global,
        include_hot=True,
    )
    selected_goal_ids = [
        cap["id"]
        for cap in stable_goals + candidate_goals
        if cap["id"] in set(recall.diagnostics["selected_capsule_ids"])
    ]
    recall_has_goals = bool(selected_goal_ids) or _pack_has_goal_section(recall.pack)
    passed = bool(stable_goals) and hot_has_goals and recall_has_goals
    recommendations = _recommend(
        passed=passed,
        stable_goals=len(stable_goals),
        candidate_goals=len(candidate_goals),
        hot_has_goals=hot_has_goals,
        recall_has_goals=recall_has_goals,
    )
    return PurposeReport(
        scope=scope,
        query=query,
        passed=passed,
        stable_goals=len(stable_goals),
        candidate_goals=len(candidate_goals),
        hot_has_goals=hot_has_goals,
        recall_has_goals=recall_has_goals,
        recall_tokens=int(recall.diagnostics["estimated_tokens_after"]),
        selected_goal_ids=selected_goal_ids,
        recommendations=recommendations,
    )


def _hot_has_goals(text: str) -> bool:
    marker = "## Active Goals"
    if marker not in text:
        return False
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return "- None." not in section and bool(section.strip())


def _pack_has_goal_section(text: str) -> bool:
    marker = "## Active Goals / Intent"
    if marker not in text:
        return False
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return "- None found." not in section and bool(section.strip())


def _recommend(
    *,
    passed: bool,
    stable_goals: int,
    candidate_goals: int,
    hot_has_goals: bool,
    recall_has_goals: bool,
) -> list[str]:
    if passed:
        return ["Purpose memory is visible in stable goals, hot memory, and purpose recall."]
    out: list[str] = []
    if stable_goals == 0 and candidate_goals > 0:
        out.append("Promote or consolidate the strongest goal candidate before relying on purpose recall.")
    elif stable_goals == 0:
        out.append("Capture an explicit goal memory for the long-running purpose of this scope.")
    if not hot_has_goals:
        out.append("Rebuild hot memory after goal capture or promotion.")
    if not recall_has_goals:
        out.append("Inspect recall ranking for purpose queries; the goal capsule is not selected.")
    return out
