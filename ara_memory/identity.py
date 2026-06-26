from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_IDENTITY_QUERY = "Ara identity judgment principles free will coding partner"


@dataclass(slots=True)
class IdentityReport:
    scope: str
    query: str
    passed: bool
    stable_self: int
    candidate_self: int
    hot_has_identity: bool
    recall_has_identity: bool
    selected_self_ids: list[str]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "query": self.query,
            "passed": self.passed,
            "stable_self": self.stable_self,
            "candidate_self": self.candidate_self,
            "hot_has_identity": self.hot_has_identity,
            "recall_has_identity": self.recall_has_identity,
            "selected_self_ids": self.selected_self_ids,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Identity Check: {self.scope}",
            f"passed: {self.passed}",
            f"stable_self: {self.stable_self}",
            f"candidate_self: {self.candidate_self}",
            f"hot_has_identity: {self.hot_has_identity}",
            f"recall_has_identity: {self.recall_has_identity}",
            "## Recommendations",
        ]
        lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def run_identity_check(
    memory: Any,
    *,
    scope: str = "global",
    query: str = DEFAULT_IDENTITY_QUERY,
    budget: int = 1000,
    hot_budget: int = 1200,
    include_global: bool = True,
    repair_hot: bool = False,
) -> IdentityReport:
    hot = memory.read_hot(scope=scope) or memory.build_hot(scope=scope, budget=hot_budget)
    stable_self = memory.list_capsules(scope=scope, status="stable", kind="self", limit=20)
    candidate_self = memory.list_capsules(scope=scope, status="candidate", kind="self", limit=20)
    if include_global and scope != "global":
        stable_self.extend(memory.list_capsules(scope="global", status="stable", kind="self", limit=20))
        candidate_self.extend(memory.list_capsules(scope="global", status="candidate", kind="self", limit=20))
    hot_has_identity = _hot_has_identity(hot.text)
    if repair_hot and stable_self and not hot_has_identity:
        hot = memory.build_hot(scope=scope, budget=hot_budget)
        hot_has_identity = _hot_has_identity(hot.text)
    recall = memory.recall_result(query, scope=scope, budget=budget, include_global=include_global, include_hot=True)
    selected = [
        cap["id"]
        for cap in stable_self + candidate_self
        if cap["id"] in set(recall.diagnostics["selected_capsule_ids"])
    ]
    recall_has_identity = bool(selected) or _pack_has_identity_section(recall.pack)
    passed = bool(stable_self) and hot_has_identity and recall_has_identity
    return IdentityReport(
        scope=scope,
        query=query,
        passed=passed,
        stable_self=len(stable_self),
        candidate_self=len(candidate_self),
        hot_has_identity=hot_has_identity,
        recall_has_identity=recall_has_identity,
        selected_self_ids=selected,
        recommendations=_recommend(
            stable_self=len(stable_self),
            candidate_self=len(candidate_self),
            hot_has_identity=hot_has_identity,
            recall_has_identity=recall_has_identity,
        ),
    )


def _hot_has_identity(text: str) -> bool:
    lowered = text.lower()
    if "## stable identity / preferences" not in lowered:
        return False
    return "[self]" in lowered or "identity" in lowered or "judgment" in lowered or "free will" in lowered


def _pack_has_identity_section(text: str) -> bool:
    lowered = text.lower()
    return "## stable / relational memory" in lowered and (
        "[self/" in lowered or "identity" in lowered or "judgment" in lowered or "free will" in lowered
    )


def _recommend(
    *,
    stable_self: int,
    candidate_self: int,
    hot_has_identity: bool,
    recall_has_identity: bool,
) -> list[str]:
    if stable_self and hot_has_identity and recall_has_identity:
        return ["Identity memory is visible in stable self memory, hot memory, and identity recall."]
    out: list[str] = []
    if stable_self == 0 and candidate_self > 0:
        out.append("Promote or consolidate the strongest self candidate before relying on identity recall.")
    elif stable_self == 0:
        out.append("Capture an explicit self memory for Ara's durable identity and judgment principles.")
    if not hot_has_identity:
        out.append("Rebuild hot memory after self-memory capture or promotion.")
    if not recall_has_identity:
        out.append("Inspect recall ranking for identity queries; the self capsule is not selected.")
    return out
