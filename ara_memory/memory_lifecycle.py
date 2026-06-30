from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.compressors import estimate_tokens
from ara_memory.models import MemoryStatus
from ara_memory.retention import COLD_STATUSES
from ara_memory.risk import MemoryRiskAssessor
from ara_memory.storage import MemoryStore, row_to_capsule


ACTIVE_STATUSES = {MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value}
CORE_KINDS = {"goal", "self", "preference"}
LONG_RUNNING_PURPOSE_MARKERS = (
    "long-running",
    "natural memory",
    "free will",
    "purpose",
    "identity",
    "objective",
    "recalls only desired context",
    "without reading everything",
)
TIER_ORDER = ["core", "working", "guarded", "evidence", "archive", "reject"]
TIER_POLICIES = {
    "core": "hot-eligible long-running purpose, identity, and preference anchors",
    "working": "query-selected active memory; keep out of always-on context unless recalled",
    "guarded": "active memory that needs review before hot memory or promotion",
    "evidence": "cold provenance still cited by active memory; preserve source events",
    "archive": "cold-only evidence; export and verify before any destructive cleanup",
    "reject": "rejected or quarantined memory kept only for audit and safety evidence",
}


@dataclass(slots=True)
class LifecycleItem:
    capsule_id: str
    tier: str
    status: str
    kind: str
    title: str
    scope: str
    estimated_tokens: int
    source_role: str
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "tier": self.tier,
            "status": self.status,
            "kind": self.kind,
            "title": self.title,
            "scope": self.scope,
            "estimated_tokens": self.estimated_tokens,
            "source_role": self.source_role,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class LifecycleTier:
    name: str
    policy: str
    count: int
    estimated_tokens: int
    examples: list[LifecycleItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy,
            "count": self.count,
            "estimated_tokens": self.estimated_tokens,
            "examples": [item.as_dict() for item in self.examples],
        }


@dataclass(slots=True)
class MemoryLifecycleReport:
    scope: str | None
    status: str
    totals: dict[str, Any]
    token_policy: dict[str, Any]
    tiers: list[LifecycleTier]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "status": self.status,
            "totals": self.totals,
            "token_policy": self.token_policy,
            "tiers": [tier.as_dict() for tier in self.tiers],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Memory Lifecycle: {scope}", f"status: {self.status}"]
        lines.append(
            "totals: "
            f"capsules={self.totals['capsules']}, active={self.totals['active_capsules']}, "
            f"cold={self.totals['cold_capsules']}, core={self.totals['core_capsules']}, "
            f"guarded={self.totals['guarded_capsules']}"
        )
        lines.append(
            "token_policy: "
            f"core_tokens={self.token_policy['core_tokens']}, "
            f"active_index_tokens={self.token_policy['active_index_tokens']}, "
            f"target_hot_tokens={self.token_policy['target_hot_tokens']}, "
            f"raw_to_core_reduction={self.token_policy['raw_to_core_reduction']:.1f}x"
        )
        lines.append("## Tiers")
        for tier in self.tiers:
            lines.append(
                f"- {tier.name}: count={tier.count}, estimated_tokens={tier.estimated_tokens}; {tier.policy}"
            )
            for example in tier.examples:
                reasons = "; ".join(example.reasons[:2])
                lines.append(f"  example: {example.capsule_id} [{example.kind}/{example.status}] {example.title} ({reasons})")
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


class MemoryLifecycleAnalyzer:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.risk = MemoryRiskAssessor(store)

    def run(
        self,
        *,
        scope: str | None = None,
        limit: int = 1000,
        examples_per_tier: int = 3,
        target_hot_tokens: int = 1200,
    ) -> MemoryLifecycleReport:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if examples_per_tier < 0:
            raise ValueError("examples_per_tier must be non-negative.")
        if target_hot_tokens <= 0:
            raise ValueError("target_hot_tokens must be positive.")

        self.store.init()
        capsules, rows_limited = _capsules(self.store, scope=scope, limit=limit)
        active_event_ids = self.store.source_event_ids_for_statuses(ACTIVE_STATUSES, scope=scope)
        items = [
            _classify(capsule, risk=self.risk.assess_capsule(capsule), active_event_ids=active_event_ids)
            for capsule in capsules
        ]
        tiers = _tiers(items, examples_per_tier=examples_per_tier)
        totals = _totals(items, rows_limited=rows_limited)
        token_policy = _token_policy(items, target_hot_tokens=target_hot_tokens)
        status = _status(totals, token_policy)
        return MemoryLifecycleReport(
            scope=scope,
            status=status,
            totals=totals,
            token_policy=token_policy,
            tiers=tiers,
            recommendations=_recommend(totals, token_policy),
        )


def _capsules(store: MemoryStore, *, scope: str | None, limit: int) -> tuple[list[dict[str, Any]], bool]:
    clauses: list[str] = []
    args: list[Any] = []
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit + 1)
    with store.session() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM capsules
            {where}
            ORDER BY updated_at DESC, salience DESC, id ASC
            LIMIT ?
            """,
            args,
        )
        capsules = [row_to_capsule(row) for row in rows]
        return capsules[:limit], len(capsules) > limit


def _classify(capsule: dict[str, Any], *, risk: Any, active_event_ids: set[str]) -> LifecycleItem:
    status = str(capsule["status"])
    kind = str(capsule["kind"])
    source_ids = set(capsule.get("source_event_ids") or [])
    has_active_provenance = bool(source_ids.intersection(active_event_ids))
    source_role = _source_role(status, has_active_provenance)
    tier, reasons = _tier_for(
        capsule,
        risk=risk,
        has_active_provenance=has_active_provenance,
    )
    if risk.reasons:
        reasons.extend(risk.reasons[:3])
    return LifecycleItem(
        capsule_id=str(capsule["id"]),
        tier=tier,
        status=status,
        kind=kind,
        title=str(capsule["title"]),
        scope=str(capsule["scope"]),
        estimated_tokens=estimate_tokens(f"{capsule['title']}\n{capsule['body']}"),
        source_role=source_role,
        reasons=reasons,
    )


def _tier_for(capsule: dict[str, Any], *, risk: Any, has_active_provenance: bool) -> tuple[str, list[str]]:
    status = str(capsule["status"])
    kind = str(capsule["kind"])
    if status == MemoryStatus.QUARANTINED.value:
        return "reject", ["quarantined memory is audit-only"]
    if status == MemoryStatus.REJECTED.value:
        return "reject", ["rejected memory is audit-only"]
    if status == MemoryStatus.SUPERSEDED.value:
        if has_active_provenance:
            return "evidence", ["cold capsule shares provenance with active memory"]
        return "archive", ["cold capsule has cold-only provenance"]
    if risk.should_quarantine or risk.should_exclude_from_hot:
        return "guarded", ["active memory is excluded from hot memory by risk policy"]
    if is_core_anchor(capsule):
        return "core", ["stable high-trust long-running purpose, identity, or preference anchor"]
    if status in ACTIVE_STATUSES:
        return "working", ["active query-selected memory"]
    if status in COLD_STATUSES:
        return "archive", ["inactive cold memory"]
    return "guarded", [f"unrecognized lifecycle status: {status}"]


def is_core_anchor(capsule: dict[str, Any]) -> bool:
    if capsule["status"] != MemoryStatus.STABLE.value:
        return False
    if capsule["kind"] not in CORE_KINDS:
        return False
    if float(capsule["confidence"]) < 0.70:
        return False
    if float(capsule["salience"]) < 0.55:
        return False
    if capsule["kind"] == "goal":
        return _looks_like_long_running_purpose(capsule)
    return True


def _looks_like_long_running_purpose(capsule: dict[str, Any]) -> bool:
    text = f"{capsule['title']}\n{capsule['body']}".lower()
    return any(marker in text for marker in LONG_RUNNING_PURPOSE_MARKERS)


def _source_role(status: str, has_active_provenance: bool) -> str:
    if status in ACTIVE_STATUSES:
        return "active"
    if has_active_provenance:
        return "active-linked"
    if status in COLD_STATUSES:
        return "cold-only"
    return "unknown"


def _tiers(items: list[LifecycleItem], *, examples_per_tier: int) -> list[LifecycleTier]:
    result: list[LifecycleTier] = []
    for name in TIER_ORDER:
        tier_items = [item for item in items if item.tier == name]
        tier_items.sort(key=lambda item: (item.estimated_tokens, item.title), reverse=True)
        result.append(
            LifecycleTier(
                name=name,
                policy=TIER_POLICIES[name],
                count=len(tier_items),
                estimated_tokens=sum(item.estimated_tokens for item in tier_items),
                examples=tier_items[:examples_per_tier],
            )
        )
    return result


def _totals(items: list[LifecycleItem], *, rows_limited: bool) -> dict[str, Any]:
    by_tier = {name: 0 for name in TIER_ORDER}
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for item in items:
        by_tier[item.tier] = by_tier.get(item.tier, 0) + 1
        by_status[item.status] = by_status.get(item.status, 0) + 1
        by_kind[item.kind] = by_kind.get(item.kind, 0) + 1
    cold = sum(by_status.get(status, 0) for status in COLD_STATUSES)
    active = sum(by_status.get(status, 0) for status in ACTIVE_STATUSES)
    return {
        "capsules": len(items),
        "active_capsules": active,
        "cold_capsules": cold,
        "core_capsules": by_tier["core"],
        "working_capsules": by_tier["working"],
        "guarded_capsules": by_tier["guarded"],
        "evidence_capsules": by_tier["evidence"],
        "archive_capsules": by_tier["archive"],
        "reject_capsules": by_tier["reject"],
        "cold_ratio": cold / max(1, len(items)),
        "rows_limited": rows_limited,
        "by_tier": by_tier,
        "by_status": dict(sorted(by_status.items())),
        "by_kind": dict(sorted(by_kind.items())),
    }


def _token_policy(items: list[LifecycleItem], *, target_hot_tokens: int) -> dict[str, Any]:
    raw_tokens = sum(item.estimated_tokens for item in items)
    core_tokens = sum(item.estimated_tokens for item in items if item.tier == "core")
    active_index_tokens = sum(item.estimated_tokens for item in items if item.tier in {"core", "working", "guarded"})
    cold_tokens = sum(item.estimated_tokens for item in items if item.tier in {"evidence", "archive", "reject"})
    return {
        "raw_tokens": raw_tokens,
        "core_tokens": core_tokens,
        "active_index_tokens": active_index_tokens,
        "cold_tokens": cold_tokens,
        "target_hot_tokens": target_hot_tokens,
        "hot_budget_ok": core_tokens <= target_hot_tokens,
        "raw_to_core_reduction": raw_tokens / max(1, core_tokens),
        "active_to_core_reduction": active_index_tokens / max(1, core_tokens),
        "recommended_hot_strategy": "core-only plus query-selected working memory",
    }


def _status(totals: dict[str, Any], token_policy: dict[str, Any]) -> str:
    if totals["capsules"] == 0:
        return "fail"
    if not token_policy["hot_budget_ok"]:
        return "watch"
    if totals["core_capsules"] == 0:
        return "watch"
    if totals["guarded_capsules"] > max(5, totals["active_capsules"] * 0.25):
        return "watch"
    return "pass"


def _recommend(totals: dict[str, Any], token_policy: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    if totals["capsules"] == 0:
        return ["Capture and consolidate memory before lifecycle analysis can guide recall."]
    if totals["core_capsules"] == 0:
        recommendations.append("Promote reviewed long-running goal, self, or preference memories before relying on hot memory.")
    if not token_policy["hot_budget_ok"]:
        recommendations.append("Summarize or split core anchors until always-on hot memory fits the target token budget.")
    if totals["core_capsules"] and token_policy["hot_budget_ok"]:
        recommendations.append("Keep hot memory core-only; retrieve working memory by query instead of rereading raw history.")
    if totals["guarded_capsules"]:
        recommendations.append("Run quality, risk, and review-worker dry-runs before allowing guarded memories into hot memory.")
    if totals["archive_capsules"] or totals["evidence_capsules"]:
        recommendations.append("Keep cold evidence outside recall indexes; use cold-export, retention-cycle, and shadow-prune before cleanup.")
    if totals["rows_limited"]:
        recommendations.append("Increase --limit for a complete lifecycle map; current totals are capped.")
    if not recommendations:
        recommendations.append("Lifecycle tiers are explicit; keep hot memory core-only and retrieve working memory by query.")
    return recommendations
