from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ara_memory.compressors import estimate_tokens
from ara_memory.models import MemoryStatus
from ara_memory.retention import COLD_STATUSES
from ara_memory.storage import MemoryStore


ACTIVE_STATUSES = {
    MemoryStatus.CANDIDATE.value,
    MemoryStatus.STABLE.value,
}


@dataclass(slots=True)
class ColdGroup:
    status: str
    kind: str
    pattern: str
    count: int
    tier_counts: dict[str, int]
    source_roles: dict[str, int]
    recommended_action: str
    source_events: int
    protected_source_events: int
    prunable_source_events: int
    oldest_updated_at: str
    newest_updated_at: str
    examples: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "pattern": self.pattern,
            "count": self.count,
            "tier_counts": self.tier_counts,
            "source_roles": self.source_roles,
            "recommended_action": self.recommended_action,
            "source_events": self.source_events,
            "protected_source_events": self.protected_source_events,
            "prunable_source_events": self.prunable_source_events,
            "oldest_updated_at": self.oldest_updated_at,
            "newest_updated_at": self.newest_updated_at,
            "examples": self.examples,
        }


@dataclass(slots=True)
class ColdLifecycleTier:
    name: str
    policy: str
    count: int
    estimated_tokens: int
    source_events: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": self.policy,
            "count": self.count,
            "estimated_tokens": self.estimated_tokens,
            "source_events": self.source_events,
        }


@dataclass(slots=True)
class ActivePinGroup:
    status: str
    kind: str
    pattern: str
    count: int
    source_events: int
    examples: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "kind": self.kind,
            "pattern": self.pattern,
            "count": self.count,
            "source_events": self.source_events,
            "examples": self.examples,
        }


@dataclass(slots=True)
class ColdStewardshipReport:
    scope: str | None
    status: str
    totals: dict[str, Any]
    groups: list[ColdGroup]
    lifecycle_tiers: list[ColdLifecycleTier]
    active_pins: list[ActivePinGroup]
    latest_retention_cycle: dict[str, Any] | None
    cycle_evidence: dict[str, Any]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "status": self.status,
            "totals": self.totals,
            "groups": [group.as_dict() for group in self.groups],
            "lifecycle_tiers": [tier.as_dict() for tier in self.lifecycle_tiers],
            "active_pins": [pin.as_dict() for pin in self.active_pins],
            "latest_retention_cycle": self.latest_retention_cycle,
            "cycle_evidence": self.cycle_evidence,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Cold Stewardship: {scope}", f"status: {self.status}"]
        lines.append(
            "totals: "
            f"capsules={self.totals['capsules']}, cold={self.totals['cold_capsules']} "
            f"({self.totals['cold_ratio']:.0%}), "
            f"protected_source_events={self.totals['protected_source_events']}, "
            f"prunable_source_events={self.totals['prunable_source_events']}, "
            f"evidence={self.totals['evidence_capsules']}, "
            f"archive={self.totals['archive_capsules']}, "
            f"reject={self.totals['reject_capsules']}"
        )
        if self.latest_retention_cycle:
            cycle = self.latest_retention_cycle
            lines.append(
                "latest_retention_cycle: "
                f"passed={cycle.get('passed')}, age_hours={cycle.get('age_hours')}, "
                f"path={cycle.get('path')}"
            )
        else:
            lines.append("latest_retention_cycle: none")
        if self.cycle_evidence:
            lines.append(
                "cycle_evidence: "
                f"fresh={self.cycle_evidence.get('fresh')}, "
                f"matches_current={self.cycle_evidence.get('matches_current')}, "
                f"shadow_events_preserved={self.cycle_evidence.get('shadow_events_preserved')}"
            )
        lines.append("## Cold Lifecycle Tiers")
        for tier in self.lifecycle_tiers:
            lines.append(
                f"- {tier.name}: count={tier.count}, estimated_tokens={tier.estimated_tokens}, "
                f"source_events={tier.source_events}; {tier.policy}"
            )
        lines.append("## Active Provenance Pins")
        if not self.active_pins:
            lines.append("- None")
        for pin in self.active_pins:
            pattern = pin.pattern.rstrip(":")
            lines.append(
                f"- {pin.status}/{pin.kind}/{pattern}: "
                f"count={pin.count}, source_events={pin.source_events}"
            )
            for example in pin.examples:
                lines.append(f"  example: {example['id']} {example['title']}")
        lines.append("## Top Cold Groups")
        if not self.groups:
            lines.append("- None")
        for group in self.groups:
            pattern = group.pattern.rstrip(":")
            lines.append(
                f"- {group.status}/{group.kind}/{pattern}: "
                f"count={group.count}, action={group.recommended_action}, "
                f"tiers={_format_counts(group.tier_counts)}, "
                f"protected_events={group.protected_source_events}, "
                f"prunable_events={group.prunable_source_events}"
            )
            for example in group.examples:
                lines.append(
                    f"  example: {example['id']} [{example['tier']}/{example['source_role']}] {example['title']}"
                )
        lines.append("## Recommendations")
        for item in self.recommendations:
            lines.append(f"- {item}")
        return "\n".join(lines)


class ColdStewardshipAnalyzer:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str | None = None,
        group_limit: int = 12,
        examples_per_group: int = 2,
        max_cycle_age_hours: float = 72.0,
    ) -> ColdStewardshipReport:
        self.store.init()
        rows = _cold_rows(self.store, scope=scope)
        active_event_ids = _active_event_ids(self.store, scope=scope)
        rows = [_annotate_cold_row(row, active_event_ids=active_event_ids) for row in rows]
        cold_source_events = sorted({event_id for row in rows for event_id in row["source_event_ids"]})
        protected_source_events = sorted(set(cold_source_events).intersection(active_event_ids))
        prunable_source_events = sorted(set(cold_source_events) - set(protected_source_events))
        lifecycle_tiers = _lifecycle_tiers(rows)
        active_pins = _active_pin_groups(
            _active_rows(self.store, scope=scope),
            cold_source_events=set(cold_source_events),
            group_limit=group_limit,
            examples_per_group=examples_per_group,
        )
        totals = _totals(
            self.store,
            scope=scope,
            cold_count=len(rows),
            cold_source_events=cold_source_events,
            protected_source_events=protected_source_events,
            prunable_source_events=prunable_source_events,
            lifecycle_tiers=lifecycle_tiers,
        )
        groups = _groups(
            rows,
            active_event_ids=active_event_ids,
            group_limit=group_limit,
            examples_per_group=examples_per_group,
        )
        latest_cycle = _latest_retention_cycle(self.store.root, scope=scope)
        cycle_evidence = _cycle_evidence(
            totals,
            latest_cycle,
            max_cycle_age_hours=max_cycle_age_hours,
        )
        status = _status(totals, latest_cycle, cycle_evidence)
        return ColdStewardshipReport(
            scope=scope,
            status=status,
            totals=totals,
            groups=groups,
            lifecycle_tiers=lifecycle_tiers,
            active_pins=active_pins,
            latest_retention_cycle=latest_cycle,
            cycle_evidence=cycle_evidence,
            recommendations=_recommend(totals, groups, lifecycle_tiers, active_pins, latest_cycle, cycle_evidence),
        )


def _cold_rows(store: MemoryStore, *, scope: str | None) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in COLD_STATUSES)
    clauses = [f"status IN ({placeholders})"]
    args: list[Any] = [*COLD_STATUSES]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    with store.session() as conn:
        rows = conn.execute(
            f"""
            SELECT id, kind, status, title, body, updated_at, source_event_ids_json
            FROM capsules
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at ASC, id ASC
            """,
            args,
        )
        return [_row_to_cold(row) for row in rows]


def _row_to_cold(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "title": row["title"],
        "body": row["body"],
        "updated_at": row["updated_at"],
        "source_event_ids": json.loads(row["source_event_ids_json"]),
    }


def _annotate_cold_row(row: dict[str, Any], *, active_event_ids: set[str]) -> dict[str, Any]:
    source_ids = set(row["source_event_ids"])
    has_active_provenance = bool(source_ids.intersection(active_event_ids))
    annotated = dict(row)
    annotated["tier"] = _cold_tier(str(row["status"]), has_active_provenance=has_active_provenance)
    annotated["source_role"] = "active-linked" if has_active_provenance else "cold-only"
    annotated["estimated_tokens"] = estimate_tokens(f"{row['title']}\n{row.get('body') or ''}")
    return annotated


def _cold_tier(status: str, *, has_active_provenance: bool) -> str:
    if status in {MemoryStatus.REJECTED.value, MemoryStatus.QUARANTINED.value}:
        return "reject"
    if status == MemoryStatus.SUPERSEDED.value and has_active_provenance:
        return "evidence"
    return "archive"


def _active_event_ids(store: MemoryStore, *, scope: str | None) -> set[str]:
    return store.source_event_ids_for_statuses(ACTIVE_STATUSES, scope=scope)


def _active_rows(store: MemoryStore, *, scope: str | None) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    clauses = [f"status IN ({placeholders})"]
    args: list[Any] = [*sorted(ACTIVE_STATUSES)]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    with store.session() as conn:
        rows = conn.execute(
            f"""
            SELECT id, kind, status, title, updated_at, source_event_ids_json
            FROM capsules
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, id ASC
            """,
            args,
        )
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "status": row["status"],
                "title": row["title"],
                "updated_at": row["updated_at"],
                "source_event_ids": json.loads(row["source_event_ids_json"]),
            }
            for row in rows
        ]


def _totals(
    store: MemoryStore,
    *,
    scope: str | None,
    cold_count: int,
    cold_source_events: list[str],
    protected_source_events: list[str],
    prunable_source_events: list[str],
    lifecycle_tiers: list[ColdLifecycleTier],
) -> dict[str, Any]:
    with store.session() as conn:
        if scope:
            capsules = conn.execute("SELECT COUNT(*) FROM capsules WHERE scope = ?", (scope,)).fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM events WHERE scope = ?", (scope,)).fetchone()[0]
        else:
            capsules = conn.execute("SELECT COUNT(*) FROM capsules").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    totals = {
        "scope": scope,
        "events": events,
        "capsules": capsules,
        "cold_capsules": cold_count,
        "cold_ratio": cold_count / max(1, capsules),
        "cold_source_events": len(cold_source_events),
        "protected_source_events": len(protected_source_events),
        "prunable_source_events": len(prunable_source_events),
    }
    for tier in lifecycle_tiers:
        totals[f"{tier.name}_capsules"] = tier.count
        totals[f"{tier.name}_tokens"] = tier.estimated_tokens
    for tier_name in COLD_TIER_POLICIES:
        totals.setdefault(f"{tier_name}_capsules", 0)
        totals.setdefault(f"{tier_name}_tokens", 0)
    totals.update(store.storage_breakdown())
    return totals


def _groups(
    rows: list[dict[str, Any]],
    *,
    active_event_ids: set[str],
    group_limit: int,
    examples_per_group: int,
) -> list[ColdGroup]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["status"], row["kind"], _title_pattern(row["title"]))
        bucket = buckets.setdefault(
            key,
            {
                "rows": [],
                "source_event_ids": set(),
            },
        )
        bucket["rows"].append(row)
        bucket["source_event_ids"].update(row["source_event_ids"])

    groups: list[ColdGroup] = []
    for (status, kind, pattern), bucket in buckets.items():
        group_rows = bucket["rows"]
        source_event_ids = set(bucket["source_event_ids"])
        protected = sorted(source_event_ids.intersection(active_event_ids))
        prunable = sorted(source_event_ids - set(protected))
        updated = [row["updated_at"] for row in group_rows]
        tier_counts = _count_values(group_rows, "tier")
        source_roles = _count_values(group_rows, "source_role")
        examples = [
            {
                "id": row["id"],
                "title": row["title"],
                "updated_at": row["updated_at"],
                "tier": row["tier"],
                "source_role": row["source_role"],
            }
            for row in group_rows[: max(0, examples_per_group)]
        ]
        groups.append(
            ColdGroup(
                status=status,
                kind=kind,
                pattern=pattern,
                count=len(group_rows),
                tier_counts=tier_counts,
                source_roles=source_roles,
                recommended_action=_group_action(tier_counts),
                source_events=len(source_event_ids),
                protected_source_events=len(protected),
                prunable_source_events=len(prunable),
                oldest_updated_at=min(updated) if updated else "",
                newest_updated_at=max(updated) if updated else "",
                examples=examples,
            )
        )
    return sorted(
        groups,
        key=lambda group: (
            -group.count,
            -group.prunable_source_events,
            -group.protected_source_events,
            group.status,
            group.kind,
            group.pattern,
        ),
    )[: max(0, group_limit)]


COLD_TIER_POLICIES = {
    "evidence": "cold provenance linked to active memory; preserve source events and inspect before pruning",
    "archive": "cold-only provenance; export and verify before any destructive cleanup",
    "reject": "rejected or quarantined memory; keep as audit/safety evidence until explicitly reviewed",
}


def _lifecycle_tiers(rows: list[dict[str, Any]]) -> list[ColdLifecycleTier]:
    tiers: list[ColdLifecycleTier] = []
    for name, policy in COLD_TIER_POLICIES.items():
        tier_rows = [row for row in rows if row["tier"] == name]
        source_events = {event_id for row in tier_rows for event_id in row["source_event_ids"]}
        tiers.append(
            ColdLifecycleTier(
                name=name,
                policy=policy,
                count=len(tier_rows),
                estimated_tokens=sum(int(row["estimated_tokens"]) for row in tier_rows),
                source_events=len(source_events),
            )
        )
    return tiers


def _active_pin_groups(
    rows: list[dict[str, Any]],
    *,
    cold_source_events: set[str],
    group_limit: int,
    examples_per_group: int,
) -> list[ActivePinGroup]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        pinned = sorted(set(row["source_event_ids"]).intersection(cold_source_events))
        if not pinned:
            continue
        key = (row["status"], row["kind"], _title_pattern(row["title"]))
        bucket = buckets.setdefault(key, {"rows": [], "source_event_ids": set()})
        bucket["rows"].append(row)
        bucket["source_event_ids"].update(pinned)

    groups: list[ActivePinGroup] = []
    for (status, kind, pattern), bucket in buckets.items():
        group_rows = bucket["rows"]
        examples = [
            {"id": row["id"], "title": row["title"], "updated_at": row["updated_at"]}
            for row in group_rows[: max(0, examples_per_group)]
        ]
        groups.append(
            ActivePinGroup(
                status=status,
                kind=kind,
                pattern=pattern,
                count=len(group_rows),
                source_events=len(bucket["source_event_ids"]),
                examples=examples,
            )
        )
    return sorted(
        groups,
        key=lambda group: (-group.source_events, -group.count, group.status, group.kind, group.pattern),
    )[: max(0, group_limit)]


def _count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _group_action(tier_counts: dict[str, int]) -> str:
    non_zero = {tier for tier, count in tier_counts.items() if count}
    if len(non_zero) > 1:
        return "mixed-review"
    if "evidence" in non_zero:
        return "preserve-provenance"
    if "reject" in non_zero:
        return "audit-only"
    return "export-before-prune"


def _format_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{key}={value}" for key, value in counts.items()) or "none"


def _title_pattern(title: str) -> str:
    compact = " ".join(str(title).split())
    known_prefixes = (
        "Project memory: Untracked file",
        "Project memory: Git status",
        "Project memory: Diff",
        "Failure memory: Command",
        "Procedure memory: Command",
        "Decision memory:",
        "Goal memory:",
        "Self memory:",
        "Consolidated session episode narrative",
        "Consolidated file artifact episode evidence",
        "Consolidated command episode outcomes",
        "Latest artifact memory:",
        "Command:",
        "Git status for",
    )
    for prefix in known_prefixes:
        if compact.startswith(prefix):
            return prefix
    if ":" in compact:
        return compact.split(":", 1)[0] + ":"
    words = compact.split()
    return " ".join(words[:6]) if words else "untitled"


def _latest_retention_cycle(root: Path, *, scope: str | None) -> dict[str, Any] | None:
    cycle_dir = root / "archive" / "retention-cycles"
    if not cycle_dir.exists():
        return None
    candidates = sorted(cycle_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("scope") not in (scope, None):
            continue
        created_at = _created_at(path, payload)
        plan = payload.get("prune_plan") or {}
        shadow = payload.get("shadow_prune") or {}
        return {
            "path": str(path),
            "passed": bool(payload.get("passed")),
            "created_at": created_at.isoformat(),
            "age_hours": round((datetime.now(timezone.utc) - created_at).total_seconds() / 3600, 2),
            "plan_totals": plan.get("totals", {}),
            "shadow_deletion": (shadow.get("deletion") or {}) if shadow else {},
            "backup_path": (payload.get("backup") or {}).get("path"),
            "cold_export_path": (payload.get("cold_export") or {}).get("path"),
            "recommendations": payload.get("recommendations", []),
        }
    return None


def _created_at(path: Path, payload: dict[str, Any]) -> datetime:
    raw = payload.get("created_at")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _cycle_evidence(
    totals: dict[str, Any],
    latest_cycle: dict[str, Any] | None,
    *,
    max_cycle_age_hours: float,
) -> dict[str, Any]:
    current = {
        "cold_capsules": totals["cold_capsules"],
        "protected_events": totals["protected_source_events"],
        "prunable_events": totals["prunable_source_events"],
    }
    if not latest_cycle:
        return {
            "required": totals["cold_capsules"] > 0 and totals["cold_ratio"] > 0.5,
            "fresh": False,
            "max_age_hours": max_cycle_age_hours,
            "matches_current": False,
            "shadow_events_preserved": False,
            "current": current,
            "cycle": None,
        }

    plan_totals = dict(latest_cycle.get("plan_totals") or {})
    shadow_deletion = dict(latest_cycle.get("shadow_deletion") or {})
    cycle = {
        "cold_capsules": plan_totals.get("cold_capsules"),
        "protected_events": plan_totals.get("protected_events"),
        "prunable_events": plan_totals.get("prunable_events"),
    }
    fresh = float(latest_cycle.get("age_hours") or 0) <= max_cycle_age_hours
    matches_current = all(current[key] == cycle.get(key) for key in current)
    shadow_events_preserved = bool(shadow_deletion) and shadow_deletion.get("events_removed", 0) == 0
    return {
        "required": totals["cold_capsules"] > 0 and totals["cold_ratio"] > 0.5,
        "fresh": fresh,
        "max_age_hours": max_cycle_age_hours,
        "matches_current": matches_current,
        "shadow_events_preserved": shadow_events_preserved,
        "current": current,
        "cycle": cycle,
    }


def _status(
    totals: dict[str, Any],
    latest_cycle: dict[str, Any] | None,
    cycle_evidence: dict[str, Any],
) -> str:
    if totals["cold_capsules"] == 0:
        return "pass"
    if totals["cold_ratio"] <= 0.5:
        return "pass"
    if (
        latest_cycle
        and latest_cycle.get("passed")
        and cycle_evidence.get("fresh")
        and cycle_evidence.get("matches_current")
        and cycle_evidence.get("shadow_events_preserved")
    ):
        return "pass"
    return "watch"


def _recommend(
    totals: dict[str, Any],
    groups: list[ColdGroup],
    lifecycle_tiers: list[ColdLifecycleTier],
    active_pins: list[ActivePinGroup],
    latest_cycle: dict[str, Any] | None,
    cycle_evidence: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if totals["cold_capsules"] == 0:
        return ["No cold capsules are present; keep scheduled maintenance and backups."]
    if groups:
        top = groups[0]
        pattern = top.pattern.rstrip(":")
        recommendations.append(
            "Review top cold group before pruning: "
            f"{top.status}/{top.kind}/{pattern} has {top.count} capsules "
            f"and action={top.recommended_action}."
        )
    tier_counts = {tier.name: tier.count for tier in lifecycle_tiers}
    if tier_counts.get("evidence", 0):
        recommendations.append(
            f"Treat {tier_counts['evidence']} cold capsules as evidence because active memories still share provenance."
        )
    if tier_counts.get("archive", 0):
        recommendations.append(
            f"Archive-tier cold capsules={tier_counts['archive']}; export and verify them before any destructive cleanup."
        )
    if tier_counts.get("reject", 0):
        recommendations.append(
            f"Reject-tier cold capsules={tier_counts['reject']}; keep as audit evidence until reviewed."
        )
    if active_pins:
        top_pin = active_pins[0]
        recommendations.append(
            "Review active provenance pins before expecting cold pressure to fall: "
            f"{top_pin.status}/{top_pin.kind}/{top_pin.pattern.rstrip(':')} pins {top_pin.source_events} source events."
        )
    if totals["protected_source_events"]:
        recommendations.append(
            f"Keep {totals['protected_source_events']} source events because active memories still cite them."
        )
    if totals["prunable_source_events"]:
        recommendations.append(
            f"{totals['prunable_source_events']} source events are only cited by cold capsules; export before any destructive cleanup."
        )
    if not latest_cycle or not latest_cycle.get("passed"):
        recommendations.append("Run retention-cycle with representative recall queries before preparing live pruning.")
    else:
        if cycle_evidence.get("fresh") and cycle_evidence.get("matches_current") and cycle_evidence.get("shadow_events_preserved"):
            recommendations.append(
                "Latest retention-cycle is fresh, matches the current cold set, and preserved source events in shadow-prune."
            )
        else:
            if not cycle_evidence.get("fresh"):
                recommendations.append("Latest retention-cycle is stale; rerun it before relying on cold stewardship evidence.")
            if not cycle_evidence.get("matches_current"):
                recommendations.append("Cold memory totals drifted since the latest retention-cycle; rerun retention-cycle.")
            if not cycle_evidence.get("shadow_events_preserved"):
                recommendations.append("Latest shadow-prune did not prove source-event preservation; rerun retention-cycle.")
        recommendations.append("Latest retention-cycle passed; use it as evidence, not approval, before any live prune.")
    return recommendations
