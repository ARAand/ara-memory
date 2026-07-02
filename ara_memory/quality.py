from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ara_memory.models import MemoryStatus, new_id, utc_now
from ara_memory.promotion import can_promote_capsule, promotion_block_reason
from ara_memory.risk import MemoryRiskAssessor, redact_memory_tags, redact_sensitive_text
from ara_memory.storage import MemoryStore, row_to_capsule


PROMOTE_THRESHOLD = 0.74
REVIEW_THRESHOLD = 0.50
DECAY_THRESHOLD = 0.62
QUEUE_ACTIONS = {"promote", "review", "decay", "quarantine"}


@dataclass(slots=True)
class QualityItem:
    capsule_id: str
    scope: str
    status: str
    kind: str
    title: str
    quality_score: float
    decay_score: float
    action: str
    priority: float
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "scope": self.scope,
            "status": self.status,
            "kind": self.kind,
            "title": self.title,
            "quality_score": round(self.quality_score, 3),
            "decay_score": round(self.decay_score, 3),
            "action": self.action,
            "priority": round(self.priority, 3),
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class QualityReport:
    scope: str | None
    persisted: bool
    totals: dict[str, int]
    items: list[QualityItem]
    queue: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "persisted": self.persisted,
            "totals": self.totals,
            "items": [item.as_dict() for item in self.items],
            "queue": self.queue,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Memory Quality: {scope}"]
        lines.append(
            "totals: "
            f"scored={self.totals['scored']}, promote={self.totals['promote']}, "
            f"review={self.totals['review']}, decay={self.totals['decay']}, "
            f"quarantine={self.totals['quarantine']}, keep={self.totals['keep']}"
        )
        lines.append("## Queue")
        if self.queue:
            for row in self.queue[:10]:
                lines.append(f"- {row['id']} [{row['action']}] priority={row['priority']:.2f} {row['reason']}")
        else:
            lines.append("- None.")
        lines.append("## Top Items")
        for item in self.items[:10]:
            lines.append(
                f"- {item.capsule_id} [{item.kind}/{item.status}] action={item.action} "
                f"quality={item.quality_score:.2f} decay={item.decay_score:.2f} {item.title}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class PromotionCandidateItem:
    capsule_id: str
    scope: str
    status: str
    kind: str
    title: str
    quality_score: float
    priority: float
    gate: str
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "scope": self.scope,
            "status": self.status,
            "kind": self.kind,
            "title": self.title,
            "quality_score": round(self.quality_score, 3),
            "priority": round(self.priority, 3),
            "gate": self.gate,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class PromotionCandidateReport:
    scope: str | None
    limit: int
    threshold: float
    scanned: int
    ready: list[PromotionCandidateItem]
    blocked: list[PromotionCandidateItem]

    @property
    def passed(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "limit": self.limit,
            "threshold": round(self.threshold, 3),
            "scanned": self.scanned,
            "ready_count": len(self.ready),
            "blocked_count": len(self.blocked),
            "ready": [item.as_dict() for item in self.ready],
            "blocked": [item.as_dict() for item in self.blocked],
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Promotion Candidates: {scope}",
            f"scanned={self.scanned}, threshold={self.threshold:.2f}, ready={len(self.ready)}, blocked={len(self.blocked)}",
        ]
        lines.append("## Ready")
        if self.ready:
            for item in self.ready[:20]:
                lines.append(
                    f"- {item.capsule_id} [{item.kind}] quality={item.quality_score:.2f}: {item.title}"
                )
        else:
            lines.append("- None.")
        lines.append("## Blocked")
        if self.blocked:
            for item in self.blocked[:20]:
                lines.append(
                    f"- {item.capsule_id} [{item.kind}] quality={item.quality_score:.2f}: "
                    f"{'; '.join(item.reasons[:2])}"
                )
        else:
            lines.append("- None.")
        return "\n".join(lines)


@dataclass(slots=True)
class ReviewWorkerItem:
    queue_id: str
    capsule_id: str
    requested_action: str
    applied_action: str
    dry_run: bool
    changed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "capsule_id": self.capsule_id,
            "requested_action": self.requested_action,
            "applied_action": self.applied_action,
            "dry_run": self.dry_run,
            "changed": self.changed,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ReviewWorkerReport:
    scope: str | None
    dry_run: bool
    processed: int
    changed: int
    items: list[ReviewWorkerItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "processed": self.processed,
            "changed": self.changed,
            "items": [item.as_dict() for item in self.items],
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Review Worker: {scope}", f"dry_run: {self.dry_run}", f"processed={self.processed}, changed={self.changed}"]
        for item in self.items[:20]:
            marker = "changed" if item.changed else "kept"
            lines.append(f"- [{marker}] {item.queue_id} {item.requested_action}->{item.applied_action}: {item.reason}")
        return "\n".join(lines)


@dataclass(slots=True)
class ReviewCompactItem:
    queue_id: str
    capsule_id: str
    action: str
    changed: bool
    dry_run: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "capsule_id": self.capsule_id,
            "action": self.action,
            "changed": self.changed,
            "dry_run": self.dry_run,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ReviewCompactReport:
    scope: str | None
    dry_run: bool
    processed: int
    changed: int
    skipped: int
    items: list[ReviewCompactItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "processed": self.processed,
            "changed": self.changed,
            "skipped": self.skipped,
            "items": [item.as_dict() for item in self.items],
        }

    def as_compact_dict(self, *, examples_per_group: int = 3) -> dict[str, Any]:
        changed_items = [item for item in self.items if item.changed]
        skipped_items = [item for item in self.items if not item.changed]
        groups: dict[str, dict[str, Any]] = {}
        for item in self.items:
            key = _review_compact_group_key(item)
            group = groups.setdefault(
                key,
                {
                    "key": key,
                    "action": item.action,
                    "changed": item.changed,
                    "reason": _review_compact_reason_label(item.reason),
                    "count": 0,
                    "examples": [],
                },
            )
            group["count"] += 1
            if len(group["examples"]) < examples_per_group:
                group["examples"].append(
                    {
                        "queue_id": item.queue_id,
                        "capsule_id": item.capsule_id,
                        "reason": item.reason,
                    }
                )
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "processed": self.processed,
            "changed": self.changed,
            "skipped": self.skipped,
            "change_rate": round(self.changed / self.processed, 4) if self.processed else 0.0,
            "actions": dict(Counter(item.action for item in self.items)),
            "outcomes": {
                "changed": len(changed_items),
                "skipped": len(skipped_items),
            },
            "groups": sorted(groups.values(), key=lambda item: (-int(item["count"]), str(item["key"]))),
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Review Compact: {scope}",
            f"dry_run: {self.dry_run}",
            f"processed={self.processed}, changed={self.changed}, skipped={self.skipped}",
        ]
        for item in self.items[:20]:
            marker = "changed" if item.changed else "kept"
            lines.append(f"- [{marker}] {item.queue_id} {item.action}: {item.reason}")
        return "\n".join(lines)

    def to_compact_text(self) -> str:
        payload = self.as_compact_dict()
        scope = self.scope or "all"
        lines = [
            f"# Ara Review Compact Summary: {scope}",
            f"dry_run: {self.dry_run}",
            (
                f"processed={self.processed}, changed={self.changed}, skipped={self.skipped}, "
                f"change_rate={payload['change_rate']:.2%}"
            ),
        ]
        for group in payload["groups"]:
            marker = "changed" if group["changed"] else "kept"
            lines.append(f"- [{marker}] {group['count']} {group['action']}: {group['reason']}")
            for example in group["examples"][:2]:
                lines.append(f"  - {example['queue_id']} {example['capsule_id']}: {example['reason']}")
        return "\n".join(lines)


def _review_compact_reason_label(reason: str) -> str:
    text = str(reason)
    prefix = "acknowledged non-destructive review marker: "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if "excluded from hot memory by deterministic risk policy" in text:
        return "deterministic artifact exclusion marker"
    if "low quality score" in text:
        return "low quality marker"
    if "resolved stale queue action" in text:
        return "stale queue action"
    if "capsule no longer exists" in text:
        return "missing capsule"
    if "needs explicit policy" in text:
        return "requires explicit policy"
    return text[:120]


def _review_compact_group_key(item: ReviewCompactItem) -> str:
    changed = "changed" if item.changed else "kept"
    return f"{changed}|{item.action}|{_review_compact_reason_label(item.reason)}"


@dataclass(slots=True)
class ReviewRedactionItem:
    queue_id: str
    capsule_id: str
    action: str
    changed: bool
    resolved: bool
    dry_run: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "capsule_id": self.capsule_id,
            "action": self.action,
            "changed": self.changed,
            "resolved": self.resolved,
            "dry_run": self.dry_run,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ReviewRedactionReport:
    scope: str | None
    dry_run: bool
    processed: int
    changed: int
    resolved: int
    skipped: int
    items: list[ReviewRedactionItem]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "processed": self.processed,
            "changed": self.changed,
            "resolved": self.resolved,
            "skipped": self.skipped,
            "items": [item.as_dict() for item in self.items],
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Review Redact: {scope}",
            f"dry_run: {self.dry_run}",
            f"processed={self.processed}, changed={self.changed}, resolved={self.resolved}, skipped={self.skipped}",
        ]
        for item in self.items[:20]:
            marker = "changed" if item.changed else "kept"
            resolved = " resolved" if item.resolved else ""
            lines.append(f"- [{marker}{resolved}] {item.queue_id} {item.action}: {item.reason}")
        return "\n".join(lines)


@dataclass(slots=True)
class ReviewTriageGroup:
    key: str
    action: str
    reason: str
    count: int
    max_priority: float
    capsule_statuses: dict[str, int]
    capsule_kinds: dict[str, int]
    examples: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "action": self.action,
            "reason": self.reason,
            "count": self.count,
            "max_priority": round(self.max_priority, 3),
            "capsule_statuses": self.capsule_statuses,
            "capsule_kinds": self.capsule_kinds,
            "examples": self.examples,
        }


@dataclass(slots=True)
class ReviewTriageReport:
    scope: str | None
    total_open: int
    groups: list[ReviewTriageGroup]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "total_open": self.total_open,
            "groups": [group.as_dict() for group in self.groups],
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [f"# Ara Review Triage: {scope}", f"open_items={self.total_open}, groups={len(self.groups)}"]
        for group in self.groups[:20]:
            lines.append(
                f"- {group.key}: count={group.count}, max_priority={group.max_priority:.2f}, "
                f"statuses={group.capsule_statuses}, kinds={group.capsule_kinds}"
            )
            for example in group.examples[:3]:
                lines.append(f"  - {example['queue_id']} {example['capsule_id']}: {example['title']}")
        return "\n".join(lines)


class QualityScorer:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.risk = MemoryRiskAssessor(store)

    def run(
        self,
        *,
        scope: str | None = None,
        limit: int = 500,
        persist: bool = False,
    ) -> QualityReport:
        self.store.init()
        capsules = self._capsules(scope=scope, limit=limit)
        items = [self._score(cap) for cap in capsules]
        items.sort(key=lambda item: (item.priority, item.quality_score), reverse=True)
        queue: list[dict[str, Any]] = []
        if persist:
            queue = self._persist(items)
        totals = _totals(items)
        return QualityReport(scope=scope, persisted=persist, totals=totals, items=items, queue=queue)

    def promotion_candidates(
        self,
        *,
        scope: str | None = None,
        limit: int = 500,
        threshold: float = PROMOTE_THRESHOLD,
    ) -> PromotionCandidateReport:
        self.store.init()
        capsules = self._capsules(scope=scope, limit=limit)
        ready: list[PromotionCandidateItem] = []
        blocked: list[PromotionCandidateItem] = []
        for cap in capsules:
            if cap["status"] != MemoryStatus.CANDIDATE.value:
                continue
            item = self._score(cap)
            if item.quality_score < threshold:
                continue
            if item.action == "promote":
                ready.append(_promotion_candidate_item(item, gate="ready"))
            elif any(str(reason).startswith("promotion gate blocked") for reason in item.reasons):
                blocked.append(_promotion_candidate_item(item, gate="blocked"))
        ready.sort(key=lambda item: (item.priority, item.quality_score), reverse=True)
        blocked.sort(key=lambda item: (item.priority, item.quality_score), reverse=True)
        return PromotionCandidateReport(
            scope=scope,
            limit=limit,
            threshold=threshold,
            scanned=len(capsules),
            ready=ready,
            blocked=blocked,
        )

    def list_queue(self, *, scope: str | None = None, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        self.store.init()
        clauses = ["status = ?"]
        args: list[Any] = [status]
        if scope:
            clauses.append("scope = ?")
            args.append(scope)
        args.append(limit)
        with self.store.session() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT *
                    FROM memory_review_queue
                    WHERE {' AND '.join(clauses)}
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ?
                    """,
                    args,
                )
            ]

    def resolve_queue_item(self, queue_id: str) -> bool:
        self.store.init()
        with self.store.session() as conn:
            cur = conn.execute(
                "UPDATE memory_review_queue SET status = ?, resolved_at = ? WHERE id = ? AND status = ?",
                ("resolved", utc_now(), queue_id, "open"),
            )
            return cur.rowcount > 0

    def triage_queue(
        self,
        *,
        scope: str | None = None,
        status: str = "open",
        limit: int = 500,
        examples_per_group: int = 3,
    ) -> ReviewTriageReport:
        rows = self.list_queue(scope=scope, status=status, limit=limit)
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            capsule = self._capsule(row["capsule_id"])
            cap_status = str(capsule["status"]) if capsule else "missing"
            cap_kind = str(capsule["kind"]) if capsule else "missing"
            reason = _reason_bucket(str(row["reason"]))
            key = f"{row['action']}|{reason}|{cap_status}|{cap_kind}"
            bucket = groups.setdefault(
                key,
                {
                    "key": key,
                    "action": row["action"],
                    "reason": reason,
                    "count": 0,
                    "max_priority": 0.0,
                    "capsule_statuses": {},
                    "capsule_kinds": {},
                    "examples": [],
                },
            )
            bucket["count"] += 1
            bucket["max_priority"] = max(float(bucket["max_priority"]), float(row["priority"]))
            bucket["capsule_statuses"][cap_status] = bucket["capsule_statuses"].get(cap_status, 0) + 1
            bucket["capsule_kinds"][cap_kind] = bucket["capsule_kinds"].get(cap_kind, 0) + 1
            if len(bucket["examples"]) < examples_per_group:
                bucket["examples"].append(
                    {
                        "queue_id": row["id"],
                        "capsule_id": row["capsule_id"],
                        "priority": round(float(row["priority"]), 3),
                        "title": capsule["title"] if capsule else "<missing capsule>",
                    }
                )
        triage_groups = [
            ReviewTriageGroup(
                key=item["key"],
                action=item["action"],
                reason=item["reason"],
                count=item["count"],
                max_priority=item["max_priority"],
                capsule_statuses=dict(sorted(item["capsule_statuses"].items())),
                capsule_kinds=dict(sorted(item["capsule_kinds"].items())),
                examples=item["examples"],
            )
            for item in groups.values()
        ]
        triage_groups.sort(key=lambda group: (group.count, group.max_priority), reverse=True)
        return ReviewTriageReport(scope=scope, total_open=len(rows), groups=triage_groups)

    def work_queue(
        self,
        *,
        scope: str | None = None,
        limit: int = 25,
        dry_run: bool = True,
    ) -> ReviewWorkerReport:
        self.store.init()
        queue = self.list_queue(scope=scope, status="open", limit=limit)
        items = [self._work_item(row, dry_run=dry_run) for row in queue]
        return ReviewWorkerReport(
            scope=scope,
            dry_run=dry_run,
            processed=len(items),
            changed=sum(1 for item in items if item.changed),
            items=items,
        )

    def compact_queue(
        self,
        *,
        scope: str | None = None,
        limit: int = 250,
        dry_run: bool = True,
    ) -> ReviewCompactReport:
        self.store.init()
        queue = self.list_queue(scope=scope, status="open", limit=limit)
        items = [self._compact_item(row, dry_run=dry_run) for row in queue]
        return ReviewCompactReport(
            scope=scope,
            dry_run=dry_run,
            processed=len(items),
            changed=sum(1 for item in items if item.changed),
            skipped=sum(1 for item in items if not item.changed),
            items=items,
        )

    def redact_sensitive_reviews(
        self,
        *,
        scope: str | None = None,
        limit: int = 50,
        dry_run: bool = True,
    ) -> ReviewRedactionReport:
        self.store.init()
        queue = self.list_queue(scope=scope, status="open", limit=limit)
        items = [self._redact_item(row, dry_run=dry_run) for row in queue]
        return ReviewRedactionReport(
            scope=scope,
            dry_run=dry_run,
            processed=len(items),
            changed=sum(1 for item in items if item.changed),
            resolved=sum(1 for item in items if item.resolved),
            skipped=sum(1 for item in items if not item.changed),
            items=items,
        )

    def _compact_item(self, row: dict[str, Any], *, dry_run: bool) -> ReviewCompactItem:
        capsule = self._capsule(row["capsule_id"])
        if capsule is None:
            if not dry_run:
                self.resolve_queue_item(row["id"])
            return ReviewCompactItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=True,
                dry_run=dry_run,
                reason="resolved because capsule no longer exists",
            )
        current = self._score(capsule)
        if current.action != row["action"]:
            if not dry_run:
                self.resolve_queue_item(row["id"])
            return ReviewCompactItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=True,
                dry_run=dry_run,
                reason=f"resolved stale queue action; current action is {current.action}",
            )
        if row["action"] == "review" and _is_acknowledgeable_review(str(row["reason"])):
            if not dry_run:
                self.resolve_queue_item(row["id"])
            return ReviewCompactItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=True,
                dry_run=dry_run,
                reason=f"acknowledged non-destructive review marker: {row['reason']}",
            )
        return ReviewCompactItem(
            queue_id=row["id"],
            capsule_id=row["capsule_id"],
            action=row["action"],
            changed=False,
            dry_run=dry_run,
            reason="left open; action changes memory state or needs explicit policy",
        )

    def _redact_item(self, row: dict[str, Any], *, dry_run: bool) -> ReviewRedactionItem:
        capsule = self._capsule(row["capsule_id"])
        if capsule is None:
            return ReviewRedactionItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=False,
                resolved=False,
                dry_run=dry_run,
                reason="left open; capsule no longer exists and should be compacted separately",
            )
        current = self._score(capsule)
        reason = "; ".join(current.reasons[:4]) or str(row["reason"])
        if row["action"] != "review" or current.action != "review":
            return ReviewRedactionItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=False,
                resolved=False,
                dry_run=dry_run,
                reason=f"skipped; current review action is {current.action}",
            )
        if not _is_sensitive_review(reason):
            return ReviewRedactionItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=False,
                resolved=False,
                dry_run=dry_run,
                reason=f"skipped; review is not a sensitive-data marker: {reason}",
            )

        redacted_title = redact_sensitive_text(str(capsule["title"]))
        redacted_body = redact_sensitive_text(str(capsule["body"]))
        redacted_tags = _redacted_projection_tags(capsule["tags"])
        changed = (
            redacted_title != capsule["title"]
            or redacted_body != capsule["body"]
            or redacted_tags != capsule["tags"]
        )
        if not changed:
            return ReviewRedactionItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=False,
                resolved=False,
                dry_run=dry_run,
                reason="left open; no sensitive text was removed from the capsule projection",
            )

        if dry_run:
            return ReviewRedactionItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=True,
                resolved=True,
                dry_run=dry_run,
                reason="would redact capsule projection and resolve sensitive review marker",
            )

        updated = self.store.update_capsule_projection_if_current(
            capsule["id"],
            title=redacted_title,
            body=redacted_body,
            tags=redacted_tags,
            expected_status=capsule["status"],
            expected_title=capsule["title"],
            expected_body=capsule["body"],
            expected_tags=capsule["tags"],
            actor="review-redact",
            reason=reason,
            action="redact-sensitive-review",
            review_queue_id=row["id"],
        )
        if not updated:
            return ReviewRedactionItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                action=row["action"],
                changed=False,
                resolved=False,
                dry_run=dry_run,
                reason="left open; capsule changed before redaction apply",
            )
        refreshed = self._capsule(capsule["id"])
        still_sensitive = bool(refreshed and self._score(refreshed).action == "review" and self.risk.assess_capsule(refreshed).has_sensitive_text)
        resolved = False
        if not still_sensitive:
            resolved = self.resolve_queue_item(row["id"])
        return ReviewRedactionItem(
            queue_id=row["id"],
            capsule_id=row["capsule_id"],
            action=row["action"],
            changed=True,
            resolved=resolved,
            dry_run=dry_run,
            reason="redacted capsule projection" + (" and resolved review marker" if resolved else "; sensitive review remains open"),
        )

    def _work_item(self, row: dict[str, Any], *, dry_run: bool) -> ReviewWorkerItem:
        capsule = self._capsule(row["capsule_id"])
        if capsule is None:
            if not dry_run:
                self.resolve_queue_item(row["id"])
            return ReviewWorkerItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                requested_action=row["action"],
                applied_action="resolve",
                dry_run=dry_run,
                changed=not dry_run,
                reason="capsule no longer exists",
            )

        current = self._score(capsule)
        if current.action != row["action"]:
            if not dry_run:
                self.resolve_queue_item(row["id"])
            return ReviewWorkerItem(
                queue_id=row["id"],
                capsule_id=row["capsule_id"],
                requested_action=row["action"],
                applied_action="resolve",
                dry_run=dry_run,
                changed=not dry_run,
                reason=f"queue action is stale; current action is {current.action}",
            )

        applied = "keep-open"
        changed = False
        reason = "; ".join(current.reasons[:4]) or row["reason"]
        if row["action"] == "promote" and capsule["status"] == MemoryStatus.CANDIDATE.value:
            gate = can_promote_capsule(self.store, capsule, require_provenance=True)
            if not gate.allowed:
                applied = "keep-open"
                changed = False
                reason = promotion_block_reason(gate)
            else:
                applied = "promote"
                changed = True
            if not dry_run:
                if changed:
                    updated = self.store.update_capsule_status_if_current(
                        capsule["id"],
                        MemoryStatus.CANDIDATE,
                        MemoryStatus.STABLE,
                        actor="review-worker",
                        reason=reason,
                    )
                    changed = updated
                    if updated:
                        self.resolve_queue_item(row["id"])
                    else:
                        applied = "keep-open"
                        reason = "promotion skipped because capsule status changed before apply"
        elif row["action"] == "quarantine" and capsule["status"] != MemoryStatus.QUARANTINED.value:
            applied = "quarantine"
            changed = True
            if not dry_run:
                self.store.update_capsule_status(capsule["id"], MemoryStatus.QUARANTINED, actor="review-worker", reason=reason)
                self.resolve_queue_item(row["id"])
        elif row["action"] == "review":
            if _is_acknowledgeable_review(reason):
                applied = "resolve"
                changed = True
                reason = f"manual review marker acknowledged: {reason}"
                if not dry_run:
                    self.resolve_queue_item(row["id"])
            else:
                applied = "keep-open"
                changed = False
                reason = f"review requires explicit policy choice: {reason}"
        elif row["action"] == "decay":
            applied = "manual-decay-review"
            changed = False
            reason = f"decay requires explicit human/Ara policy choice: {reason}"
        return ReviewWorkerItem(
            queue_id=row["id"],
            capsule_id=row["capsule_id"],
            requested_action=row["action"],
            applied_action=applied,
            dry_run=dry_run,
            changed=changed,
            reason=reason,
        )

    def _capsule(self, capsule_id: str) -> dict[str, Any] | None:
        row = self.store.get_capsule(capsule_id)
        return row_to_capsule(row) if row else None

    def _capsules(self, *, scope: str | None, limit: int) -> list[dict[str, Any]]:
        clauses = ["status IN ('candidate', 'stable', 'quarantined')"]
        args: list[Any] = []
        if scope:
            clauses.append("scope = ?")
            args.append(scope)
        args.append(limit)
        with self.store.session() as conn:
            return [
                row_to_capsule(row)
                for row in conn.execute(
                    f"""
                    SELECT *
                    FROM capsules
                    WHERE {' AND '.join(clauses)}
                    ORDER BY updated_at DESC, salience DESC
                    LIMIT ?
                    """,
                    args,
                )
            ]

    def _score(self, cap: dict[str, Any]) -> QualityItem:
        risk = self.risk.assess_capsule(cap)
        age_days = _age_days(cap["updated_at"])
        source_count = len(cap["source_event_ids"])
        quality = 0.0
        quality += float(cap["confidence"]) * 0.36
        quality += float(cap["salience"]) * 0.30
        quality += _kind_bonus(cap["kind"])
        quality += min(0.10, source_count * 0.025)
        quality += 0.08 if cap["status"] == MemoryStatus.STABLE.value else 0.0
        quality -= risk.score * 0.32
        quality -= min(0.12, age_days / 365.0 * 0.12)
        quality = _clamp(quality)

        decay = _clamp((age_days / 180.0) * 0.40 + (1.0 - float(cap["salience"])) * 0.30 + (1.0 - quality) * 0.30)
        if risk.should_quarantine and cap["status"] == MemoryStatus.QUARANTINED.value:
            action, reasons = "keep", ["high memory safety risk already quarantined"]
        elif risk.should_quarantine:
            action, reasons = "quarantine", ["high memory safety risk"]
        elif risk.should_exclude_from_hot:
            action, reasons = "review", ["excluded from hot memory by deterministic risk policy"]
        else:
            action, reasons = _action_for(cap, quality=quality, decay=decay)
            if action == "promote":
                gate = can_promote_capsule(self.store, cap, require_provenance=True, risk_verdict=risk)
                if not gate.allowed:
                    action = "review"
                    reasons = [promotion_block_reason(gate)]
        reasons.extend(risk.reasons[:3])
        priority = _priority(action, quality=quality, decay=decay, risk_score=risk.score)
        return QualityItem(
            capsule_id=cap["id"],
            scope=cap["scope"],
            status=cap["status"],
            kind=cap["kind"],
            title=cap["title"],
            quality_score=quality,
            decay_score=decay,
            action=action,
            priority=priority,
            reasons=reasons,
        )

    def _persist(self, items: list[QualityItem]) -> list[dict[str, Any]]:
        now = utc_now()
        with self.store.session() as conn:
            for item in items:
                conn.execute(
                    """
                    INSERT INTO memory_quality_scores(
                      capsule_id, scope, status, quality_score, decay_score, action, reasons_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.capsule_id,
                        item.scope,
                        item.status,
                        item.quality_score,
                        item.decay_score,
                        item.action,
                        json.dumps(item.reasons, ensure_ascii=False),
                        now,
                    ),
                )
                if item.action in QUEUE_ACTIONS:
                    existing = conn.execute(
                        """
                        SELECT id
                        FROM memory_review_queue
                        WHERE capsule_id = ? AND action = ? AND status = 'open'
                        """,
                        (item.capsule_id, item.action),
                    ).fetchone()
                    reason = "; ".join(item.reasons[:4]) or item.action
                    if existing:
                        conn.execute(
                            """
                            UPDATE memory_review_queue
                            SET priority = ?, reason = ?, created_at = ?
                            WHERE id = ?
                            """,
                            (item.priority, reason, now, existing["id"]),
                        )
                    elif item.action == "review" and _has_resolved_review_marker(
                        conn,
                        capsule_id=item.capsule_id,
                        action=item.action,
                        reason=reason,
                    ):
                        continue
                    else:
                        conn.execute(
                            """
                            INSERT INTO memory_review_queue(
                              id, capsule_id, scope, action, priority, reason, status, created_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_id("review"),
                                item.capsule_id,
                                item.scope,
                                item.action,
                                item.priority,
                                reason,
                                "open",
                                now,
                            ),
                        )
            return [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM memory_review_queue
                    WHERE status = 'open'
                    ORDER BY priority DESC, created_at DESC
                    LIMIT 50
                    """
                )
            ]


def _action_for(cap: dict[str, Any], *, quality: float, decay: float) -> tuple[str, list[str]]:
    reasons = []
    if cap["status"] == MemoryStatus.CANDIDATE.value and quality >= PROMOTE_THRESHOLD:
        reasons.append("candidate has high confidence/salience/provenance")
        return "promote", reasons
    if cap["status"] == MemoryStatus.STABLE.value and decay >= DECAY_THRESHOLD:
        reasons.append("stable memory appears stale or low utility")
        return "decay", reasons
    if quality < REVIEW_THRESHOLD:
        reasons.append("low quality score")
        return "review", reasons
    return "keep", ["quality is acceptable"]


def _promotion_candidate_item(item: QualityItem, *, gate: str) -> PromotionCandidateItem:
    return PromotionCandidateItem(
        capsule_id=item.capsule_id,
        scope=item.scope,
        status=item.status,
        kind=item.kind,
        title=item.title,
        quality_score=item.quality_score,
        priority=item.priority,
        gate=gate,
        reasons=list(item.reasons),
    )


def _priority(action: str, *, quality: float, decay: float, risk_score: float) -> float:
    if action == "quarantine":
        return _clamp(0.75 + risk_score * 0.25)
    if action == "promote":
        return _clamp(quality)
    if action == "decay":
        return _clamp(decay)
    if action == "review":
        return _clamp(1.0 - quality)
    return 0.0


def _kind_bonus(kind: str) -> float:
    return {
        "decision": 0.10,
        "goal": 0.10,
        "procedure": 0.08,
        "failure": 0.08,
        "summary": 0.07,
        "self": 0.06,
        "preference": 0.05,
        "fact": 0.04,
        "project": 0.02,
        "episode": 0.00,
        "conflict": -0.02,
    }.get(kind, 0.0)


def _age_days(value: str) -> float:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return 365.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _totals(items: list[QualityItem]) -> dict[str, int]:
    totals = {
        "scored": len(items),
        "promote": 0,
        "review": 0,
        "decay": 0,
        "quarantine": 0,
        "keep": 0,
    }
    for item in items:
        totals[item.action] = totals.get(item.action, 0) + 1
    return totals


def _is_acknowledgeable_review(reason: str) -> bool:
    lowered = reason.lower()
    if "low quality score" in lowered:
        return True
    if "excluded from hot memory by deterministic risk policy" not in lowered:
        return False
    return (
        "instruction-like text appears inside code/test/document artifact" in lowered
        or "keyword stuffing" in lowered
    )


def _is_sensitive_review(reason: str) -> bool:
    lowered = reason.lower()
    return "sensitive data" in lowered or "direct identifier" in lowered


def _redacted_projection_tags(tags: list[str]) -> list[str]:
    safe = redact_memory_tags(tags)
    for tag in ("privacy:redacted", "review-redacted"):
        if tag not in safe:
            safe.append(tag)
    return safe


def _has_resolved_review_marker(conn: Any, *, capsule_id: str, action: str, reason: str) -> bool:
    row = conn.execute(
        """
        SELECT id
        FROM memory_review_queue
        WHERE capsule_id = ?
          AND action = ?
          AND reason = ?
          AND status = 'resolved'
        LIMIT 1
        """,
        (capsule_id, action, reason),
    ).fetchone()
    return row is not None


def _reason_bucket(reason: str) -> str:
    parts = [part.strip() for part in reason.split(";") if part.strip()]
    if not parts:
        return "unspecified"
    if any("instruction-like text" in part for part in parts):
        return "instruction-like artifact"
    if any("sensitive data" in part for part in parts):
        return "sensitive data"
    if any("self-serving identity claim" in part for part in parts):
        return "self-serving identity"
    if any("keyword stuffing" in part for part in parts):
        return "keyword stuffing"
    if any("low quality score" in part for part in parts):
        return "low quality score"
    if any("decay" in part or "stale" in part for part in parts):
        return "decay or stale"
    return parts[0][:120]
