from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import sqlite3
import secrets
from typing import Any

from ara_memory.models import new_id, utc_now
from ara_memory.storage import MemoryStore


RELATION_MERGE_CONFIRMATION = "MERGE RELATION NODES"

LOW_VALUE_RELATION_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "memory",
    "ara",
    "relation",
    "node",
    "edge",
    "scope",
    "manifest",
    "baseline",
    "budget",
    "query",
}


@dataclass(slots=True)
class RelationMergeCandidate:
    scope: str
    canonical_node_id: str
    candidate_node_id: str
    canonical_label: str
    candidate_label: str
    score: float
    reasons: list[str]
    shared_neighbors: int
    canonical_degree: int
    candidate_degree: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "canonical_node_id": self.canonical_node_id,
            "candidate_node_id": self.candidate_node_id,
            "canonical_label": self.canonical_label,
            "candidate_label": self.candidate_label,
            "score": self.score,
            "reasons": self.reasons,
            "shared_neighbors": self.shared_neighbors,
            "canonical_degree": self.canonical_degree,
            "candidate_degree": self.candidate_degree,
        }


@dataclass(slots=True)
class RelationMergeReport:
    scope: str
    candidates: list[RelationMergeCandidate]
    nodes_considered: int
    pairs_considered: int
    threshold: float
    limit: int

    @property
    def passed(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "nodes_considered": self.nodes_considered,
            "pairs_considered": self.pairs_considered,
            "threshold": self.threshold,
            "limit": self.limit,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Relation Merge Dry Run: {self.scope}",
            "status: pass",
            f"nodes_considered: {self.nodes_considered}",
            f"pairs_considered: {self.pairs_considered}",
            f"threshold: {self.threshold:.2f}",
            f"candidate_count: {len(self.candidates)}",
            "## Candidates",
        ]
        if not self.candidates:
            lines.append("- None above threshold.")
        for candidate in self.candidates:
            reasons = ", ".join(candidate.reasons) if candidate.reasons else "score"
            lines.append(
                "- "
                f"{candidate.score:.2f} "
                f"{candidate.canonical_label!r} <- {candidate.candidate_label!r} "
                f"(shared_neighbors={candidate.shared_neighbors}, reasons={reasons})"
            )
        lines.extend(
            [
                "## Safety",
                "- Dry-run only: no relation_nodes, relation_edges, capsules, or source events are mutated.",
                "- Merge candidates require lexical similarity or shared graph neighborhoods.",
                "- Review candidates before any future apply mode.",
            ]
        )
        return "\n".join(lines)


@dataclass(slots=True)
class RelationMergeApproval:
    approval_id: str | None
    token: str | None
    expires_at: str | None
    dry_run: RelationMergeReport
    relation_fingerprint: str
    rollback_witness: dict[str, Any]

    @property
    def prepared(self) -> bool:
        return self.approval_id is not None and self.token is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "prepared": self.prepared,
            "token": self.token,
            "expires_at": self.expires_at,
            "scope": self.dry_run.scope,
            "candidate_count": len(self.dry_run.candidates),
            "relation_fingerprint": self.relation_fingerprint,
            "rollback_witness": self.rollback_witness,
            "dry_run": self.dry_run.as_dict(),
            "safety": [
                "No relation nodes or edges are mutated by prepare.",
                "Future apply must recheck the fingerprint and consume the token once.",
                "Rollback witness snapshots are stored before any future mutation exists.",
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Relation Merge Prepare: {self.dry_run.scope}",
            "status: pass",
            f"candidate_count: {len(self.dry_run.candidates)}",
            f"relation_fingerprint: {self.relation_fingerprint}",
        ]
        if self.prepared:
            lines.extend(
                [
                    f"approval_id: {self.approval_id}",
                    f"expires_at: {self.expires_at}",
                    f"approval_token: {self.token}",
                ]
            )
        else:
            lines.append("approval_id: none (no candidates above threshold)")
        lines.extend(
            [
                "## Safety",
                "- Prepare only: no relation_nodes, relation_edges, capsules, or source events are mutated.",
                "- Any future apply mode must re-run dry-run, compare the fingerprint, and require this token.",
                "- Rollback witness preview stores labels, ids, scores, and neighbor evidence.",
            ]
        )
        return "\n".join(lines)


@dataclass(slots=True)
class RelationMergeApplyReport:
    scope: str | None
    passed: bool
    approval_id: str | None
    merged: int
    witness_ids: list[str]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "approval_id": self.approval_id,
            "merged": self.merged,
            "witness_ids": self.witness_ids,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        status = "pass" if self.passed else "blocked"
        lines = [
            f"# Ara Relation Merge Apply: {self.scope or 'unknown'}",
            f"status: {status}",
            f"approval_id: {self.approval_id or 'none'}",
            f"merged: {self.merged}",
        ]
        if self.witness_ids:
            lines.append("## Witnesses")
            lines.extend(f"- {witness_id}" for witness_id in self.witness_ids)
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class RelationMergeReviewItem:
    witness_id: str
    approval_id: str
    scope: str
    canonical_node_id: str
    candidate_node_id: str
    status: str
    before_edge_count: int
    after_edge_count: int
    before_evidence_count: int
    after_evidence_count: int
    duplicate_edges_coalesced: int
    self_loops_removed: int
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "approval_id": self.approval_id,
            "scope": self.scope,
            "canonical_node_id": self.canonical_node_id,
            "candidate_node_id": self.candidate_node_id,
            "status": self.status,
            "passed": self.passed,
            "before_edge_count": self.before_edge_count,
            "after_edge_count": self.after_edge_count,
            "before_evidence_count": self.before_evidence_count,
            "after_evidence_count": self.after_evidence_count,
            "duplicate_edges_coalesced": self.duplicate_edges_coalesced,
            "self_loops_removed": self.self_loops_removed,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class RelationMergeReviewReport:
    scope: str | None
    approval_id: str | None
    passed: bool
    reviewed: int
    pass_count: int
    watch_count: int
    fail_count: int
    items: list[RelationMergeReviewItem]
    recommendations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "approval_id": self.approval_id,
            "passed": self.passed,
            "reviewed": self.reviewed,
            "pass_count": self.pass_count,
            "watch_count": self.watch_count,
            "fail_count": self.fail_count,
            "items": [item.as_dict() for item in self.items],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        status = "pass" if self.passed else "watch"
        lines = [
            f"# Ara Relation Merge Review: {scope}",
            f"status: {status}",
            f"reviewed={self.reviewed}, pass={self.pass_count}, watch={self.watch_count}, fail={self.fail_count}",
        ]
        if not self.items:
            lines.append("- No relation merge witnesses matched the review filter.")
        for item in self.items[:20]:
            warning = "; ".join(item.warnings) if item.warnings else "witness invariants hold"
            lines.append(
                "- "
                f"[{item.status}] {item.witness_id} approval={item.approval_id} "
                f"edges {item.before_edge_count}->{item.after_edge_count}, "
                f"evidence {item.before_evidence_count}->{item.after_evidence_count}: {warning}"
            )
        if self.recommendations:
            lines.append("## Recommendations")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


@dataclass(slots=True)
class RelationMergeReviewQueueReport:
    scope: str | None
    recorded: int
    resolved: int
    open_items: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "recorded": self.recorded,
            "resolved": self.resolved,
            "open_items": self.open_items,
        }

    def to_text(self) -> str:
        scope = self.scope or "all"
        lines = [
            f"# Ara Relation Merge Review Queue: {scope}",
            f"recorded={self.recorded}, resolved={self.resolved}, open={len(self.open_items)}",
        ]
        if not self.open_items:
            lines.append("- None.")
        for item in self.open_items[:20]:
            lines.append(
                "- "
                f"{item['id']} [{item['review_status']}/{item['action']}] "
                f"priority={float(item['priority']):.2f} {item['reason']}"
            )
        return "\n".join(lines)


def run_relation_merge_dry_run(
    store: MemoryStore,
    *,
    scope: str = "global",
    limit: int = 20,
    threshold: float = 0.72,
    node_limit: int = 800,
    include_global: bool = False,
) -> RelationMergeReport:
    store.init()
    nodes = store.relation_nodes_for_merge(
        scope=scope,
        limit=node_limit,
        include_global=include_global,
    )
    profiles = {str(node["id"]): _node_profile(node) for node in nodes}
    pairs_considered = 0
    candidates: list[RelationMergeCandidate] = []
    node_items = list(profiles.values())
    for index, left in enumerate(node_items):
        if not _is_mergeable_profile(left):
            continue
        for right in node_items[index + 1 :]:
            if left["scope"] != right["scope"]:
                continue
            if not _is_mergeable_profile(right):
                continue
            pairs_considered += 1
            score, reasons = _merge_score(left, right)
            if score < threshold:
                continue
            canonical, candidate = _canonical_pair(left, right)
            candidates.append(
                RelationMergeCandidate(
                    scope=str(canonical["scope"]),
                    canonical_node_id=str(canonical["id"]),
                    candidate_node_id=str(candidate["id"]),
                    canonical_label=str(canonical["label"]),
                    candidate_label=str(candidate["label"]),
                    score=round(score, 4),
                    reasons=reasons,
                    shared_neighbors=len(canonical["neighbors"] & candidate["neighbors"]),
                    canonical_degree=int(canonical["degree"]),
                    candidate_degree=int(candidate["degree"]),
                )
            )
    candidates.sort(
        key=lambda item: (
            -item.score,
            -item.shared_neighbors,
            item.canonical_label,
            item.candidate_label,
        )
    )
    return RelationMergeReport(
        scope=scope,
        candidates=candidates[:limit],
        nodes_considered=len(nodes),
        pairs_considered=pairs_considered,
        threshold=threshold,
        limit=limit,
    )


def prepare_relation_merge_approval(
    store: MemoryStore,
    *,
    scope: str = "global",
    limit: int = 20,
    threshold: float = 0.72,
    node_limit: int = 800,
    include_global: bool = False,
    ttl_minutes: int = 60,
) -> RelationMergeApproval:
    dry_run = run_relation_merge_dry_run(
        store,
        scope=scope,
        limit=limit,
        threshold=threshold,
        node_limit=node_limit,
        include_global=include_global,
    )
    fingerprint = _relation_fingerprint(dry_run)
    witness = _rollback_witness(dry_run)
    if not dry_run.candidates:
        return RelationMergeApproval(
            approval_id=None,
            token=None,
            expires_at=None,
            dry_run=dry_run,
            relation_fingerprint=fingerprint,
            rollback_witness=witness,
        )
    token = secrets.token_urlsafe(24)
    approval_id = new_id("relation_merge_approval")
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
    with store.session() as conn:
        conn.execute(
            """
            INSERT INTO relation_merge_approvals(
              id, token_hash, scope, candidates_json, relation_fingerprint,
              rollback_witness_json, candidate_limit, threshold, node_limit, include_global,
              expires_at, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                _token_hash(token),
                scope,
                json.dumps(
                    [candidate.as_dict() for candidate in dry_run.candidates],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                fingerprint,
                json.dumps(witness, ensure_ascii=False, sort_keys=True),
                limit,
                threshold,
                node_limit,
                1 if include_global else 0,
                expires_at,
                "prepared",
                utc_now(),
            ),
        )
    return RelationMergeApproval(
        approval_id=approval_id,
        token=token,
        expires_at=expires_at,
        dry_run=dry_run,
        relation_fingerprint=fingerprint,
        rollback_witness=witness,
    )


def apply_relation_merge_approval(
    store: MemoryStore,
    *,
    approval_token: str,
    confirmation: str,
) -> RelationMergeApplyReport:
    store.init()
    if confirmation != RELATION_MERGE_CONFIRMATION:
        return RelationMergeApplyReport(
            scope=None,
            passed=False,
            approval_id=None,
            merged=0,
            witness_ids=[],
            recommendations=[f"Confirmation must exactly match: {RELATION_MERGE_CONFIRMATION}"],
        )
    approval = _load_approval(store, approval_token)
    if approval is None:
        return RelationMergeApplyReport(
            scope=None,
            passed=False,
            approval_id=None,
            merged=0,
            witness_ids=[],
            recommendations=["Approval token was not found."],
        )
    if approval["status"] != "prepared":
        return _blocked_apply_report(approval, f"Approval status is {approval['status']}, not prepared.")
    if _is_expired(str(approval["expires_at"])):
        _mark_approval_status(store, str(approval["id"]), "expired")
        return _blocked_apply_report(approval, "Approval token is expired.")

    scope = str(approval["scope"])
    candidate_limit = int(approval["candidate_limit"])
    threshold = float(approval["threshold"])
    node_limit = int(approval["node_limit"])
    include_global = bool(int(approval["include_global"]))
    current = run_relation_merge_dry_run(
        store,
        scope=scope,
        limit=candidate_limit,
        threshold=threshold,
        node_limit=node_limit,
        include_global=include_global,
    )
    current_fingerprint = _relation_fingerprint(current)
    if current_fingerprint != approval["relation_fingerprint"]:
        return _blocked_apply_report(
            approval,
            "Relation merge candidates changed after approval; rerun relation-merge-prepare.",
        )
    approved_candidates = json.loads(str(approval["candidates_json"]))
    if not approved_candidates:
        return _blocked_apply_report(approval, "Approved relation merge has no candidates.")
    if approved_candidates != [candidate.as_dict() for candidate in current.candidates]:
        return _blocked_apply_report(
            approval,
            "Approved relation merge candidate snapshot no longer matches the live dry-run.",
        )

    now = utc_now()
    witness_ids: list[str] = []
    with store.session() as conn:
        status = conn.execute(
            "SELECT status FROM relation_merge_approvals WHERE id = ?",
            (approval["id"],),
        ).fetchone()
        if status is None or status["status"] != "prepared":
            return _blocked_apply_report(approval, "Approval was consumed or changed before apply.")
        for candidate in approved_candidates:
            witness_ids.append(
                _apply_single_candidate(
                    conn,
                    approval_id=str(approval["id"]),
                    candidate=candidate,
                    now=now,
                )
            )
        conn.execute(
            "UPDATE relation_merge_approvals SET status = ?, used_at = ? WHERE id = ?",
            ("used", now, approval["id"]),
        )
    return RelationMergeApplyReport(
        scope=scope,
        passed=True,
        approval_id=str(approval["id"]),
        merged=len(witness_ids),
        witness_ids=witness_ids,
        recommendations=[
            "Relation merge apply consumed the approval token and wrote rollback witnesses.",
            "Run doctor, recall-regression, context-eval, backup, and restore-drill after reviewed relation merges.",
        ],
    )


def review_relation_merge_witnesses(
    store: MemoryStore,
    *,
    scope: str | None = None,
    approval_id: str | None = None,
    limit: int = 50,
) -> RelationMergeReviewReport:
    store.init()
    clauses: list[str] = []
    args: list[Any] = []
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    if approval_id:
        clauses.append("approval_id = ?")
        args.append(approval_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    with store.session() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT *
                FROM relation_merge_witnesses
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                args,
            )
        ]
    items = [_review_witness(row) for row in rows]
    pass_count = sum(1 for item in items if item.status == "pass")
    watch_count = sum(1 for item in items if item.status == "watch")
    fail_count = sum(1 for item in items if item.status == "fail")
    recommendations = [
        "Run recall-regression and context-eval after any watch/fail relation merge review.",
        "Use rollback witnesses for manual repair if a merge loses evidence or leaves candidate references.",
    ]
    if not items:
        recommendations = ["Run relation-merge-apply before reviewing applied merge witnesses."]
    elif fail_count:
        recommendations.insert(0, "Block further relation-merge-apply until failed witnesses are repaired or explained.")
    elif watch_count:
        recommendations.insert(0, "Inspect watched witnesses before applying more relation-node merges.")
    return RelationMergeReviewReport(
        scope=scope,
        approval_id=approval_id,
        passed=fail_count == 0,
        reviewed=len(items),
        pass_count=pass_count,
        watch_count=watch_count,
        fail_count=fail_count,
        items=items,
        recommendations=recommendations,
    )


def record_relation_merge_review_queue(
    store: MemoryStore,
    report: RelationMergeReviewReport,
    *,
    regression: dict[str, Any] | None = None,
    limit: int = 50,
) -> RelationMergeReviewQueueReport:
    store.init()
    now = utc_now()
    recorded = 0
    resolved = 0
    regression_passed = True if regression is None else bool(regression.get("passed"))
    with store.session() as conn:
        for item in report.items:
            if item.status == "pass":
                resolved += _resolve_relation_review_queue(
                    conn,
                    scope=item.scope,
                    witness_id=item.witness_id,
                    now=now,
                )
                continue
            reason = "; ".join(item.warnings) or f"relation merge witness status {item.status}"
            if _upsert_relation_review_queue(
                conn,
                scope=item.scope,
                approval_id=item.approval_id,
                witness_id=item.witness_id,
                action="inspect-relation-merge",
                priority=0.95 if item.status == "fail" else 0.70,
                reason=reason,
                review_status=item.status,
                review_json=item.as_dict(),
                now=now,
            ):
                recorded += 1
        if regression is not None:
            if regression_passed:
                resolved += _resolve_relation_review_queue(
                    conn,
                    scope=report.scope,
                    witness_id=None,
                    now=now,
                    action="inspect-relation-regression",
                )
            elif _upsert_relation_review_queue(
                conn,
                scope=report.scope or "global",
                approval_id=report.approval_id,
                witness_id=None,
                action="inspect-relation-regression",
                priority=0.98,
                reason="relation merge review recall-regression sandbox failed",
                review_status="fail",
                review_json=regression,
                now=now,
            ):
                recorded += 1
        open_items = _list_relation_review_queue(conn, scope=report.scope, status="open", limit=limit)
    return RelationMergeReviewQueueReport(
        scope=report.scope,
        recorded=recorded,
        resolved=resolved,
        open_items=open_items,
    )


def list_relation_merge_review_queue(
    store: MemoryStore,
    *,
    scope: str | None = None,
    status: str = "open",
    limit: int = 50,
) -> RelationMergeReviewQueueReport:
    store.init()
    with store.session() as conn:
        rows = _list_relation_review_queue(conn, scope=scope, status=status, limit=limit)
    return RelationMergeReviewQueueReport(scope=scope, recorded=0, resolved=0, open_items=rows)


def _node_profile(node: dict[str, Any]) -> dict[str, Any]:
    label = str(node.get("label") or "")
    normalized = str(node.get("normalized_label") or label).lower().strip()
    tokens = _tokens(normalized)
    neighbors = set(str(item) for item in node.get("neighbor_keys", []) if str(item))
    return {
        "id": str(node["id"]),
        "scope": str(node["scope"]),
        "label": label,
        "normalized": normalized,
        "tokens": tokens,
        "neighbors": neighbors,
        "degree": int(node.get("degree", 0) or 0),
    }


def _merge_score(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if left["normalized"] == right["normalized"]:
        return 1.0, ["exact-normalized-label"]
    if len(left["tokens"]) < 2 or len(right["tokens"]) < 2:
        return 0.0, []
    token_score = _jaccard(left["tokens"], right["tokens"])
    if token_score >= 0.8:
        reasons.append("high-token-overlap")
    elif token_score >= 0.55:
        reasons.append("partial-token-overlap")
    prefix_score = _prefix_score(str(left["normalized"]), str(right["normalized"]))
    if prefix_score >= 0.85:
        reasons.append("prefix-or-substring")
    neighbor_score = _jaccard(left["neighbors"], right["neighbors"])
    if neighbor_score >= 0.35 and left["neighbors"] and right["neighbors"]:
        reasons.append("shared-relation-neighborhood")
    degree_gap = abs(int(left["degree"]) - int(right["degree"]))
    max_degree = max(1, max(int(left["degree"]), int(right["degree"])))
    degree_balance = 1.0 - min(1.0, degree_gap / max_degree)
    score = (
        0.52 * token_score
        + 0.24 * prefix_score
        + 0.18 * neighbor_score
        + 0.06 * degree_balance
    )
    return score, reasons


def _canonical_pair(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_key = (
        -int(left["degree"]),
        len(str(left["label"])),
        str(left["label"]).lower(),
        str(left["id"]),
    )
    right_key = (
        -int(right["degree"]),
        len(str(right["label"])),
        str(right["label"]).lower(),
        str(right["id"]),
    )
    if left_key <= right_key:
        return left, right
    return right, left


def _is_mergeable_profile(profile: dict[str, Any]) -> bool:
    normalized = str(profile["normalized"])
    if normalized.startswith("-"):
        return False
    if len(normalized) < 6:
        return False
    return len(profile["tokens"]) >= 2


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _token_parts(text.lower()):
        if raw in LOW_VALUE_RELATION_TOKENS:
            continue
        if len(raw) < 3 and not _has_hangul(raw):
            continue
        out.add(raw)
    return out


def _token_parts(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isascii():
            keep = char.isalnum() or char in {"_", "-"}
        else:
            codepoint = ord(char)
            keep = 0xAC00 <= codepoint <= 0xD7A3
        if keep:
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return parts


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(char) <= 0xD7A3 for char in text)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _prefix_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    short, long = sorted((left, right), key=len)
    if short == long:
        return 1.0
    if len(short) >= 5 and short in long:
        return min(1.0, len(short) / max(1, len(long)) + 0.25)
    common = 0
    for a, b in zip(short, long):
        if a != b:
            break
        common += 1
    return common / max(1, len(long))


def _relation_fingerprint(report: RelationMergeReport) -> str:
    payload = {
        "scope": report.scope,
        "threshold": report.threshold,
        "limit": report.limit,
        "nodes_considered": report.nodes_considered,
        "pairs_considered": report.pairs_considered,
        "candidates": [
            {
                "canonical_node_id": item.canonical_node_id,
                "candidate_node_id": item.candidate_node_id,
                "score": item.score,
                "shared_neighbors": item.shared_neighbors,
                "canonical_degree": item.canonical_degree,
                "candidate_degree": item.candidate_degree,
            }
            for item in report.candidates
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _rollback_witness(report: RelationMergeReport) -> dict[str, Any]:
    return {
        "scope": report.scope,
        "candidate_count": len(report.candidates),
        "candidates": [
            {
                "canonical_node_id": item.canonical_node_id,
                "candidate_node_id": item.candidate_node_id,
                "canonical_label": item.canonical_label,
                "candidate_label": item.candidate_label,
                "score": item.score,
                "reasons": item.reasons,
                "shared_neighbors": item.shared_neighbors,
                "canonical_degree": item.canonical_degree,
                "candidate_degree": item.candidate_degree,
            }
            for item in report.candidates
        ],
    }


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _load_approval(store: MemoryStore, token: str) -> dict[str, Any] | None:
    with store.session() as conn:
        row = conn.execute(
            "SELECT * FROM relation_merge_approvals WHERE token_hash = ?",
            (_token_hash(token),),
        ).fetchone()
        return dict(row) if row else None


def _mark_approval_status(store: MemoryStore, approval_id: str, status: str) -> None:
    with store.session() as conn:
        conn.execute(
            "UPDATE relation_merge_approvals SET status = ? WHERE id = ?",
            (status, approval_id),
        )


def _blocked_apply_report(approval: dict[str, Any], message: str) -> RelationMergeApplyReport:
    return RelationMergeApplyReport(
        scope=approval.get("scope"),
        passed=False,
        approval_id=approval.get("id"),
        merged=0,
        witness_ids=[],
        recommendations=[message],
    )


def _is_expired(expires_at: str) -> bool:
    try:
        parsed = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= parsed


def _apply_single_candidate(
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    candidate: dict[str, Any],
    now: str,
) -> str:
    scope = str(candidate["scope"])
    canonical_id = str(candidate["canonical_node_id"])
    candidate_id = str(candidate["candidate_node_id"])
    canonical = _load_relation_node(conn, canonical_id)
    merge_node = _load_relation_node(conn, candidate_id)
    if canonical is None or merge_node is None:
        raise ValueError("Approved relation merge node no longer exists.")
    if canonical["scope"] != scope or merge_node["scope"] != scope:
        raise ValueError("Approved relation merge node scope changed.")
    before = {
        "canonical_node": dict(canonical),
        "candidate_node": dict(merge_node),
        "edges": _affected_edges(conn, scope=scope, node_ids=[canonical_id, candidate_id]),
        "approved_candidate": candidate,
    }
    for edge in _candidate_edges(conn, scope=scope, candidate_node_id=candidate_id):
        _rewire_candidate_edge(
            conn,
            edge=edge,
            canonical_node_id=canonical_id,
            candidate_node_id=candidate_id,
            now=now,
        )
    conn.execute("DELETE FROM relation_nodes WHERE id = ?", (candidate_id,))
    after = {
        "canonical_node": _load_relation_node(conn, canonical_id),
        "candidate_node": _load_relation_node(conn, candidate_id),
        "edges": _affected_edges(conn, scope=scope, node_ids=[canonical_id]),
    }
    witness_id = new_id("relation_merge_witness")
    conn.execute(
        """
        INSERT INTO relation_merge_witnesses(
          id, approval_id, scope, canonical_node_id, candidate_node_id,
          action, reason, before_json, after_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            witness_id,
            approval_id,
            scope,
            canonical_id,
            candidate_id,
            "merge-relation-node",
            "approved one-use relation merge",
            json.dumps(before, ensure_ascii=False, sort_keys=True),
            json.dumps(after, ensure_ascii=False, sort_keys=True),
            now,
        ),
    )
    return witness_id


def _load_relation_node(conn: sqlite3.Connection, node_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM relation_nodes WHERE id = ?", (node_id,)).fetchone()
    return dict(row) if row else None


def _affected_edges(
    conn: sqlite3.Connection,
    *,
    scope: str,
    node_ids: list[str],
) -> list[dict[str, Any]]:
    if not node_ids:
        return []
    placeholders = ", ".join("?" for _ in node_ids)
    rows = conn.execute(
        f"""
        SELECT *
        FROM relation_edges
        WHERE scope = ?
          AND (subject_node_id IN ({placeholders}) OR object_node_id IN ({placeholders}))
        ORDER BY id
        """,
        tuple([scope, *node_ids, *node_ids]),
    )
    return [dict(row) for row in rows]


def _candidate_edges(
    conn: sqlite3.Connection,
    *,
    scope: str,
    candidate_node_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM relation_edges
        WHERE scope = ?
          AND (subject_node_id = ? OR object_node_id = ?)
        ORDER BY id
        """,
        (scope, candidate_node_id, candidate_node_id),
    )
    return [dict(row) for row in rows]


def _rewire_candidate_edge(
    conn: sqlite3.Connection,
    *,
    edge: dict[str, Any],
    canonical_node_id: str,
    candidate_node_id: str,
    now: str,
) -> None:
    subject_id = str(edge["subject_node_id"])
    object_id = str(edge["object_node_id"])
    new_subject = canonical_node_id if subject_id == candidate_node_id else subject_id
    new_object = canonical_node_id if object_id == candidate_node_id else object_id
    if new_subject == new_object:
        conn.execute("DELETE FROM relation_edges WHERE id = ?", (edge["id"],))
        return
    new_id = _relation_edge_id(
        subject_node_id=new_subject,
        predicate_norm=str(edge["predicate_norm"]),
        object_node_id=new_object,
        scope=str(edge["scope"]),
        source_capsule_id=edge["source_capsule_id"],
    )
    existing = conn.execute("SELECT * FROM relation_edges WHERE id = ?", (new_id,)).fetchone()
    if existing is not None and str(existing["id"]) != str(edge["id"]):
        conn.execute(
            """
            UPDATE relation_edges
            SET confidence = max(confidence, ?),
                evidence_count = evidence_count + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                float(edge["confidence"]),
                int(edge["evidence_count"]),
                now,
                new_id,
            ),
        )
        conn.execute("DELETE FROM relation_edges WHERE id = ?", (edge["id"],))
        return
    conn.execute(
        """
        UPDATE relation_edges
        SET id = ?,
            subject_node_id = ?,
            object_node_id = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (new_id, new_subject, new_object, now, edge["id"]),
    )


def _relation_edge_id(
    *,
    subject_node_id: str,
    predicate_norm: str,
    object_node_id: str,
    scope: str,
    source_capsule_id: str | None,
) -> str:
    source = source_capsule_id or ""
    digest = sha256(
        f"{subject_node_id}\0{predicate_norm}\0{object_node_id}\0{scope}\0{source}".encode("utf-8")
    ).hexdigest()[:18]
    return f"rel_edge_{digest}"


def _review_witness(row: dict[str, Any]) -> RelationMergeReviewItem:
    warnings: list[str] = []
    before = json.loads(str(row["before_json"]))
    after = json.loads(str(row["after_json"]))
    before_edges = _json_edges(before)
    after_edges = _json_edges(after)
    before_evidence = _edge_evidence_total(before_edges)
    after_evidence = _edge_evidence_total(after_edges)
    candidate_id = str(row["candidate_node_id"])
    canonical_id = str(row["canonical_node_id"])
    before_candidate_refs = _edge_node_refs(before_edges, candidate_id)
    after_candidate_refs = _edge_node_refs(after_edges, candidate_id)
    after_canonical_refs = _edge_node_refs(after_edges, canonical_id)
    duplicate_edges_coalesced = max(
        0,
        _edge_signature_duplicate_count(before_edges, candidate_id=candidate_id, canonical_id=canonical_id)
        - _edge_signature_duplicate_count(after_edges, candidate_id=candidate_id, canonical_id=canonical_id),
    )
    after_self_loops = _self_loop_count(after_edges)
    self_loops_removed = max(0, _self_loop_count(before_edges) - after_self_loops)
    if not before.get("candidate_node"):
        warnings.append("before snapshot is missing candidate_node")
    if after.get("candidate_node") is not None:
        warnings.append("candidate node still exists after merge")
    if before_candidate_refs <= 0:
        warnings.append("before snapshot had no candidate edge references")
    if after_candidate_refs > 0:
        warnings.append("candidate edge references remain after merge")
    if after_canonical_refs <= 0 and before_edges:
        warnings.append("canonical node has no affected edges after merge")
    if after_evidence < before_evidence:
        warnings.append("relation evidence_count decreased")
    if after_self_loops > 0:
        warnings.append("self-loop relation edge remained after merge")
    status = "pass"
    if warnings:
        status = "fail" if any(_is_failed_review_warning(item) for item in warnings) else "watch"
    return RelationMergeReviewItem(
        witness_id=str(row["id"]),
        approval_id=str(row["approval_id"]),
        scope=str(row["scope"]),
        canonical_node_id=canonical_id,
        candidate_node_id=candidate_id,
        status=status,
        before_edge_count=len(before_edges),
        after_edge_count=len(after_edges),
        before_evidence_count=before_evidence,
        after_evidence_count=after_evidence,
        duplicate_edges_coalesced=duplicate_edges_coalesced,
        self_loops_removed=self_loops_removed,
        warnings=warnings,
    )


def _json_edges(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    edges = snapshot.get("edges")
    if not isinstance(edges, list):
        return []
    return [dict(item) for item in edges if isinstance(item, dict)]


def _edge_evidence_total(edges: list[dict[str, Any]]) -> int:
    total = 0
    for edge in edges:
        try:
            total += int(edge.get("evidence_count") or 0)
        except (TypeError, ValueError):
            continue
    return total


def _edge_node_refs(edges: list[dict[str, Any]], node_id: str) -> int:
    return sum(
        1
        for edge in edges
        if str(edge.get("subject_node_id")) == node_id or str(edge.get("object_node_id")) == node_id
    )


def _edge_signature_duplicate_count(
    edges: list[dict[str, Any]],
    *,
    candidate_id: str,
    canonical_id: str,
) -> int:
    seen: dict[tuple[str, str, str, str], int] = {}
    for edge in edges:
        subject = canonical_id if str(edge.get("subject_node_id")) == candidate_id else str(edge.get("subject_node_id"))
        object_ = canonical_id if str(edge.get("object_node_id")) == candidate_id else str(edge.get("object_node_id"))
        key = (
            subject,
            str(edge.get("predicate_norm")),
            object_,
            str(edge.get("source_capsule_id") or ""),
        )
        seen[key] = seen.get(key, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


def _self_loop_count(edges: list[dict[str, Any]]) -> int:
    return sum(1 for edge in edges if str(edge.get("subject_node_id")) == str(edge.get("object_node_id")))


def _is_failed_review_warning(warning: str) -> bool:
    return warning in {
        "candidate node still exists after merge",
        "candidate edge references remain after merge",
        "relation evidence_count decreased",
        "self-loop relation edge remained after merge",
    }


def _upsert_relation_review_queue(
    conn: sqlite3.Connection,
    *,
    scope: str,
    approval_id: str | None,
    witness_id: str | None,
    action: str,
    priority: float,
    reason: str,
    review_status: str,
    review_json: dict[str, Any],
    now: str,
) -> bool:
    existing = conn.execute(
        f"""
        SELECT id
        FROM relation_merge_review_queue
        WHERE scope = ?
          AND action = ?
          AND status = 'open'
          AND {'witness_id IS NULL' if witness_id is None else 'witness_id = ?'}
        LIMIT 1
        """,
        (scope, action) if witness_id is None else (scope, action, witness_id),
    ).fetchone()
    payload = json.dumps(review_json, ensure_ascii=False, sort_keys=True)
    if existing:
        conn.execute(
            """
            UPDATE relation_merge_review_queue
            SET priority = ?, reason = ?, review_status = ?, review_json = ?, created_at = ?
            WHERE id = ?
            """,
            (priority, reason, review_status, payload, now, existing["id"]),
        )
        return False
    conn.execute(
        """
        INSERT INTO relation_merge_review_queue(
          id, scope, approval_id, witness_id, action, priority, reason,
          review_status, review_json, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id("relation_review"),
            scope,
            approval_id,
            witness_id,
            action,
            priority,
            reason,
            review_status,
            payload,
            "open",
            now,
        ),
    )
    return True


def _resolve_relation_review_queue(
    conn: sqlite3.Connection,
    *,
    scope: str | None,
    witness_id: str | None,
    now: str,
    action: str | None = None,
) -> int:
    clauses = ["status = 'open'"]
    where_args: list[Any] = []
    if scope:
        clauses.append("scope = ?")
        where_args.append(scope)
    if witness_id is None:
        clauses.append("witness_id IS NULL")
    else:
        clauses.append("witness_id = ?")
        where_args.append(witness_id)
    if action:
        clauses.append("action = ?")
        where_args.append(action)
    cur = conn.execute(
        f"""
        UPDATE relation_merge_review_queue
        SET status = ?, resolved_at = ?
        WHERE {' AND '.join(clauses)}
        """,
        ("resolved", now, *where_args),
    )
    return cur.rowcount


def _list_relation_review_queue(
    conn: sqlite3.Connection,
    *,
    scope: str | None,
    status: str,
    limit: int,
) -> list[dict[str, Any]]:
    clauses = ["status = ?"]
    args: list[Any] = [status]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    args.append(limit)
    return [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM relation_merge_review_queue
            WHERE {' AND '.join(clauses)}
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            args,
        )
    ]
