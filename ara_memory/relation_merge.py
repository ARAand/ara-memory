from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import secrets
from typing import Any

from ara_memory.models import new_id, utc_now
from ara_memory.storage import MemoryStore


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
              rollback_witness_json, threshold, node_limit, expires_at, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                threshold,
                node_limit,
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
