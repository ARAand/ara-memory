from __future__ import annotations

import re
import json
from collections import defaultdict
from dataclasses import dataclass

from ara_memory.advisor import MemoryAdvisor, MemoryReviewEngine
from ara_memory.compressors import compact_text
from ara_memory.models import Capsule, CapsuleKind, MemoryStatus, new_id
from ara_memory.storage import MemoryStore, row_to_capsule


MERGE_KINDS = {"project", "procedure", "failure", "fact", "goal"}
CONFLICT_KINDS = {"decision", "preference", "procedure", "self", "fact", "goal"}
NEGATION_MARKERS = (" not ", " never ", "avoid ", "do not ", "don't ", "instead of ", "rather than ")
TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]{3,}")


@dataclass(slots=True)
class SleepReport:
    run_id: str
    scope: str
    candidates_seen: int
    promoted: int = 0
    rejected: int = 0
    quarantined: int = 0
    merged: int = 0
    conflicts: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "run_id": self.run_id,
            "scope": self.scope,
            "candidates_seen": self.candidates_seen,
            "promoted": self.promoted,
            "rejected": self.rejected,
            "quarantined": self.quarantined,
            "merged": self.merged,
            "conflicts": self.conflicts,
        }


class SleepConsolidator:
    def __init__(self, store: MemoryStore, advisor: MemoryAdvisor | None = None) -> None:
        self.store = store
        self.review = MemoryReviewEngine(store, advisor=advisor)

    def run(self, *, scope: str = "global", dry_run: bool = False) -> SleepReport:
        candidates = [
            row_to_capsule(row)
            for row in self.store.list_capsules(scope=scope, status=MemoryStatus.CANDIDATE, limit=500)
        ]
        stable = [
            row_to_capsule(row)
            for row in self.store.list_capsules(scope=scope, status=MemoryStatus.STABLE, limit=500)
        ]
        report = SleepReport(run_id=new_id("sleep"), scope=scope, candidates_seen=len(candidates))
        recommendations = self.review.advisor.review(candidates)
        quarantined_ids: set[str] = set()
        promote_ids: set[str] = set()

        for rec in recommendations:
            if rec.action == "quarantine":
                quarantined_ids.add(rec.capsule_id)
                report.quarantined += 1
                if not dry_run:
                    self.store.update_capsule_status(
                        rec.capsule_id,
                        MemoryStatus.QUARANTINED,
                        actor=rec.actor,
                        reason=rec.reason,
                    )
            elif rec.action == "promote":
                promote_ids.add(rec.capsule_id)

        candidates = [cap for cap in candidates if cap["id"] not in quarantined_ids]

        malformed_summaries = _malformed_artifact_summaries(stable)
        if malformed_summaries:
            report.rejected += len(malformed_summaries)
            if not dry_run:
                for cap in malformed_summaries:
                    self.store.update_capsule_status(
                        cap["id"],
                        MemoryStatus.SUPERSEDED,
                        actor="sleep-consolidator",
                        reason="malformed artifact identity summary",
                    )

        stale_summaries = _stale_artifact_summaries(stable)
        if stale_summaries:
            report.rejected += len(stale_summaries)
            if not dry_run:
                for cap in stale_summaries:
                    self.store.update_capsule_status(
                        cap["id"],
                        MemoryStatus.SUPERSEDED,
                        actor="sleep-consolidator",
                        reason="superseded by latest artifact summary",
                    )
        by_signature: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for cap in candidates:
            by_signature[(_signature(cap), cap["kind"])].append(cap)

        merged_ids: set[str] = set()
        for group in by_signature.values():
            if len(group) >= 2 and group[0]["kind"] in MERGE_KINDS:
                report.merged += 1
                merged_ids.update(cap["id"] for cap in group)
                if not dry_run:
                    self._merge_group(group, scope=scope)

        artifact_groups = _artifact_groups(candidates + stable)
        for artifact, group in artifact_groups.items():
            if len(group) < 2:
                continue
            report.merged += 1
            if not dry_run:
                self._merge_artifact_group(artifact, group, scope=scope)

        for cap in candidates:
            if cap["id"] in merged_ids:
                continue
            if cap["id"] in promote_ids:
                report.promoted += 1
                if not dry_run:
                    self.store.update_capsule_status(
                        cap["id"],
                        MemoryStatus.STABLE,
                        actor="memory-curator",
                        reason="advisor recommended promotion",
                    )

        conflicts = [
            (left, right)
            for left, right in _detect_conflicts(candidates)
            if not self._conflict_exists(left, right, scope=scope)
        ]
        report.conflicts = len(conflicts)
        if not dry_run:
            for left, right in conflicts:
                self._record_conflict(left, right, scope=scope)
            self.store.record_consolidation_run(
                run_id=report.run_id,
                scope=scope,
                candidates_seen=report.candidates_seen,
                promoted=report.promoted,
                rejected=report.rejected,
                merged=report.merged,
                conflicts=report.conflicts,
                notes=f"dry_run=false; quarantined={report.quarantined}",
            )
        return report

    def _merge_group(self, group: list[dict], *, scope: str) -> None:
        source_ids: list[str] = []
        bodies: list[str] = []
        seen_bodies: set[str] = set()
        tags: set[str] = set()
        for cap in group:
            source_ids.extend(cap["source_event_ids"])
            normalized = " ".join(cap["body"].split())
            if normalized not in seen_bodies:
                seen_bodies.add(normalized)
                bodies.append(cap["body"])
            tags.update(cap["tags"])
            self.store.update_capsule_status(
                cap["id"],
                MemoryStatus.SUPERSEDED,
                actor="sleep-consolidator",
                reason="merged into summary capsule",
            )
        summary = Capsule.create(
            kind=CapsuleKind.SUMMARY,
            title=f"Consolidated {group[0]['kind']} memory: {group[0]['title'][:72]}",
            body=compact_text("\n".join(bodies), limit=1200),
            scope=scope,
            confidence=max(cap["confidence"] for cap in group),
            salience=max(cap["salience"] for cap in group),
            source_event_ids=sorted(set(source_ids)),
            tags=sorted(tags | {"consolidated", group[0]["kind"]}),
            status=MemoryStatus.STABLE,
        )
        self.store.upsert_capsule(summary)

    def _record_conflict(self, left: dict, right: dict, *, scope: str) -> None:
        if self._conflict_exists(left, right, scope=scope):
            return
        conflict = Capsule.create(
            kind=CapsuleKind.CONFLICT,
            title=f"Potential memory conflict: {left['title'][:48]}",
            body=(
                "Potential contradiction detected between two candidate memories.\n"
                f"Left: {left['body']}\n"
                f"Right: {right['body']}"
            ),
            scope=scope,
            confidence=0.45,
            salience=max(left["salience"], right["salience"], 0.75),
            source_event_ids=sorted(set(left["source_event_ids"] + right["source_event_ids"])),
            tags=sorted(set(left["tags"] + right["tags"] + ["conflict"])),
            status=MemoryStatus.CANDIDATE,
        )
        self.store.upsert_capsule(conflict)

    def _conflict_exists(self, left: dict, right: dict, *, scope: str) -> bool:
        signature = _conflict_signature(left, right)
        with self.store.session() as conn:
            rows = conn.execute(
                """
                SELECT source_event_ids_json
                FROM capsules
                WHERE scope = ?
                  AND kind = ?
                  AND status IN (?, ?)
                LIMIT 500
                """,
                (
                    scope,
                    CapsuleKind.CONFLICT.value,
                    MemoryStatus.CANDIDATE.value,
                    MemoryStatus.STABLE.value,
                ),
            )
            for row in rows:
                try:
                    source_ids = json.loads(row["source_event_ids_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if "|".join(sorted(str(item) for item in source_ids)) == signature:
                    return True
        return False

    def _merge_artifact_group(self, artifact: str, group: list[dict], *, scope: str) -> None:
        source_ids: list[str] = []
        bodies: list[str] = []
        tags: set[str] = {f"artifact:{artifact}", "artifact-consolidated"}
        ordered = sorted(group, key=lambda cap: cap["updated_at"])
        for cap in ordered:
            source_ids.extend(cap["source_event_ids"])
            bodies.append(cap["body"])
            tags.update(cap["tags"])
            self.store.update_capsule_status(
                cap["id"],
                MemoryStatus.SUPERSEDED,
                actor="sleep-consolidator",
                reason=f"superseded by artifact summary for {artifact}",
            )
        latest = ordered[-1]
        summary = Capsule.create(
            kind=CapsuleKind.SUMMARY,
            title=f"Latest artifact memory: {artifact}",
            body=compact_text(bodies[-1], limit=1200),
            scope=scope,
            confidence=max(cap["confidence"] for cap in group),
            salience=max(cap["salience"] for cap in group),
            source_event_ids=sorted(set(source_ids)),
            tags=sorted(tags),
            status=MemoryStatus.STABLE,
        )
        self._supersede_prior_latest_artifact_summaries(artifact, summary.id, scope=scope)
        self.store.upsert_capsule(summary)

    def _supersede_prior_latest_artifact_summaries(self, artifact: str, new_summary_id: str, *, scope: str) -> None:
        title = f"Latest artifact memory: {artifact}"
        for row in self.store.list_capsules(scope=scope, status=MemoryStatus.STABLE, kind="summary", limit=500):
            cap = row_to_capsule(row)
            if cap["id"] == new_summary_id:
                continue
            if cap["title"].lower() != title.lower():
                continue
            self.store.update_capsule_status(
                cap["id"],
                MemoryStatus.SUPERSEDED,
                actor="sleep-consolidator",
                reason=f"superseded by newer artifact summary for {artifact}",
            )


def _signature(cap: dict) -> str:
    tags = [tag for tag in cap["tags"] if len(tag) >= 4][:5]
    return "|".join(sorted(tags)) or cap["title"][:48].lower()


def _detect_conflicts(candidates: list[dict]) -> list[tuple[dict, dict]]:
    conflicts: list[tuple[dict, dict]] = []
    for i, left in enumerate(candidates):
        if left["kind"] not in CONFLICT_KINDS:
            continue
        left_tags = set(left["tags"])
        if not left_tags:
            continue
        left_neg = _has_negation(left["body"])
        for right in candidates[i + 1 :]:
            if left["kind"] != right["kind"]:
                continue
            overlap = left_tags.intersection(right["tags"])
            body_overlap = _body_tokens(left["body"]).intersection(_body_tokens(right["body"]))
            if len(overlap) < 3 and len(body_overlap) < 5:
                continue
            if left_neg != _has_negation(right["body"]):
                conflicts.append((left, right))
    return conflicts[:25]


def _conflict_signature(left: dict, right: dict) -> str:
    return "|".join(sorted(str(item) for item in left["source_event_ids"] + right["source_event_ids"]))


def _artifact_groups(candidates: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for cap in candidates:
        if cap["kind"] != "project":
            continue
        artifact = _artifact_identity(cap)
        if artifact:
            groups[artifact].append(cap)
    return groups


def _artifact_identity(cap: dict) -> str:
    text = f"{cap['body']}\n{cap['title']}"
    for tag in cap["tags"]:
        if tag.startswith("artifact:"):
            artifact = tag.removeprefix("artifact:")
            if re.fullmatch(r"[a-z]", artifact.strip().lower()) and _looks_like_artifact_body(text):
                continue
            return artifact
    markers = ("Untracked file ", "File artifact: ", "Image artifact: ", "Binary artifact: ")
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        rest = text[idx + len(marker) :]
        first = rest.splitlines()[0].strip().rstrip(":")
        first = _strip_inline_artifact_payload(first)
        if first:
            return first.replace("\\", "/").lower()
    return ""


def _strip_inline_artifact_payload(value: str) -> str:
    if ": " in value:
        return value.split(": ", 1)[0]
    return value


def _malformed_artifact_summaries(capsules: list[dict]) -> list[dict]:
    malformed: list[dict] = []
    prefix = "Latest artifact memory: "
    for cap in capsules:
        if cap["kind"] != "summary" or not cap["title"].startswith(prefix):
            continue
        rest = cap["title"][len(prefix) :]
        if re.fullmatch(r"[a-z]", rest.strip().lower()) and _looks_like_artifact_body(cap["body"]):
            malformed.append(cap)
            continue
        if ": " in rest:
            artifact = rest.split(": ", 1)[0]
            if "/" in artifact or "." in artifact:
                malformed.append(cap)
                continue
        for tag in cap["tags"]:
            if tag.startswith("artifact:") and (" # " in tag or len(tag) > 120):
                malformed.append(cap)
                break
    return malformed


def _looks_like_artifact_body(body: str) -> bool:
    lowered = body.lower()
    return any(marker.lower() in lowered for marker in ("Untracked file ", "File artifact: ", "Image artifact: "))


def _stale_artifact_summaries(capsules: list[dict]) -> list[dict]:
    return _stale_consolidated_summaries(capsules) + _older_latest_artifact_summaries(capsules)


def _stale_consolidated_summaries(capsules: list[dict]) -> list[dict]:
    latest_artifacts = {
        _summary_artifact(cap)
        for cap in capsules
        if cap["kind"] == "summary" and cap["title"].startswith("Latest artifact memory: ")
    }
    latest_artifacts.discard("")
    stale: list[dict] = []
    for cap in capsules:
        if cap["kind"] != "summary" or not cap["title"].startswith("Consolidated project memory: "):
            continue
        artifact = _artifact_identity(cap)
        if artifact and artifact in latest_artifacts:
            stale.append(cap)
    return stale


def _older_latest_artifact_summaries(capsules: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for cap in capsules:
        if cap["kind"] != "summary" or not cap["title"].startswith("Latest artifact memory: "):
            continue
        artifact = _summary_artifact(cap)
        if artifact:
            groups[artifact].append(cap)

    stale: list[dict] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda cap: (cap["updated_at"], cap["salience"], cap["confidence"]), reverse=True)
        stale.extend(ordered[1:])
    return stale


def _summary_artifact(cap: dict) -> str:
    prefix = "Latest artifact memory: "
    if not cap["title"].startswith(prefix):
        return ""
    return cap["title"][len(prefix) :].strip().lower()


def _has_negation(text: str) -> bool:
    lowered = f" {_semantic_text(text).lower()} "
    return any(marker in lowered for marker in NEGATION_MARKERS)


def _body_tokens(text: str) -> set[str]:
    stop = {"decision", "should", "must", "with", "that", "this", "from", "into", "global"}
    semantic = _semantic_text(text)
    return {token.lower() for token in TOKEN_RE.findall(semantic) if token.lower() not in stop}


def _semantic_text(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        for key in ("decision", "fact", "summary", "body"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return text
