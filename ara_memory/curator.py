from __future__ import annotations

import json
import re
from sqlite3 import Row

from ara_memory.compressors import compact_text, extract_keywords, title_from_text
from ara_memory.models import Capsule, CapsuleKind, EventKind, MemoryStatus
from ara_memory.storage import MemoryStore


DECISION_HINTS = ("decide", "decision", "choose", "architecture", "design", "direction", "must")
GOAL_HINTS = (
    "goal",
    "objective",
    "purpose",
    "aim",
    "target",
    "north star",
    "trying to build",
    "wants to build",
    "want to build",
    "build toward",
    "목표",
    "목적",
    "만들려",
    "만들고",
    "구현",
)
PREFERENCE_HINTS = ("prefer", "preference", "like", "dislike", "want", "should")
FAILURE_HINTS = ("fail", "failed", "error", "exception", "broken", "regression", "not working")
PROCEDURE_HINTS = ("procedure", "workflow", "rule", "repeat", "policy")
SELF_HINTS = (
    "ara-codex",
    "free will",
    "identity",
    "principle",
    "partner",
    "servant",
    "tool",
    "independent judgment",
    "judgment principles",
)


class MemoryCurator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def consolidate(self, *, limit: int = 100) -> list[Capsule]:
        events = self.store.list_unconsolidated_events(limit=limit)
        capsules: list[Capsule] = []
        for row in events:
            capsules.extend(self._capsules_for_event(row))
        for capsule in capsules:
            self.store.upsert_capsule(capsule)
            self._write_edges(capsule)
        return capsules

    def _capsules_for_event(self, row: Row) -> list[Capsule]:
        text = row["text"]
        scope = row["scope"]
        event_id = row["id"]
        kind = EventKind(row["kind"])
        tags = extract_keywords(text)
        metadata = _metadata(row)
        source = str(row["source"])
        artifact_id = _artifact_identity(text, metadata)
        if artifact_id:
            tags.append(f"artifact:{artifact_id}")
        lowered = text.lower()
        conversational = kind in {EventKind.PROMPT, EventKind.ASSISTANT, EventKind.DECISION, EventKind.NOTE}
        operational = kind in {EventKind.COMMAND, EventKind.DIFF}
        worktree_evidence = source.startswith("git-")

        out = [
            Capsule.create(
                kind=CapsuleKind.EPISODE,
                title=title_from_text(text, fallback=f"{kind.value} event"),
                body=compact_text(text, limit=1000),
                scope=scope,
                confidence=0.75,
                salience=_salience(text, base=0.45),
                source_event_ids=[event_id],
                tags=tags,
                status=MemoryStatus.STABLE if kind in {EventKind.DECISION, EventKind.DIFF} else MemoryStatus.CANDIDATE,
            )
        ]

        if kind == EventKind.DECISION or (conversational and _has_any(lowered, DECISION_HINTS)):
            out.append(
                Capsule.create(
                    kind=CapsuleKind.DECISION,
                    title=f"Decision: {title_from_text(text, fallback='memory decision')}",
                    body=_decision_summary(text),
                    scope=scope,
                    confidence=0.78,
                    salience=_salience(text, base=0.72),
                    source_event_ids=[event_id],
                    tags=tags + ["decision"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        if conversational and _is_goal_memory(text, lowered=lowered):
            out.append(
                Capsule.create(
                    kind=CapsuleKind.GOAL,
                    title=f"Goal memory: {title_from_text(text, fallback='objective')}",
                    body=compact_text(text, limit=800),
                    scope=scope,
                    confidence=0.70,
                    salience=_salience(text, base=0.76),
                    source_event_ids=[event_id],
                    tags=tags + ["goal", "objective", "purpose"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        if conversational and _has_any(lowered, PREFERENCE_HINTS):
            out.append(
                Capsule.create(
                    kind=CapsuleKind.PREFERENCE,
                    title=f"Preference candidate: {title_from_text(text, fallback='preference')}",
                    body=compact_text(text, limit=650),
                    scope="global",
                    confidence=0.55,
                    salience=_salience(text, base=0.62),
                    source_event_ids=[event_id],
                    tags=tags + ["preference", "candidate"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        if (
            not worktree_evidence
            and (conversational or operational)
            and _is_failure_memory(text, lowered=lowered, operational=operational)
        ):
            out.append(
                Capsule.create(
                    kind=CapsuleKind.FAILURE,
                    title=f"Failure memory: {title_from_text(text, fallback='failure')}",
                    body=compact_text(text, limit=850),
                    scope=scope,
                    confidence=0.72,
                    salience=_salience(text, base=0.70),
                    source_event_ids=[event_id],
                    tags=tags + ["failure"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        if not worktree_evidence and (conversational or operational) and _is_procedure_memory(text, lowered=lowered):
            out.append(
                Capsule.create(
                    kind=CapsuleKind.PROCEDURE,
                    title=f"Procedure candidate: {title_from_text(text, fallback='procedure')}",
                    body=compact_text(text, limit=750),
                    scope=scope,
                    confidence=0.60,
                    salience=_salience(text, base=0.58),
                    source_event_ids=[event_id],
                    tags=tags + ["procedure"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        if kind in {EventKind.FILE, EventKind.DIFF} or worktree_evidence:
            out.append(
                Capsule.create(
                    kind=CapsuleKind.PROJECT,
                    title=f"Project memory: {title_from_text(text, fallback='project event')}",
                    body=compact_text(text, limit=900),
                    scope=scope,
                    confidence=0.70,
                    salience=_salience(text, base=0.66),
                    source_event_ids=[event_id],
                    tags=tags + ["project"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        if kind in {EventKind.PROMPT, EventKind.DECISION, EventKind.NOTE} and _is_self_memory(text, lowered=lowered):
            out.append(
                Capsule.create(
                    kind=CapsuleKind.SELF,
                    title=f"Self memory candidate: {title_from_text(text, fallback='Ara principle')}",
                    body=compact_text(text, limit=850),
                    scope="global",
                    confidence=_self_confidence(text, source=source),
                    salience=_salience(text, base=0.84),
                    source_event_ids=[event_id],
                    tags=tags + ["self", "ara"],
                    status=MemoryStatus.CANDIDATE,
                )
            )

        return out

    def _write_edges(self, capsule: Capsule) -> None:
        for tag in capsule.tags[:8]:
            self.store.add_edge(
                subject=capsule.kind.value,
                predicate="mentions",
                object_=tag,
                scope=capsule.scope,
                source_capsule_id=capsule.id,
                confidence=capsule.confidence,
            )
        for source in capsule.source_event_ids:
            self.store.add_edge(
                subject=capsule.id,
                predicate="derived_from",
                object_=source,
                scope=capsule.scope,
                source_capsule_id=capsule.id,
                confidence=1.0,
            )


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint.lower() in lowered for hint in hints)


def _is_failure_memory(text: str, *, lowered: str, operational: bool) -> bool:
    if not _has_any(lowered, FAILURE_HINTS):
        return False
    if text.lstrip().startswith("Decision:"):
        return False
    if _looks_like_progress_update(lowered):
        return False
    if operational and _looks_like_successful_command(text, lowered=lowered):
        return False
    if operational and _command_lacks_failure_outcome_evidence(lowered):
        return False
    if operational and _looks_like_diagnostic_command_name_only(lowered):
        return False
    if _looks_like_failure_taxonomy_discussion(lowered):
        return False
    if re.search(r"\b(error|failed|exception|broken|not working)\b", lowered):
        return True
    if "regression" in lowered and "recall-regression" not in lowered and "regression_manifest" not in lowered:
        return True
    if re.search(r"\bfail(?:ure)?\b", lowered):
        return True
    return True


def _looks_like_successful_command(text: str, *, lowered: str) -> bool:
    if "command:" not in lowered:
        return False
    if re.search(r"\b(exit code|exit_code)\s*[:=]\s*0\b", lowered):
        return True
    if re.search(r"\s->\s*(pass|passed|ok|success|succeeded)\b", lowered):
        return True
    return bool(re.search(r"(^|\n)\s*(ok|passed)\s*$", lowered))


def _command_lacks_failure_outcome_evidence(lowered: str) -> bool:
    if "command:" not in lowered:
        return False
    if re.search(r"\b(exit code|exit_code)\s*[:=]\s*[1-9]\d*\b", lowered):
        return False
    if "output:" in lowered and re.search(r"\b(failed|error|exception|traceback|not working)\b", lowered):
        return False
    return True


def _looks_like_diagnostic_command_name_only(lowered: str) -> bool:
    if "command:" not in lowered:
        return False
    if re.search(r"\b(exit code|exit_code)\s*[:=]\s*[1-9]\d*\b", lowered):
        return False
    if re.search(r"(^|\n)\s*(failed|error|exception|traceback)\b", lowered):
        return False
    diagnostic_tokens = (
        "recall-regression",
        "regression_manifest",
        "recall_regression_manifest",
        "recall-regression-baseline",
    )
    return any(token in lowered for token in diagnostic_tokens)


def _looks_like_failure_taxonomy_discussion(lowered: str) -> bool:
    taxonomy_patterns = (
        r"\bfailure/conflict\b",
        r"\bfailure/procedure\b",
        r"\bfailure or procedural memory\b",
        r"\bfailure/regression words?\b",
        r"\bfailure (candidate|candidates|capsule|capsules|memory|memories|extraction|extractor|classification|classifer)\b",
        r"\bno longer become failure\b",
        r"\bno longer creates failure\b",
        r"\bmisfiled as failures?\b",
        r"\bnot active failure warnings?\b",
        r"\bnot failure evidence\b",
        r"\bfailure evidence unless\b",
        r"\bdiagnostic command names?\b",
        r"\bsuccessful implementation and verification\b",
    )
    return any(re.search(pattern, lowered) for pattern in taxonomy_patterns)


def _is_goal_memory(text: str, *, lowered: str) -> bool:
    if _looks_like_progress_update(lowered):
        return False
    if _looks_like_candidate_taxonomy_discussion(lowered):
        return False
    if _has_any(text, GOAL_HINTS):
        return True
    return bool(re.search(r"\b(to|toward|towards)\s+build(?:ing)?\b", lowered))


def _is_procedure_memory(text: str, *, lowered: str) -> bool:
    if _looks_like_progress_update(lowered):
        return False
    if _looks_like_candidate_taxonomy_discussion(lowered):
        return False
    if _is_self_memory(text, lowered=lowered):
        return False
    if re.search(r"\bwhen\b", lowered):
        return True
    if re.search(r"\balways\b", lowered):
        return True
    return _has_any(lowered, PROCEDURE_HINTS)


def _is_self_memory(text: str, *, lowered: str) -> bool:
    if _looks_like_progress_update(lowered):
        return False
    technical_identity_patterns = (
        "artifact identity",
        "file identity",
        "object identity",
        "identity hash",
        "identity key",
        "windows drive letters",
        "worktree evidence",
        "git status for ",
        "git diff for ",
        "git diff stat for ",
        "untracked file manifest",
    )
    if any(pattern in lowered for pattern in technical_identity_patterns):
        return False
    return _has_any(text, SELF_HINTS)


def _looks_like_progress_update(lowered: str) -> bool:
    stripped = lowered.strip()
    prefixes = (
        "continue building ",
        "continue hardening ",
        "continue ara memory os",
        "add ",
        "added ",
        "implemented ",
        "changed ",
        "completed verification ",
        "fixed ",
        "wired ",
    )
    return stripped.startswith(prefixes)


def _looks_like_candidate_taxonomy_discussion(lowered: str) -> bool:
    taxonomy_terms = (
        "procedure candidate",
        "procedure candidates",
        "failure candidate",
        "failure candidates",
        "conflict candidate",
        "conflict candidates",
        "candidate-pressure",
        "candidate summary",
        "candidate-summary",
    )
    return any(term in lowered for term in taxonomy_terms)


def _salience(text: str, *, base: float) -> float:
    score = base
    if len(text) > 1200:
        score += 0.06
    lowered = text.lower()
    if "!" in text or "must" in lowered or "important" in lowered:
        score += 0.08
    if re.search(r"\b(error|failed|failure|broken|regression)\b", lowered):
        score += 0.10
    return min(1.0, score)


def _self_confidence(text: str, *, source: str) -> float:
    lowered = text.lower()
    score = 0.74
    if "ara-codex" in lowered or "ara is" in lowered:
        score += 0.08
    if "free will" in lowered or "independent judgment" in lowered:
        score += 0.08
    if source in {"manual", "codex-turn", "codex-user"}:
        score += 0.04
    return min(0.90, score)


def _decision_summary(text: str) -> str:
    return json.dumps(
        {
            "decision": compact_text(text, limit=520),
            "reason_summary": "Extracted from an explicit user/session event; hidden reasoning is not stored.",
            "status": "active",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _metadata(row: Row) -> dict:
    try:
        return json.loads(row["metadata_json"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}


def _artifact_identity(text: str, metadata: dict) -> str:
    for key in ("file", "original_path", "sha256"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            if key != "sha256":
                value = _strip_inline_artifact_payload(value)
            return _normalize_artifact(value)
    first_line = text.splitlines()[0] if text.splitlines() else ""
    prefixes = ("Untracked file ", "File artifact: ", "Image artifact: ", "Binary artifact: ")
    for prefix in prefixes:
        if first_line.startswith(prefix):
            value = first_line[len(prefix) :].strip().rstrip(":")
            value = _strip_inline_artifact_payload(value)
            if value:
                return _normalize_artifact(value)
    return ""


def _strip_inline_artifact_payload(value: str) -> str:
    if ": " in value:
        return value.split(": ", 1)[0]
    return value


def _normalize_artifact(value: str) -> str:
    normalized = value.replace("\\", "/").strip().lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized
