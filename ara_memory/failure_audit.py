from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ara_memory.models import CapsuleKind, MemoryStatus, utc_now
from ara_memory.storage import MemoryStore


@dataclass(slots=True)
class FailureKindAuditResult:
    scope: str | None
    dry_run: bool
    reviewed: int
    changed: int
    items: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dry_run": self.dry_run,
            "reviewed": self.reviewed,
            "changed": self.changed,
            "items": self.items,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Failure Kind Audit: {self.scope or 'all'}",
            f"dry_run: {self.dry_run}",
            f"reviewed={self.reviewed}, changed={self.changed}",
        ]
        for item in self.items[:20]:
            status = "changed" if item["changed"] else "kept"
            lines.append(
                f"- [{status}] {item['id']} {item['from_kind']} -> {item['to_kind']}: {item['reason']}"
            )
        if len(self.items) > 20:
            lines.append(f"... {len(self.items) - 20} more")
        return "\n".join(lines)


def audit_failure_kinds(
    store: MemoryStore,
    *,
    scope: str | None = None,
    statuses: list[str] | None = None,
    limit: int = 200,
    dry_run: bool = True,
) -> FailureKindAuditResult:
    store.init()
    status_values = statuses or [MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value]
    rows = _failure_rows(store, scope=scope, statuses=status_values, limit=limit)
    items: list[dict[str, Any]] = []
    changed = 0
    for row in rows:
        target, reason = _target_kind(row["body"])
        if target is None:
            items.append(_item(row, CapsuleKind.FAILURE.value, False, "actual failure or unresolved failure evidence"))
            continue
        if not dry_run:
            _update_kind(store, row, target, reason=reason)
        changed += 1
        items.append(_item(row, target.value, True, reason))
    return FailureKindAuditResult(
        scope=scope,
        dry_run=dry_run,
        reviewed=len(rows),
        changed=changed,
        items=items,
    )


def _failure_rows(
    store: MemoryStore,
    *,
    scope: str | None,
    statuses: list[str],
    limit: int,
) -> list[Any]:
    clauses = ["kind = ?"]
    args: list[Any] = [CapsuleKind.FAILURE.value]
    if scope:
        clauses.append("scope = ?")
        args.append(scope)
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        args.extend(statuses)
    args.append(limit)
    with store.session() as conn:
        return list(
            conn.execute(
                f"""
                SELECT *
                FROM capsules
                WHERE {' AND '.join(clauses)}
                ORDER BY status ASC, updated_at DESC
                LIMIT ?
                """,
                args,
            )
        )


def _target_kind(body: str) -> tuple[CapsuleKind | None, str]:
    text = body.strip()
    lower = text.lower()
    if text.startswith("Decision:"):
        return CapsuleKind.DECISION, "decision text misfiled as failure"
    if text.startswith("Command:") and _has_success_marker(lower) and not _has_concrete_failure_evidence(lower):
        return CapsuleKind.EPISODE, "successful command evidence misfiled as failure"
    if _looks_like_successful_turn_episode(lower):
        return CapsuleKind.PROJECT, "successful turn episode misfiled as failure"
    if _looks_like_successful_turn_progress(text, lower):
        return CapsuleKind.PROJECT, "successful turn progress update misfiled as failure"
    if text.startswith("Verification:") and _has_success_marker(lower) and not _has_concrete_failure_evidence(lower):
        return CapsuleKind.PROJECT, "successful verification evidence misfiled as failure"
    project_prefixes = (
        "Added ",
        "After capture, ",
        "Applied ",
        "Changed ",
        "Completed verification",
        "Committed ",
        "Continue Ara Memory OS",
        "Continue building Ara Memory OS",
        "Fixed ",
        "Implemented ",
        "Installed ",
        "Narrowed ",
        "Wired ",
    )
    if text.startswith(project_prefixes) and not _has_concrete_failure_evidence(lower):
        return CapsuleKind.PROJECT, "operational progress update misfiled as failure"
    if lower.startswith("git diff for ") or lower.startswith("git status for ") or lower.startswith("git diff stat for "):
        return CapsuleKind.PROJECT, "worktree evidence misfiled as failure"
    return None, ""


def _looks_like_successful_turn_progress(text: str, lower: str) -> bool:
    progress_prefixes = (
        "turn episode: assistant outcome: add ",
        "turn episode: assistant outcome: added ",
        "turn episode: assistant outcome: changed ",
        "turn episode: assistant outcome: completed ",
        "turn episode: assistant outcome: fixed ",
        "turn episode: assistant outcome: implemented ",
        "turn episode: assistant outcome: wired ",
    )
    if not lower.startswith(progress_prefixes):
        return False
    if _has_concrete_failure_evidence(lower):
        return False
    if "command outcomes:" in lower:
        return True
    return any(marker in lower for marker in (" passed", " pass,", " ok", " succeeded", " pushed "))


def _looks_like_successful_turn_episode(lower: str) -> bool:
    if "turn episode:" not in lower or "assistant outcome:" not in lower:
        return False
    outcome_actions = (
        "assistant outcome: add",
        "assistant outcome: added",
        "assistant outcome: change",
        "assistant outcome: changed",
        "assistant outcome: complete",
        "assistant outcome: completed",
        "assistant outcome: fix",
        "assistant outcome: fixed",
        "assistant outcome: implement",
        "assistant outcome: implemented",
        "assistant outcome: wire",
        "assistant outcome: wired",
    )
    if not any(action in lower for action in outcome_actions):
        return False
    if _has_concrete_failure_evidence(lower):
        return False
    success_markers = (
        "command outcomes:",
        "=> pass",
        "=> passed",
        "-> pass",
        "-> passed",
        " ok",
        " succeeded",
        " pushed",
        "status: pass",
        "changed=0",
        "fail=0",
        "failures=0",
    )
    return any(marker in lower for marker in success_markers)


def _has_concrete_failure_evidence(lower: str) -> bool:
    if re.search(r"\b(exit code|exit_code)\s*[:=]\s*[1-9]\d*\b", lower):
        return True
    if re.search(r"\b(traceback|exception|assertionerror)\b", lower):
        return True
    if re.search(r"\b(still fail|still fails|failed|failing|unresolved failure|not refreshed|not fixed)\b", lower):
        return True
    return False


def _has_success_marker(lower: str) -> bool:
    markers = (
        "=> pass",
        "=> passed",
        "-> pass",
        "-> passed",
        "status: pass",
        " ok",
        " passed",
        " succeeded",
        " pushed",
        "changed=0",
        "fail=0",
        "failures=0",
        "raw=0",
    )
    return any(marker in lower for marker in markers)


def _update_kind(store: MemoryStore, row: Any, kind: CapsuleKind, *, reason: str) -> None:
    import json

    now = utc_now()
    title = _retitled(row["title"], kind)
    tags = _tags_without_failure_noise(row["tags_json"])
    with store.session() as conn:
        conn.execute(
            "UPDATE capsules SET kind = ?, title = ?, tags_json = ?, updated_at = ? WHERE id = ?",
            (kind.value, title, json.dumps(tags, ensure_ascii=False), now, row["id"]),
        )
        conn.execute("DELETE FROM capsules_fts WHERE id = ?", (row["id"],))
        if row["status"] in {MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value}:
            conn.execute(
                "INSERT INTO capsules_fts(id, title, body, kind, scope, tags) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    title,
                    row["body"],
                    kind.value,
                    row["scope"],
                    " ".join(tags),
                ),
            )
        conn.execute(
            """
            INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("reclassify-kind", row["id"], row["scope"], reason, "failure-kind-audit", now),
        )


def _retitled(title: str, kind: CapsuleKind) -> str:
    prefix = "Failure memory: "
    if title.startswith(prefix):
        return f"{kind.value.title()} memory: {title[len(prefix):]}"
    return title


def _tags_without_failure_noise(raw: str) -> list[str]:
    import json

    try:
        tags = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [tag for tag in tags if tag != "failure"]


def _item(row: Any, to_kind: str, changed: bool, reason: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "status": row["status"],
        "from_kind": row["kind"],
        "to_kind": to_kind,
        "changed": changed,
        "reason": reason,
        "title": row["title"],
    }
