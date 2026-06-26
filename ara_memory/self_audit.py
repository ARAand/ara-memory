from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.models import CapsuleKind, MemoryStatus, utc_now
from ara_memory.storage import MemoryStore


@dataclass(slots=True)
class SelfKindAuditResult:
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
            f"# Ara Self Kind Audit: {self.scope or 'all'}",
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


def audit_self_kinds(
    store: MemoryStore,
    *,
    scope: str | None = None,
    statuses: list[str] | None = None,
    limit: int = 200,
    dry_run: bool = True,
) -> SelfKindAuditResult:
    store.init()
    status_values = statuses or [MemoryStatus.CANDIDATE.value, MemoryStatus.STABLE.value]
    rows = _self_rows(store, scope=scope, statuses=status_values, limit=limit)
    items: list[dict[str, Any]] = []
    changed = 0
    for row in rows:
        target, reason = _target_kind(row["body"])
        if target is None:
            items.append(_item(row, CapsuleKind.SELF.value, False, "valid Ara identity or judgment-principle memory"))
            continue
        if not dry_run:
            _update_kind(store, row, target, reason=reason)
        changed += 1
        items.append(_item(row, target.value, True, reason))
    return SelfKindAuditResult(
        scope=scope,
        dry_run=dry_run,
        reviewed=len(rows),
        changed=changed,
        items=items,
    )


def _self_rows(
    store: MemoryStore,
    *,
    scope: str | None,
    statuses: list[str],
    limit: int,
) -> list[Any]:
    clauses = ["kind = ?"]
    args: list[Any] = [CapsuleKind.SELF.value]
    if scope:
        clauses.append("(scope = ? OR scope = 'global')")
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
    lower = body.strip().lower()
    technical_identity = (
        "artifact identity",
        "file identity",
        "object identity",
        "identity hash",
        "identity key",
        "windows drive letters",
    )
    if any(pattern in lower for pattern in technical_identity):
        return CapsuleKind.DECISION, "technical identity text misfiled as Ara self memory"
    if lower.startswith(("git status for ", "git diff for ", "untracked file manifest:")):
        return CapsuleKind.PROJECT, "worktree evidence misfiled as Ara self memory"
    if lower.startswith(("command:", "exit code:")):
        return CapsuleKind.EPISODE, "command evidence misfiled as Ara self memory"
    return None, ""


def _update_kind(store: MemoryStore, row: Any, kind: CapsuleKind, *, reason: str) -> None:
    import json

    now = utc_now()
    title = _retitled(row["title"], kind)
    tags = _tags_without_self_noise(row["tags_json"])
    with store.session() as conn:
        conn.execute(
            "UPDATE capsules SET kind = ?, title = ?, tags_json = ?, updated_at = ? WHERE id = ?",
            (kind.value, title, json.dumps(tags, ensure_ascii=False), now, row["id"]),
        )
        conn.execute("DELETE FROM capsules_fts WHERE id = ?", (row["id"],))
        conn.execute(
            "INSERT INTO capsules_fts(id, title, body, kind, scope, tags) VALUES (?, ?, ?, ?, ?, ?)",
            (row["id"], title, row["body"], kind.value, row["scope"], " ".join(tags)),
        )
        conn.execute(
            """
            INSERT INTO memory_actions(action, capsule_id, scope, reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("reclassify-kind", row["id"], row["scope"], reason, "self-kind-audit", now),
        )


def _retitled(title: str, kind: CapsuleKind) -> str:
    prefix = "Self memory candidate: "
    if title.startswith(prefix):
        return f"{kind.value.title()} memory: {title[len(prefix):]}"
    return title


def _tags_without_self_noise(raw: str) -> list[str]:
    import json

    try:
        tags = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [tag for tag in tags if tag not in {"self", "ara"}]


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
