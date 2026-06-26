from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ara_memory.compressors import compact_text, estimate_tokens
from ara_memory.storage import MemoryStore, row_to_capsule


SAFE_SCOPE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class HotState:
    scope: str
    path: Path
    text: str
    estimated_tokens: int


class HotStateBuilder:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def build(self, *, scope: str = "global", budget: int = 1200) -> HotState:
        self.store.init()
        parts = [
            "# Ara Hot Memory",
            f"Scope: {scope}",
            "## Stable Identity / Preferences\n"
            + self._section(scope=scope, kinds={"self", "preference", "fact"}, limit=6),
            "## Current Project State\n" + self._section(scope=scope, kinds={"summary", "project"}, limit=8),
            "## Procedures And Warnings\n"
            + self._section(scope=scope, kinds={"procedure", "failure", "conflict"}, limit=8),
            "## Recent Decisions\n" + self._section(scope=scope, kinds={"decision"}, limit=6),
        ]
        text = _enforce_hot_budget(parts, budget)
        path = self._path(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        return HotState(scope=scope, path=path, text=text, estimated_tokens=estimate_tokens(text))

    def read(self, *, scope: str = "global") -> HotState | None:
        path = self._path(scope)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return HotState(scope=scope, path=path, text=text, estimated_tokens=estimate_tokens(text))

    def _section(self, *, scope: str, kinds: set[str], limit: int) -> str:
        rows = []
        for kind in sorted(kinds):
            rows.extend(self.store.list_capsules(scope=scope, status=None, kind=kind, limit=limit))
            if scope != "global":
                rows.extend(self.store.list_capsules(scope="global", status=None, kind=kind, limit=limit))
        capsules = [row_to_capsule(row) for row in rows if row["status"] == "stable" and row["kind"] in kinds]
        capsules.sort(
            key=lambda cap: (
                _hot_kind_priority(cap["kind"]),
                cap["salience"],
                cap["confidence"],
                cap["updated_at"],
            ),
            reverse=True,
        )
        if not capsules:
            return "- None."
        lines = []
        seen = set()
        for cap in capsules:
            body = compact_text(cap["body"], limit=260)
            dedupe_key = (cap["kind"], cap["title"].lower(), body.lower())
            if cap["id"] in seen or dedupe_key in seen or body.lower() in seen:
                continue
            seen.add(cap["id"])
            seen.add(dedupe_key)
            seen.add(body.lower())
            lines.append(
                f"- [{cap['kind']}] {cap['title']} "
                f"(confidence {cap['confidence']:.2f}, salience {cap['salience']:.2f})\n"
                f"  {body}"
            )
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def _path(self, scope: str) -> Path:
        safe = SAFE_SCOPE_RE.sub("_", scope).strip("._") or "global"
        return self.store.hot_dir / f"{safe}.md"


def _hot_kind_priority(kind: str) -> int:
    if kind == "summary":
        return 5
    if kind in {"decision", "procedure", "failure", "conflict"}:
        return 4
    if kind in {"self", "preference", "fact"}:
        return 3
    if kind == "project":
        return 2
    return 1


def _enforce_hot_budget(parts: list[str], budget: int) -> str:
    if budget <= 0:
        return ""
    text = "\n\n".join(parts).strip()
    if estimate_tokens(text) <= budget:
        return text

    trimmed = [_trim_hot_part(part, _initial_hot_limit(part, budget, len(parts))) for part in parts]
    text = "\n\n".join(trimmed).strip()
    while estimate_tokens(text) > budget:
        index = _longest_hot_section(trimmed)
        if index is None:
            break
        next_part = _shrink_hot_part(trimmed[index])
        if next_part == trimmed[index]:
            break
        trimmed[index] = next_part
        text = "\n\n".join(trimmed).strip()
    return text


def _initial_hot_limit(part: str, budget: int, part_count: int) -> int:
    base = max(160, int((budget * 2.35) / max(1, part_count)))
    if part.startswith("# Ara Hot Memory") or part.startswith("Scope:"):
        return max(base, len(part))
    if part.startswith("## Current Project State"):
        return max(base, int(budget * 0.9))
    return base


def _trim_hot_part(part: str, char_limit: int) -> str:
    if len(part) <= char_limit:
        return part
    if "\n" not in part:
        return compact_text(part, limit=max(80, char_limit))
    header, body = part.split("\n", 1)
    return f"{header}\n{compact_text(body, limit=max(80, char_limit - len(header) - 1))}"


def _longest_hot_section(parts: list[str]) -> int | None:
    candidates = [
        (len(part), index)
        for index, part in enumerate(parts)
        if "\n" in part and not part.startswith("# Ara Hot Memory")
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def _shrink_hot_part(part: str) -> str:
    header, body = part.split("\n", 1)
    if not body.strip():
        return part
    return f"{header}\n{compact_text(body, limit=max(80, int(len(body) * 0.78)))}"
