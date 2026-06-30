from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ara_memory.compressors import compact_text, estimate_tokens
from ara_memory.memory_lifecycle import is_core_anchor
from ara_memory.models import MemoryStatus
from ara_memory.risk import MemoryRiskAssessor
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
            + self._section(scope=scope, kinds={"self", "preference"}, limit=6),
            "## Active Goals\n" + self._section(scope=scope, kinds={"goal"}, limit=5),
            "## Working Memory Boundary\n"
            "- Stable decisions, procedures, summaries, project state, failures, conflicts, facts, and episodes "
            "stay query-selected instead of always-on.",
        ]
        text = _enforce_hot_budget(parts, budget)
        path = self._path(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(text + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return HotState(scope=scope, path=path, text=text, estimated_tokens=estimate_tokens(text))

    def read(self, *, scope: str = "global") -> HotState | None:
        path = self._path(scope)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return HotState(scope=scope, path=path, text=text, estimated_tokens=estimate_tokens(text))

    def _section(self, *, scope: str, kinds: set[str], limit: int) -> str:
        rows = []
        fetch_limit = max(limit * 4, limit + 12)
        for kind in sorted(kinds):
            rows.extend(self.store.list_capsules(scope=scope, status=MemoryStatus.STABLE, kind=kind, limit=fetch_limit))
            if scope != "global":
                rows.extend(
                    self.store.list_capsules(scope="global", status=MemoryStatus.STABLE, kind=kind, limit=fetch_limit)
                )
        risk = MemoryRiskAssessor(self.store)
        capsules = [
            cap
            for cap in (row_to_capsule(row) for row in rows if row["status"] == "stable" and row["kind"] in kinds)
            if is_core_anchor(cap) and not risk.assess_capsule(cap).should_exclude_from_hot
        ]
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
    if kind == "goal":
        return 5
    if kind in {"self", "preference"}:
        return 4
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
