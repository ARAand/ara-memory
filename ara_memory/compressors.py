from __future__ import annotations

import re
from collections import Counter
from typing import Any


LATIN_KEYWORD_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")
HANGUL_KEYWORD_RE = re.compile(r"[\uac00-\ud7a3]{2,}")


def compact_text(text: str, *, limit: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    head = normalized[: int(limit * 0.65)].rstrip()
    tail = normalized[-int(limit * 0.25) :].lstrip()
    return f"{head} ... [compressed] ... {tail}"


def semantic_consolidation_text(items: list[dict[str, Any]], *, limit: int = 1400) -> str:
    """Build a compact, evidence-shaped memory from related capsules.

    This is deliberately model-free: it preserves durable cues, shared retrieval
    handles, and provenance without pasting a long raw transcript into a summary.
    """

    if not items:
        return ""
    remember = _memory_points(items, limit=6)
    use_when = _use_when_points(items)
    evidence = _evidence_points(items, limit=5)
    parts = ["Remember:"]
    parts.extend(f"- {point}" for point in remember) if remember else parts.append("- Related memories were consolidated.")
    parts.append("")
    parts.append("Use when:")
    parts.extend(f"- {point}" for point in use_when) if use_when else parts.append("- A query matches the shared scope, kind, or tags.")
    parts.append("")
    parts.append("Evidence:")
    parts.extend(f"- {point}" for point in evidence) if evidence else parts.append("- Source capsule provenance is preserved in source_event_ids.")
    text = "\n".join(parts).strip()
    return compact_text(text, limit=limit) if len(text) > limit else text


def title_from_text(text: str, *, fallback: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return fallback
    first_line = re.sub(r"\s+", " ", first_line)
    return first_line[:96]


def _memory_points(items: list[dict[str, Any]], *, limit: int) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()
    for item in items:
        for candidate in _candidate_points(item):
            normalized = _point_key(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            points.append(compact_text(candidate, limit=220))
            if len(points) >= limit:
                return points
    return points


def _candidate_points(item: dict[str, Any]) -> list[str]:
    title = str(item.get("title") or "").strip()
    body = str(item.get("body") or "").strip()
    kind = str(item.get("kind") or "memory")
    out: list[str] = []
    for text in (body, title):
        for clause in _split_clauses(text):
            point = _clean_memory_point(clause)
            if point:
                out.append(point)
    if not out and title:
        out.append(f"{kind} memory: {title}")
    return out


def _split_clauses(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\s+-\s+", text)
    return [piece.strip(" -*\t\r\n") for piece in pieces if piece.strip(" -*\t\r\n")]


def _clean_memory_point(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(Decision|Goal evidence|Goal memory|Procedure candidate|Failure memory|Project memory):\s*", "", cleaned)
    cleaned = re.sub(r"^Command:\s*", "Command outcome: ", cleaned)
    if not cleaned or cleaned in {"OK", "Exit code: 0"}:
        return ""
    if len(cleaned) < 8:
        return ""
    return cleaned


def _use_when_points(items: list[dict[str, Any]]) -> list[str]:
    scopes = _top_values(str(item.get("scope") or "") for item in items)
    kinds = _top_values(str(item.get("kind") or "") for item in items)
    tags = _top_values(
        str(tag)
        for item in items
        for tag in item.get("tags", [])
        if isinstance(tag, str) and not tag.startswith("artifact:")
    )
    out: list[str] = []
    if kinds:
        out.append("memory kind matches " + ", ".join(kinds[:3]))
    if scopes:
        out.append("scope matches " + ", ".join(scopes[:3]))
    if tags:
        out.append("query mentions " + ", ".join(tags[:6]))
    return out


def _evidence_points(items: list[dict[str, Any]], *, limit: int) -> list[str]:
    out: list[str] = []
    seen_sources: set[str] = set()
    for item in items:
        title = compact_text(str(item.get("title") or item.get("kind") or "memory"), limit=120)
        sources = [str(source) for source in item.get("source_event_ids", []) if str(source)]
        fresh = [source for source in sources if source not in seen_sources]
        for source in fresh:
            seen_sources.add(source)
        source_text = ", ".join(fresh[:3]) if fresh else "source retained"
        out.append(f"{source_text}: {title}")
        if len(out) >= limit:
            break
    return out


def _top_values(values: Any, *, limit: int = 8) -> list[str]:
    counts = Counter(value for value in values if value)
    return [value for value, _ in counts.most_common(limit)]


def _point_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:120]


def extract_keywords(text: str, *, limit: int = 12) -> list[str]:
    lowered = text.lower()
    words = [w.strip(".,;:()[]{}<>") for w in LATIN_KEYWORD_RE.findall(lowered)]
    words.extend(HANGUL_KEYWORD_RE.findall(lowered))
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "어떻게",
        "그리고",
        "하지만",
        "것은",
        "있는",
        "해야",
    }
    counts = Counter(w for w in words if w not in stop and not w.isdigit())
    return [word for word, _ in counts.most_common(limit)]


def estimate_tokens(text: str) -> int:
    # Conservative mixed Korean/English estimate.
    return max(1, int(len(text) / 2.7))


def trim_to_token_budget(parts: list[str], budget: int) -> str:
    if budget <= 0:
        return ""
    out: list[str] = []
    used = 0
    for part in parts:
        cost = estimate_tokens(part)
        if used + cost <= budget:
            out.append(part)
            used += cost
            continue
        remaining_chars = max(0, int((budget - used) * 2.7))
        if remaining_chars > 120:
            out.append(compact_text(part, limit=remaining_chars))
        break
    return "\n\n".join(out).strip()
