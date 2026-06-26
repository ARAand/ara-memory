from __future__ import annotations

import re
from collections import Counter


KEYWORD_RE = re.compile(r"[A-Za-z가-힣0-9_./:-]{3,}")


def compact_text(text: str, *, limit: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    head = normalized[: int(limit * 0.65)].rstrip()
    tail = normalized[-int(limit * 0.25) :].lstrip()
    return f"{head} ... [compressed] ... {tail}"


def title_from_text(text: str, *, fallback: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return fallback
    first_line = re.sub(r"\s+", " ", first_line)
    return first_line[:96]


def extract_keywords(text: str, *, limit: int = 12) -> list[str]:
    lowered = text.lower()
    words = [w.strip(".,;:()[]{}<>") for w in KEYWORD_RE.findall(lowered)]
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
