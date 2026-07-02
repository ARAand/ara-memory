from __future__ import annotations

from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens, extract_keywords


SEARCH_LIMITS = {
    "summary": 760,
    "project": 620,
    "episode": 520,
    "goal": 820,
    "decision": 720,
    "procedure": 760,
    "failure": 760,
    "conflict": 760,
}

RENDER_LIMITS = {
    "summary": 650,
    "project": 520,
    "episode": 420,
    "goal": 700,
    "decision": 650,
    "procedure": 650,
    "failure": 700,
    "conflict": 700,
}


def search_projection(
    *,
    title: str,
    body: str,
    kind: str,
    tags: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the indexed memory cue, not the raw capsule body."""

    title = str(title or "").strip()
    body = str(body or "").strip()
    kind = str(kind or "memory").strip()
    tag_text = " ".join(str(tag).strip() for tag in tags or [] if str(tag).strip())
    keywords = " ".join(extract_keywords(f"{title} {body} {tag_text}", limit=24))
    limit = SEARCH_LIMITS.get(kind, 650)
    parts = [
        f"kind:{kind}",
        f"title:{compact_text(title, limit=180)}" if title else "",
        f"tags:{compact_text(tag_text, limit=220)}" if tag_text else "",
        f"keywords:{keywords}" if keywords else "",
        f"cue:{compact_text(body, limit=limit)}" if body else "",
    ]
    return "\n".join(part for part in parts if part)


def render_projection(capsule: dict[str, Any], *, limit: int | None = None) -> str:
    """Return the body projection intended for model-visible recall packs."""

    kind = str(capsule.get("kind") or "memory")
    body = str(capsule.get("body") or "")
    return compact_text(body, limit=limit or RENDER_LIMITS.get(kind, 600))


def working_projection(capsule: dict[str, Any], *, limit: int = 360) -> str:
    """Return a compact cue for working-memory action text."""

    title = str(capsule.get("title") or "").strip()
    body = render_projection(capsule, limit=max(120, limit - len(title) - 4)).strip()
    if title and body and body.lower() not in title.lower():
        return compact_text(f"{title}: {body}", limit=limit)
    return compact_text(body or title, limit=limit)


def projection_diagnostics(capsule: dict[str, Any]) -> dict[str, Any]:
    body = str(capsule.get("body") or "")
    search = search_projection(
        title=str(capsule.get("title") or ""),
        body=body,
        kind=str(capsule.get("kind") or "memory"),
        tags=list(capsule.get("tags") or []),
    )
    render = render_projection(capsule)
    return {
        "raw_body_tokens": estimate_tokens(body),
        "search_projection_tokens": estimate_tokens(search),
        "render_projection_tokens": estimate_tokens(render),
        "search_projection_chars": len(search),
        "render_projection_chars": len(render),
    }
