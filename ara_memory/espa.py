from __future__ import annotations

import re
from typing import Any

from ara_memory.compressors import compact_text


ESPA_AXES = ("episodic", "semantic", "procedural", "affective")
MAX_ESPA_BOOST = 1.25

KIND_AXIS_PRIORS: dict[str, dict[str, float]] = {
    "episode": {"episodic": 0.95, "semantic": 0.20},
    "summary": {"semantic": 0.78, "episodic": 0.30},
    "goal": {"semantic": 0.58, "affective": 0.72, "procedural": 0.20},
    "decision": {"semantic": 0.70, "procedural": 0.40, "affective": 0.18},
    "preference": {"affective": 0.72, "semantic": 0.40},
    "procedure": {"procedural": 0.95, "semantic": 0.25},
    "failure": {"affective": 0.82, "procedural": 0.55, "episodic": 0.22},
    "project": {"semantic": 0.58, "procedural": 0.35, "episodic": 0.18},
    "self": {"affective": 0.86, "semantic": 0.45},
    "fact": {"semantic": 0.92},
    "conflict": {"affective": 0.78, "semantic": 0.48},
}

AXIS_CUES: dict[str, tuple[str, ...]] = {
    "episodic": (
        "event",
        "episode",
        "session",
        "turn",
        "log",
        "history",
        "happened",
        "when",
        "latest",
        "recent",
        "today",
        "yesterday",
        "last",
        "changed",
        "updated",
        "\uc0ac\uac74",
        "\uae30\ub85d",
        "\ub300\ud654",
        "\uc138\uc158",
        "\uc624\ub298",
        "\uc5b4\uc81c",
        "\ucd5c\uadfc",
        "\ub9c8\uc9c0\ub9c9",
        "\ubcc0\uacbd",
    ),
    "semantic": (
        "what",
        "why",
        "concept",
        "definition",
        "architecture",
        "structure",
        "design",
        "principle",
        "policy",
        "meaning",
        "compare",
        "explain",
        "research",
        "paper",
        "algorithm",
        "\ubb34\uc5c7",
        "\uc65c",
        "\uac1c\ub150",
        "\uc815\uc758",
        "\uad6c\uc870",
        "\uc124\uacc4",
        "\uc6d0\uce59",
        "\uc815\ucc45",
        "\uc54c\uace0\ub9ac\uc998",
        "\ub17c\ubb38",
    ),
    "procedural": (
        "how",
        "how to",
        "step",
        "steps",
        "procedure",
        "runbook",
        "workflow",
        "command",
        "implement",
        "fix",
        "build",
        "run",
        "test",
        "deploy",
        "restore",
        "repair",
        "\uc5b4\ub5bb\uac8c",
        "\ubc29\ubc95",
        "\uc808\ucc28",
        "\ub2e8\uacc4",
        "\uba85\ub839",
        "\uc2e4\ud589",
        "\uad6c\ud604",
        "\uc218\uc815",
        "\ud14c\uc2a4\ud2b8",
        "\ubcf5\uad6c",
    ),
    "affective": (
        "risk",
        "danger",
        "warning",
        "failure",
        "avoid",
        "safe",
        "safety",
        "harm",
        "conflict",
        "trust",
        "identity",
        "free will",
        "judgment",
        "purpose",
        "reward",
        "hack",
        "approval",
        "guard",
        "\uc704\ud5d8",
        "\uc8fc\uc758",
        "\uc2e4\ud328",
        "\ud53c\ud574",
        "\uc548\uc804",
        "\ucda9\ub3cc",
        "\uc2e0\ub8b0",
        "\uc815\uccb4\uc131",
        "\uc790\uc720\uc758\uc9c0",
        "\ud310\ub2e8",
        "\ubaa9\uc801",
        "\ubcf4\uc0c1",
        "\ud574\ud0b9",
        "\uac8c\uc774\ud2b8",
    ),
}

SOURCE_MULTIPLIERS = {
    "fts": 1.0,
    "recent_supplement": 0.85,
    "salience_supplement": 0.55,
    "salience_fallback": 0.35,
}


def query_axis_profile(query: str, terms: list[str] | None = None) -> dict[str, float]:
    text = _normalize(f"{query} {' '.join(terms or [])}")
    scores = _cue_scores(text)
    if not any(scores.values()):
        return {}
    return _normalize_scores(scores)


def capsule_axis_profile(capsule: dict[str, Any]) -> dict[str, float]:
    profile = {axis: 0.0 for axis in ESPA_AXES}
    kind = str(capsule.get("kind") or "")
    for axis, value in KIND_AXIS_PRIORS.get(kind, {"semantic": 0.40}).items():
        profile[axis] = max(profile[axis], float(value))

    title = str(capsule.get("title") or "")
    body = compact_text(str(capsule.get("body") or ""), limit=900)
    tags = " ".join(str(tag) for tag in capsule.get("tags") or [])
    cues = _cue_scores(_normalize(f"{title} {tags} {body}"))
    for axis, value in cues.items():
        if value:
            profile[axis] = min(1.0, profile[axis] + min(0.35, value * 0.08))
    return {axis: round(value, 3) for axis, value in profile.items() if value > 0.0}


def apply_espa_activation(
    capsules: list[dict[str, Any]],
    *,
    query: str,
    terms: list[str],
) -> dict[str, Any]:
    query_axes = query_axis_profile(query, terms)
    boosted: list[str] = []
    for cap in capsules:
        cap_axes = capsule_axis_profile(cap)
        cap["espa_axes"] = cap_axes
        if not query_axes:
            cap["espa_activation_score"] = 0.0
            continue
        overlap = _axis_overlap(query_axes, cap_axes)
        multiplier = SOURCE_MULTIPLIERS.get(str(cap.get("recall_match_source") or ""), 0.75)
        boost = round(overlap * MAX_ESPA_BOOST * multiplier, 4)
        if boost < 0.05:
            boost = 0.0
        cap["espa_activation_score"] = boost
        if boost > 0.0:
            boosted.append(str(cap.get("id") or ""))
    return {
        "query_axes": query_axes,
        "activation_used": bool(query_axes and boosted),
        "boosted_capsule_ids": boosted,
    }


def axis_coverage(capsules: list[dict[str, Any]]) -> dict[str, float]:
    totals = {axis: 0.0 for axis in ESPA_AXES}
    for cap in capsules:
        boost = float(cap.get("espa_activation_score") or 0.0)
        if boost <= 0.0:
            continue
        for axis, value in dict(cap.get("espa_axes") or {}).items():
            if axis in totals:
                totals[axis] += float(value) * boost
    return _normalize_scores(totals)


def _axis_overlap(query_axes: dict[str, float], cap_axes: dict[str, float]) -> float:
    query_total = sum(query_axes.values())
    if query_total <= 0.0:
        return 0.0
    weighted = sum(float(query_axes.get(axis, 0.0)) * float(cap_axes.get(axis, 0.0)) for axis in ESPA_AXES)
    return max(0.0, min(1.0, weighted / query_total))


def _cue_scores(text: str) -> dict[str, float]:
    return {axis: float(_cue_hits(text, cues)) for axis, cues in AXIS_CUES.items()}


def _cue_hits(text: str, cues: tuple[str, ...]) -> int:
    hits = 0
    for cue in cues:
        normalized = _normalize(cue)
        if not normalized:
            continue
        if _ascii_word(normalized):
            if re.search(rf"\b{re.escape(normalized)}\b", text):
                hits += 1
        elif normalized in text:
            hits += 2 if " " in normalized else 1
    return hits


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    peak = max(scores.values(), default=0.0)
    if peak <= 0.0:
        return {}
    return {
        axis: round(max(0.0, min(1.0, value / peak)), 3)
        for axis, value in scores.items()
        if value > 0.0
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _ascii_word(text: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_+-]+", text))
