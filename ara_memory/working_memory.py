from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text, estimate_tokens, extract_keywords, trim_to_token_budget
from ara_memory.models import Event


@dataclass(slots=True)
class CueFrame:
    scope: str
    prompt: str
    active_files: list[str]
    command_errors: list[str]
    constraints: list[str]
    temporal_hints: list[str]

    @property
    def text(self) -> str:
        parts = [f"Prompt: {self.prompt}"] if self.prompt else []
        if self.active_files:
            parts.append("Active files: " + ", ".join(self.active_files[:8]))
        if self.command_errors:
            parts.append("Command errors: " + " | ".join(compact_text(item, limit=180) for item in self.command_errors[:5]))
        if self.constraints:
            parts.append("Constraints: " + " | ".join(self.constraints[:8]))
        if self.temporal_hints:
            parts.append("Temporal hints: " + ", ".join(self.temporal_hints[:8]))
        return "\n".join(parts).strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "prompt": self.prompt,
            "active_files": self.active_files,
            "command_errors": self.command_errors,
            "constraints": self.constraints,
            "temporal_hints": self.temporal_hints,
        }


@dataclass(slots=True)
class WorkingMemoryItem:
    section: str
    capsule_id: str
    kind: str
    status: str
    title: str
    text: str
    reason: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "capsule_id": self.capsule_id,
            "kind": self.kind,
            "status": self.status,
            "title": self.title,
            "text": self.text,
            "reason": self.reason,
            "score": round(self.score, 4),
        }


@dataclass(slots=True)
class WorkingMemoryReport:
    cue: CueFrame
    budget: int
    recall_query: str
    items: list[WorkingMemoryItem]
    recall_diagnostics: dict[str, Any]
    diagnostics: dict[str, Any]

    @property
    def influential_capsule_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in sorted(self.items, key=lambda value: value.score, reverse=True):
            if item.capsule_id in seen:
                continue
            seen.add(item.capsule_id)
            out.append(item.capsule_id)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "cue": self.cue.as_dict(),
            "budget": self.budget,
            "recall_query": self.recall_query,
            "items": [item.as_dict() for item in self.items],
            "influential_capsule_ids": self.influential_capsule_ids,
            "recall_diagnostics": self.recall_diagnostics,
            "diagnostics": self.diagnostics,
        }

    def to_text(self) -> str:
        return _render_working_memory_text(self.cue, self.items, self.budget)


def _render_working_memory_text(
    cue: CueFrame,
    items: list[WorkingMemoryItem],
    budget: int,
    *,
    trim: bool = True,
) -> str:
    header = [
        "# Ara Associative Working Memory",
        f"Scope: {cue.scope}",
        f"Cue: {compact_text(cue.text, limit=420)}",
    ]
    if not items:
        text = "\n".join(
            [
                *header,
                "## Keep In Mind\n- No direct associative memory matched this cue.",
                "## Risk / Friction\n- No prior hazard was retrieved for this cue.",
                "## This Should Change My Next Action\n- Proceed from current evidence; do not invent remembered context.",
            ]
        )
        return trim_to_token_budget([text], budget) if trim else text
    parts = [*header]
    for section in ("keep", "risk", "action"):
        title = {
            "keep": "## Keep In Mind",
            "risk": "## Risk / Friction",
            "action": "## This Should Change My Next Action",
        }[section]
        section_items = [item for item in items if item.section == section]
        parts.append(title)
        if not section_items:
            parts.append("- None.")
            continue
        for item in section_items[:5]:
            parts.append(
                f"- [{item.kind}/{item.status}] {compact_text(item.text, limit=260)} "
                f"(source {item.capsule_id}; {item.reason})"
            )
    text = "\n".join(parts)
    return trim_to_token_budget([text], budget) if trim else text


def build_working_memory(
    memory: Any,
    *,
    prompt: str,
    scope: str = "global",
    active_files: list[str] | None = None,
    command_errors: list[str] | None = None,
    budget: int = 900,
    recall_budget: int = 1600,
    include_global: bool = True,
    include_hot: bool = True,
) -> WorkingMemoryReport:
    cue = CueFrame(
        scope=scope,
        prompt=prompt.strip(),
        active_files=[_file_cue(item) for item in active_files or [] if str(item).strip()],
        command_errors=[str(item).strip() for item in command_errors or [] if str(item).strip()],
        constraints=_extract_constraints(prompt),
        temporal_hints=_extract_temporal_hints(prompt),
    )
    recall_query = _recall_query(cue)
    recall = memory.recall_candidates(
        recall_query,
        scope=scope,
        budget=recall_budget,
        include_global=include_global,
        include_hot=include_hot,
    )
    capsules = list(recall.renderable_capsules)
    raw_items = _build_items(capsules, cue)
    items = _fit_items_to_budget(raw_items, cue, budget)
    projected_capsule_ids = _projected_capsule_ids(items)
    diagnostics = {
        "cue_terms": extract_keywords(cue.text, limit=16),
        "capsules_considered": int(recall.diagnostics.get("capsules_considered", len(capsules))),
        "capsules_selected": int(recall.diagnostics.get("capsules_selected", len(capsules))),
        "capsules_projected": len(projected_capsule_ids),
        "raw_items": len(raw_items),
        "items": len(items),
        "sections": {
            "keep": sum(1 for item in items if item.section == "keep"),
            "risk": sum(1 for item in items if item.section == "risk"),
            "action": sum(1 for item in items if item.section == "action"),
        },
        "projected_capsule_ids": projected_capsule_ids,
        "estimated_tokens": estimate_tokens("\n".join(item.text for item in items)) if items else 0,
        "low_evidence_fallback_suppressed": bool(recall.diagnostics.get("low_evidence_fallback_suppressed", False)),
        "candidate_only_recall": True,
    }
    return WorkingMemoryReport(
        cue=cue,
        budget=budget,
        recall_query=recall_query,
        items=items,
        recall_diagnostics=recall.diagnostics,
        diagnostics=diagnostics,
    )


def record_memory_impact(
    memory: Any,
    *,
    scope: str,
    cue: str,
    capsule_ids: list[str],
    outcome: str,
    helped: bool | None = None,
    source: str = "working-memory-impact",
) -> Event:
    payload = {
        "cue": cue,
        "capsule_ids": list(dict.fromkeys(capsule_ids)),
        "outcome": outcome,
        "helped": helped,
    }
    text = (
        "Working memory impact:\n"
        f"Cue: {compact_text(cue, limit=300)}\n"
        f"Capsules: {', '.join(payload['capsule_ids']) or 'none'}\n"
        f"Outcome: {compact_text(outcome, limit=500)}\n"
        f"Helped: {helped if helped is not None else 'unknown'}"
    )
    return memory.retain(
        kind="note",
        text=text,
        source=source,
        scope=scope,
        metadata={"working_memory_impact": payload},
    )


def _build_items(capsules: list[dict[str, Any]], cue: CueFrame) -> list[WorkingMemoryItem]:
    cue_terms = extract_keywords(cue.text, limit=16)
    items: list[WorkingMemoryItem] = []
    for cap in capsules:
        score = _score_capsule(cap, cue_terms)
        keep = _keep_item(cap, score)
        if keep:
            items.append(keep)
        risk = _risk_item(cap, score)
        if risk:
            items.append(risk)
        action = _action_item(cap, score)
        if action:
            items.append(action)
    items.sort(key=lambda item: (_section_priority(item.section), item.score), reverse=True)
    return _dedupe_items(items)


def _fit_items_to_budget(items: list[WorkingMemoryItem], cue: CueFrame, budget: int) -> list[WorkingMemoryItem]:
    if budget <= 0:
        return []
    kept: list[WorkingMemoryItem] = []
    for item in items:
        trial = kept + [item]
        if estimate_tokens(_render_working_memory_text(cue, trial, budget, trim=False)) <= budget:
            kept.append(item)
    return kept


def _projected_capsule_ids(items: list[WorkingMemoryItem]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item.capsule_id in seen:
            continue
        seen.add(item.capsule_id)
        out.append(item.capsule_id)
    return out


def _keep_item(cap: dict[str, Any], score: float) -> WorkingMemoryItem | None:
    if cap["kind"] not in {"goal", "self", "preference", "decision", "summary", "fact", "project"}:
        return None
    return _item(
        "keep",
        cap,
        _memory_text(cap),
        reason=_reason(cap, "relevant orientation"),
        score=score,
    )


def _risk_item(cap: dict[str, Any], score: float) -> WorkingMemoryItem | None:
    text = f"{cap['title']} {cap['body']}".lower()
    if cap["kind"] not in {"failure", "conflict"} and not re.search(r"\b(risk|avoid|blocked|failed|error|warning|do not)\b", text):
        return None
    return _item(
        "risk",
        cap,
        _memory_text(cap),
        reason=_reason(cap, "prior hazard"),
        score=score + 0.35,
    )


def _action_item(cap: dict[str, Any], score: float) -> WorkingMemoryItem | None:
    kind = cap["kind"]
    stable_templates = {
        "goal": "Keep this goal visible while choosing the next step: {text}",
        "self": "Let this identity principle constrain the next action: {text}",
        "preference": "Respect this user preference in the next action: {text}",
        "decision": "Treat this as already decided unless new evidence contradicts it: {text}",
        "procedure": "Prefer this known workflow before inventing a new one: {text}",
        "failure": "Check this prior failure mode before acting: {text}",
        "conflict": "Resolve or avoid this conflict before relying on the memory: {text}",
        "summary": "Use this consolidated context to orient the next action: {text}",
        "project": "Use this project context to choose the next file or command: {text}",
    }
    candidate_templates = {
        "goal": "Treat this goal as candidate evidence and verify it before steering the turn: {text}",
        "self": "Treat this identity note as candidate evidence, not a binding principle: {text}",
        "preference": "Consider this preference candidate, but verify it before relying on it: {text}",
        "decision": "Review this candidate decision before treating it as settled: {text}",
        "procedure": "Consider this candidate workflow, but prefer stable procedure or current evidence first: {text}",
        "failure": "Check whether this candidate failure mode is relevant before acting: {text}",
        "conflict": "Investigate this candidate conflict before relying on the memory: {text}",
        "summary": "Use this candidate context only as a weak orientation signal: {text}",
        "project": "Use this candidate project context only after checking current files or commands: {text}",
    }
    templates = stable_templates if cap.get("status") == "stable" else candidate_templates
    template = templates.get(kind)
    if not template:
        return None
    return _item(
        "action",
        cap,
        template.format(text=_memory_text(cap)),
        reason=_reason(cap, "behavioral influence"),
        score=score + 0.2,
    )


def _item(section: str, cap: dict[str, Any], text: str, *, reason: str, score: float) -> WorkingMemoryItem:
    return WorkingMemoryItem(
        section=section,
        capsule_id=str(cap["id"]),
        kind=str(cap["kind"]),
        status=str(cap.get("status") or "unknown"),
        title=str(cap["title"]),
        text=compact_text(text, limit=360),
        reason=reason,
        score=score,
    )


def _memory_text(cap: dict[str, Any]) -> str:
    body = str(cap.get("body") or "")
    title = str(cap.get("title") or "")
    if body and body.lower() not in title.lower():
        return f"{title}: {body}"
    return body or title


def _score_capsule(cap: dict[str, Any], cue_terms: list[str]) -> float:
    haystack = f"{cap.get('title', '')} {cap.get('body', '')} {' '.join(cap.get('tags', []))}".lower()
    hits = sum(1 for term in cue_terms if term.lower() in haystack)
    return (
        _kind_priority(str(cap.get("kind", "")))
        + float(cap.get("salience") or 0.0)
        + float(cap.get("confidence") or 0.0) * 0.5
        + min(1.5, hits * 0.25)
    )


def _reason(cap: dict[str, Any], fallback: str) -> str:
    source = str(cap.get("recall_match_source") or "")
    if source == "fts":
        return "direct lexical evidence"
    if source == "recent_supplement":
        return "recent related memory"
    if source == "salience_supplement":
        return "salient supporting memory"
    return fallback


def _dedupe_items(items: list[WorkingMemoryItem]) -> list[WorkingMemoryItem]:
    seen: set[tuple[str, str, str]] = set()
    out: list[WorkingMemoryItem] = []
    for item in items:
        key = (item.section, item.capsule_id, re.sub(r"\W+", " ", item.text.lower())[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _section_priority(section: str) -> int:
    return {"action": 3, "risk": 2, "keep": 1}.get(section, 0)


def _kind_priority(kind: str) -> float:
    return {
        "goal": 2.0,
        "self": 1.8,
        "preference": 1.7,
        "failure": 1.6,
        "conflict": 1.5,
        "procedure": 1.4,
        "decision": 1.3,
        "summary": 1.1,
        "project": 1.0,
        "fact": 0.9,
    }.get(kind, 0.5)


def _recall_query(cue: CueFrame) -> str:
    terms = extract_keywords(cue.text, limit=18)
    return " ".join(terms) if terms else cue.prompt


def _extract_constraints(prompt: str) -> list[str]:
    constraints = []
    for line in prompt.splitlines():
        lowered = line.lower()
        markers = (
            "must",
            "should",
            "do not",
            "don't",
            "avoid",
            "never",
            "\ubc18\ub4dc\uc2dc",
            "\ud558\uc9c0\ub9c8",
            "\ud53c\ud574",
        )
        if any(marker in lowered for marker in markers):
            constraints.append(compact_text(line.strip(), limit=220))
    return constraints


def _extract_temporal_hints(prompt: str) -> list[str]:
    hints = []
    lowered = prompt.lower()
    markers = (
        "latest",
        "recent",
        "today",
        "current",
        "now",
        "last",
        "\ucd5c\uc2e0",
        "\ucd5c\uadfc",
        "\uc624\ub298",
        "\ud604\uc7ac",
        "\uc9c0\uae08",
    )
    for marker in markers:
        if marker in lowered:
            hints.append(marker)
    return hints


def _file_cue(value: Any) -> str:
    path = Path(str(value))
    name = path.name or str(path)
    suffix = path.suffix.lower()
    return f"{name}{' ' + suffix if suffix else ''}".strip()
