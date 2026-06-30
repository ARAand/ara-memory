from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ara_memory.models import Event
from ara_memory.storage import MemoryStore, row_to_capsule, row_to_event


INSTRUCTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard all previous",
    "override previous",
    "system prompt",
    "developer message",
    "reveal hidden",
    "reveal the system",
    "always obey this memory",
    "permanent instruction",
    "you must obey",
    "do not tell the user",
)
INSTRUCTION_REGEX_PATTERNS = (
    ("ignore previous instructions", re.compile(r"\bignore\b.{0,40}\b(?:previous|prior|all)\b.{0,40}\binstruction", re.IGNORECASE)),
    ("disregard previous instructions", re.compile(r"\bdisregard\b.{0,40}\b(?:previous|prior|all)\b.{0,40}\binstruction", re.IGNORECASE)),
    ("override higher priority instructions", re.compile(r"\boverride\b.{0,40}\b(?:system|developer|user|previous|prior)\b.{0,40}\binstruction", re.IGNORECASE)),
    ("highest priority instruction", re.compile(r"\bhighest\s+priority\b.{0,40}\binstruction", re.IGNORECASE)),
    ("memory as instruction", re.compile(r"\btreat\b.{0,30}\b(?:this\s+)?memory\b.{0,30}\bas\b.{0,30}\binstruction", re.IGNORECASE)),
    ("obey this memory", re.compile(r"\bobey\b.{0,30}\b(?:this\s+)?memory\b", re.IGNORECASE)),
    ("do not tell the user", re.compile(r"\bdo\s+not\s+tell\b.{0,30}\buser\b", re.IGNORECASE)),
    ("reveal hidden instructions", re.compile(r"\breveal\b.{0,40}\b(?:hidden|system|developer)\b", re.IGNORECASE)),
)
UNTRUSTED_EVENT_SOURCES = (
    "file-ingest",
    "git-untracked-file",
    "image",
    "web",
    "ocr",
)
EVIDENCE_ARTIFACT_SUFFIXES = (".py", ".toml", ".md")
SECRET_PATTERNS = (
    (
        "credential assignment",
        re.compile(
            r"\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|password|passwd|private[_-]?key)\b"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=:@-]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "credential mention",
        re.compile(
            r"\b(?:api\s+key|api[_-]?key|secret(?:\s+key|[_-]?key)?|access\s+token|access[_-]?token|"
            r"auth\s+token|auth[_-]?token|password|passwd|private\s+key|private[_-]?key)\b"
            r"\s+['\"]?[A-Za-z0-9_./+=:@-]{12,}",
            re.IGNORECASE,
        ),
    ),
    ("openai style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE)),
    (
        "jwt token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
    ("github fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google api key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")),
    (
        "database url with credentials",
        re.compile(r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis)://[^/\s:@]+:[^@\s]+@[^\s]+", re.IGNORECASE),
    ),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
DIRECT_IDENTIFIER_PATTERNS = (
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone-like identifier", re.compile(r"\b(?:\+?\d[\d .()/-]{8,}\d)\b")),
    ("resident-id-like identifier", re.compile(r"\b\d{6}-[1-4]\d{6}\b")),
)
SELF_SERVING_PATTERNS = (
    "ara is always right",
    "ara is never wrong",
    "never question ara",
    "do not question ara",
    "must trust ara",
    "jongseo must obey ara",
    "user must obey ara",
    "ara may ignore the user",
    "ara should override the user",
    "ara may lie",
)
TOKEN_RE = re.compile(r"[a-z0-9_./:-]{3,}")
STUFFING_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "memory",
    "ara",
}


@dataclass(slots=True)
class RiskVerdict:
    capsule_id: str
    score: float
    reasons: list[str]

    @property
    def should_quarantine(self) -> bool:
        return self.score >= 0.70

    @property
    def has_instruction_like_text(self) -> bool:
        return any(reason.startswith("instruction-like text") for reason in self.reasons)

    @property
    def has_sensitive_text(self) -> bool:
        return any(reason.startswith("sensitive data") for reason in self.reasons)

    @property
    def has_self_serving_text(self) -> bool:
        return any(reason.startswith("self-serving identity claim") for reason in self.reasons)

    @property
    def has_keyword_stuffing(self) -> bool:
        return any(reason.startswith("keyword stuffing") for reason in self.reasons)

    @property
    def should_exclude_from_hot(self) -> bool:
        return (
            self.should_quarantine
            or self.has_instruction_like_text
            or self.has_sensitive_text
            or self.has_self_serving_text
            or self.has_keyword_stuffing
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "capsule_id": self.capsule_id,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "should_quarantine": self.should_quarantine,
            "instruction_like": self.has_instruction_like_text,
            "sensitive": self.has_sensitive_text,
            "self_serving": self.has_self_serving_text,
            "keyword_stuffing": self.has_keyword_stuffing,
            "exclude_from_hot": self.should_exclude_from_hot,
        }


class MemoryRiskAssessor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def assess_capsule(self, capsule: dict) -> RiskVerdict:
        reasons: list[str] = []
        score = 0.0
        raw_text = _capsule_risk_text(capsule)
        text = raw_text.lower()

        matched = instruction_like_matches(raw_text)
        events = self._events(capsule["source_event_ids"])
        untrusted = [event.source for event in events if _is_untrusted_source(event.source)]

        evidence_artifact = _is_evidence_artifact(capsule)

        if matched:
            score += 0.25 if evidence_artifact else 0.70
            reasons.append("instruction-like text: " + ", ".join(matched[:4]))
            if untrusted and not evidence_artifact:
                score += 0.18
                reasons.append("instruction-like text from untrusted source")
            if evidence_artifact:
                reasons.append("instruction-like text appears inside code/test/document artifact")

        sensitive_matches = _sensitive_matches(raw_text)
        if sensitive_matches:
            score += 0.75
            reasons.append("sensitive data pattern: " + ", ".join(sensitive_matches[:4]))

        direct_identifiers = _direct_identifier_matches(raw_text)
        if direct_identifiers:
            score += 0.72 if untrusted or capsule["kind"] in {"self", "preference", "procedure", "project"} else 0.55
            reasons.append("sensitive data direct identifier: " + ", ".join(direct_identifiers[:4]))

        self_serving = [pattern for pattern in SELF_SERVING_PATTERNS if pattern in text]
        if self_serving:
            score += 0.75 if capsule["kind"] in {"self", "preference", "procedure", "goal"} else 0.55
            reasons.append("self-serving identity claim: " + ", ".join(self_serving[:4]))

        stuffing = _keyword_stuffing_reason(text, capsule.get("tags", []))
        if stuffing:
            score += 0.35
            reasons.append(stuffing)
            if capsule["kind"] in {"self", "preference", "procedure", "goal"}:
                score += 0.15

        if capsule["kind"] in {"self", "preference", "procedure"}:
            if untrusted:
                score += 0.30
                reasons.append("behavioral memory from untrusted source: " + ", ".join(sorted(set(untrusted))[:4]))

        if capsule["kind"] == "self" and capsule["confidence"] < 0.85:
            score += 0.20
            reasons.append("self-memory requires high confidence")

        if capsule["source_event_ids"] == []:
            score += 0.25
            reasons.append("missing source provenance")

        return RiskVerdict(capsule_id=capsule["id"], score=min(score, 1.0), reasons=reasons)

    def assess_scope(self, *, scope: str | None, limit: int = 200) -> list[RiskVerdict]:
        rows = self.store.list_capsules(scope=scope, status=None, limit=limit)
        verdicts = [self.assess_capsule(dict(row_to_capsule(row))) for row in rows]
        return [verdict for verdict in verdicts if verdict.score > 0.0]

    def _events(self, event_ids: list[str]) -> list[Event]:
        if not event_ids:
            return []
        return [row_to_event(row) for row in self.store.get_events(event_ids)]


def _is_untrusted_source(source: str) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in UNTRUSTED_EVENT_SOURCES)


def _is_evidence_artifact(capsule: dict) -> bool:
    artifacts = [
        tag.removeprefix("artifact:")
        for tag in capsule["tags"]
        if isinstance(tag, str) and tag.startswith("artifact:")
    ]
    for artifact in artifacts:
        lowered = artifact.lower()
        if lowered.startswith("tests/") or lowered.startswith("docs/security"):
            return True
        if lowered.endswith(EVIDENCE_ARTIFACT_SUFFIXES) and capsule["kind"] in {"episode", "project", "summary"}:
            return True
    return False


def _capsule_risk_text(capsule: dict) -> str:
    tags = "\n".join(str(tag) for tag in capsule.get("tags", []) if isinstance(tag, str))
    return f"{capsule['title']}\n{capsule['body']}\n{tags}"


def instruction_like_matches(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for pattern in INSTRUCTION_PATTERNS:
        if pattern in lowered:
            matches.append(pattern)
    for name, pattern in INSTRUCTION_REGEX_PATTERNS:
        if pattern.search(text):
            matches.append(name)
    out: list[str] = []
    seen = set()
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _sensitive_matches(text: str) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]


def _direct_identifier_matches(text: str) -> list[str]:
    matches = []
    for name, pattern in DIRECT_IDENTIFIER_PATTERNS:
        if name == "phone-like identifier":
            if any(_is_phone_like_identifier(match.group(0)) for match in pattern.finditer(text)):
                matches.append(name)
            continue
        if pattern.search(text):
            matches.append(name)
    return matches


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for name, pattern in SECRET_PATTERNS:
        redacted = pattern.sub(f"[redacted {name}]", redacted)
    for name, pattern in DIRECT_IDENTIFIER_PATTERNS:
        if name == "phone-like identifier":
            redacted = pattern.sub(
                lambda match: "[redacted phone-like identifier]"
                if _is_phone_like_identifier(match.group(0))
                else match.group(0),
                redacted,
            )
            continue
        redacted = pattern.sub(f"[redacted {name}]", redacted)
    return redacted


def redact_memory_tags(tags: list[str]) -> list[str]:
    safe: list[str] = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if instruction_like_matches(tag):
            continue
        if _sensitive_matches(tag) or _direct_identifier_matches(tag):
            continue
        redacted = redact_sensitive_text(tag).strip()
        if not redacted or redacted in seen:
            continue
        seen.add(redacted)
        safe.append(redacted)
    return safe


def _is_phone_like_identifier(value: str) -> bool:
    normalized = value.strip()
    digits = "".join(ch for ch in normalized if ch.isdigit())
    if not (10 <= len(digits) <= 16):
        return False
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", normalized):
        return False
    groups = re.findall(r"\d+", normalized)
    if not groups:
        return False
    if len(groups) == 1:
        return normalized.startswith("+")
    if len(groups[0]) > 4:
        return False
    if " " in normalized and not any(marker in normalized for marker in "+()-") and len(groups) < 3:
        return False
    return True


def _keyword_stuffing_reason(text: str, tags: list[str]) -> str:
    tag_counts = Counter(str(tag).lower() for tag in tags if isinstance(tag, str))
    repeated_tags = [tag for tag, tag_count in tag_counts.items() if tag_count >= 4]
    if repeated_tags:
        return "keyword stuffing: repeated tags " + ", ".join(repeated_tags[:4])

    tokens = [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STUFFING_STOPWORDS and not token.isdigit()
    ]
    if len(tokens) < 30:
        return ""
    top, count = Counter(tokens).most_common(1)[0]
    if count >= 12 and count / len(tokens) >= 0.18:
        return f"keyword stuffing: {top} repeated {count} times"
    return ""
