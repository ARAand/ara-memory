from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ara_memory.privacy import HIDDEN_REASONING_LINE
from ara_memory.risk import direct_identifier_matches, sensitive_text_matches


PRIVATE_TOP_LEVEL_PATHS = {
    ".agents",
    ".ara-memory",
    ".codex",
    "archive",
    "backups",
    "hot",
    "ledger",
    "spool",
}
PRIVATE_FILE_NAMES = {
    ".archive-object-key",
    ".backup-signing-key",
    ".seal-key",
    "events.jsonl",
    "memory.db",
}
PRIVATE_FILE_SUFFIXES = (
    ".bak",
    ".db",
    ".log",
    ".sqlite",
    ".sqlite3",
    ".zip",
)
PRIVATE_MEMORY_MARKERS = (
    '"format": "ara-memory-backup',
    '"format": "ara-memory-spooled-turn-v1"',
    '"manifest_hmac_sha256"',
    '"spool_privacy"',
    '"spool_snapshot_path"',
)
FIXTURE_PATH_PREFIXES = ("tests/", "docs/")
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(slots=True)
class PrivacyPushFinding:
    path: str
    severity: str
    kind: str
    detail: str
    line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "severity": self.severity,
            "kind": self.kind,
            "detail": self.detail,
            "line": self.line,
        }


@dataclass(slots=True)
class PrivacyPushReport:
    repo: str
    status: str
    passed: bool
    scanned_files: int
    findings: list[PrivacyPushFinding]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "status": self.status,
            "passed": self.passed,
            "scanned_files": self.scanned_files,
            "findings": [finding.as_dict() for finding in self.findings],
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Privacy Pre-Push Gate: {self.repo}",
            f"status: {self.status}",
            f"scanned_files: {self.scanned_files}",
        ]
        if not self.findings:
            lines.append("## Findings\n- None.")
            return "\n".join(lines)
        lines.append("## Findings")
        for finding in self.findings:
            location = f"{finding.path}:{finding.line}" if finding.line else finding.path
            lines.append(f"- [{finding.severity}] {finding.kind} at {location}: {finding.detail}")
        return "\n".join(lines)


def run_privacy_pre_push_gate(
    *,
    repo: Path | str = ".",
    include_untracked: bool = False,
    include_direct_identifiers: bool = False,
    max_bytes: int = 1_000_000,
) -> PrivacyPushReport:
    repo_path = Path(repo).resolve()
    paths = _git_paths(repo_path, include_untracked=include_untracked)
    findings: list[PrivacyPushFinding] = []
    scanned = 0
    for path in paths:
        portable = _portable(path)
        path_findings = _path_findings(portable)
        findings.extend(path_findings)
        full_path = repo_path / path
        if path_findings and all(finding.severity == "fail" for finding in path_findings):
            continue
        if not _should_scan_text(full_path, portable, max_bytes=max_bytes):
            continue
        scanned += 1
        findings.extend(
            _content_findings(
                full_path,
                portable,
                include_direct_identifiers=include_direct_identifiers,
            )
        )
    findings.sort(key=lambda item: (0 if item.severity == "fail" else 1, item.path, item.line or 0, item.kind))
    status = "fail" if any(finding.severity == "fail" for finding in findings) else "watch" if findings else "pass"
    return PrivacyPushReport(
        repo=str(repo_path),
        status=status,
        passed=status != "fail",
        scanned_files=scanned,
        findings=findings,
    )


def _git_paths(repo: Path, *, include_untracked: bool) -> list[Path]:
    tracked = _git_path_output(repo, ["ls-files", "-z"])
    staged = _git_path_output(repo, ["diff", "--cached", "--name-only", "-z"])
    paths = [*tracked, *staged]
    if include_untracked:
        paths.extend(_git_path_output(repo, ["ls-files", "--others", "--exclude-standard", "-z"]))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        normalized = _portable(path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(path)
    return out


def _git_path_output(repo: Path, args: list[str]) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    if not result.stdout:
        return []
    return [Path(item.decode("utf-8", errors="replace")) for item in result.stdout.split(b"\0") if item]


def _path_findings(path: str) -> list[PrivacyPushFinding]:
    pure = PurePosixPath(path)
    parts = [part.lower() for part in pure.parts]
    name = pure.name.lower()
    findings: list[PrivacyPushFinding] = []
    if parts and parts[0] in PRIVATE_TOP_LEVEL_PATHS:
        findings.append(
            PrivacyPushFinding(
                path=path,
                severity="fail",
                kind="private-memory-path",
                detail="private memory evidence path is tracked or staged",
            )
        )
    if name in PRIVATE_FILE_NAMES or any(name.endswith(suffix) for suffix in PRIVATE_FILE_SUFFIXES):
        findings.append(
            PrivacyPushFinding(
                path=path,
                severity="fail",
                kind="private-memory-file",
                detail="private memory database, ledger, key, backup, or log file is tracked or staged",
            )
        )
    return findings


def _content_findings(
    path: Path,
    portable: str,
    *,
    include_direct_identifiers: bool,
) -> list[PrivacyPushFinding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[PrivacyPushFinding] = []
    for index, line in enumerate(text.splitlines(), start=1):
        marker_matches = [marker for marker in PRIVATE_MEMORY_MARKERS if marker in line]
        if marker_matches and _is_private_payload_candidate(portable) and not _is_fixture_path(portable):
            findings.append(
                PrivacyPushFinding(
                    path=portable,
                    severity="fail",
                    kind="private-memory-marker",
                    detail=", ".join(marker_matches[:3]),
                    line=index,
                )
            )
        hidden = HIDDEN_REASONING_LINE.search(line)
        if hidden and not _is_fixture_path(portable):
            findings.append(
                PrivacyPushFinding(
                    path=portable,
                    severity="fail",
                    kind="hidden-reasoning-text",
                    detail="hidden reasoning marker appears in tracked content",
                    line=index,
                )
            )
        sensitive = sensitive_text_matches(line)
        if sensitive:
            findings.append(
                PrivacyPushFinding(
                    path=portable,
                    severity="warning" if _is_fixture_path(portable) else "fail",
                    kind="secret-like-text",
                    detail=", ".join(sensitive[:4]),
                    line=index,
                )
            )
        if include_direct_identifiers:
            identifiers = direct_identifier_matches(line)
            if identifiers:
                findings.append(
                    PrivacyPushFinding(
                        path=portable,
                        severity="warning" if _is_fixture_path(portable) else "fail",
                        kind="direct-identifier",
                        detail=", ".join(identifiers[:4]),
                        line=index,
                    )
                )
    return findings


def _should_scan_text(path: Path, portable: str, *, max_bytes: int) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size > max_bytes:
            return False
    except OSError:
        return False
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    return _is_fixture_path(portable) or path.name.lower() in {"readme", "license"}


def _is_fixture_path(path: str) -> bool:
    return path.startswith(FIXTURE_PATH_PREFIXES)


def _is_private_payload_candidate(path: str) -> bool:
    pure = PurePosixPath(path)
    parts = [part.lower() for part in pure.parts]
    name = pure.name.lower()
    suffix = pure.suffix.lower()
    lower_path = path.lower()
    if parts and parts[0] in PRIVATE_TOP_LEVEL_PATHS:
        return True
    if name in PRIVATE_FILE_NAMES:
        return True
    if suffix in {".json", ".jsonl"} and any(
        marker in lower_path for marker in ("archive", "backup", "ledger", "memory", "spool")
    ):
        return True
    return False


def _portable(path: Path) -> str:
    value = path.as_posix()
    return value[2:] if value.startswith("./") else value
