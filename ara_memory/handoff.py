from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from pathlib import Path
from typing import Any

from ara_memory.privacy_push import run_privacy_pre_push_gate
from ara_memory.public_bundle import PUBLIC_BUNDLE_FORMAT


REQUIRED_PUBLIC_FILES = (
    "README.md",
    "docs/public-memory-bundle.md",
    "docs/public-memory-bundle.json",
    "docs/TESTING_EVALUATION_GAPS.md",
    "docs/ara-memory-status-architecture.html",
)
PRIVATE_GITIGNORE_PATTERNS = (
    ".ara-memory/",
    ".ara-memory*/",
    "*.sqlite3",
    "*.zip",
    "ledger/",
    "spool/",
    "backups/",
)
PRIVATE_TRACKED_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".zip", ".bak", ".log")
PRIVATE_TRACKED_ROOTS = {
    ".ara-memory",
    ".codex",
    ".agents",
    "archive",
    "hot",
    "ledger",
    "spool",
    "backups",
}


@dataclass(slots=True)
class HandoffCheck:
    name: str
    status: str
    evidence: str

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class HandoffDoctorReport:
    repo: str
    status: str
    checks: list[HandoffCheck]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "status": self.status,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            f"# Ara Memory Handoff Doctor: {self.repo}",
            f"status: {self.status}",
            "",
            "## Checks",
        ]
        for check in self.checks:
            lines.append(f"- [{check.status}] {check.name}: {check.evidence}")
        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def run_handoff_doctor(
    *,
    repo: Path | str = ".",
    require_clean_worktree: bool = False,
) -> HandoffDoctorReport:
    repo_path = Path(repo).resolve()
    checks: list[HandoffCheck] = []
    checks.append(_required_files_check(repo_path))
    checks.append(_public_bundle_json_check(repo_path / "docs" / "public-memory-bundle.json"))
    checks.append(_public_bundle_markdown_check(repo_path / "docs" / "public-memory-bundle.md"))
    checks.append(_testing_gaps_check(repo_path / "docs" / "TESTING_EVALUATION_GAPS.md"))
    checks.append(_gitignore_check(repo_path / ".gitignore"))
    checks.append(_tracked_private_files_check(repo_path))
    checks.append(_privacy_check(repo_path))
    if require_clean_worktree:
        checks.append(_clean_worktree_check(repo_path))

    status = "fail" if any(check.status == "fail" for check in checks) else "watch" if any(
        check.status == "watch" for check in checks
    ) else "pass"
    recommendations = _recommendations(checks, require_clean_worktree=require_clean_worktree)
    return HandoffDoctorReport(
        repo=str(repo_path),
        status=status,
        checks=checks,
        recommendations=recommendations,
    )


def _required_files_check(repo: Path) -> HandoffCheck:
    missing = [path for path in REQUIRED_PUBLIC_FILES if not (repo / path).is_file()]
    if missing:
        return HandoffCheck("public handoff files", "fail", "missing: " + ", ".join(missing))
    return HandoffCheck("public handoff files", "pass", f"{len(REQUIRED_PUBLIC_FILES)} required files exist")


def _public_bundle_json_check(path: Path) -> HandoffCheck:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return HandoffCheck("public bundle json", "fail", f"cannot parse {path}: {exc}")

    failures: list[str] = []
    if payload.get("format") != PUBLIC_BUNDLE_FORMAT:
        failures.append("unexpected format")
    policy = payload.get("publication_policy", {})
    if policy.get("raw_memory_included") is not False:
        failures.append("raw_memory_included is not false")
    excluded = set(policy.get("private_paths_excluded", []))
    if ".ara-memory" not in excluded:
        failures.append(".ara-memory is not listed as excluded")
    stats = payload.get("memory_coverage", {}).get("stats", {})
    if int(stats.get("events", 0) or 0) <= 0 or int(stats.get("capsules", 0) or 0) <= 0:
        failures.append("coverage stats are empty")
    gates = payload.get("evaluation", {}).get("gates", [])
    if not gates:
        failures.append("evaluation gates are missing")
    if payload.get("status") == "fail" or any(gate.get("status") == "fail" for gate in gates):
        failures.append("bundle or evaluation gate is failing")
    if failures:
        return HandoffCheck("public bundle json", "fail", "; ".join(failures))
    gate_statuses = ",".join(sorted({str(gate.get("status")) for gate in gates}))
    return HandoffCheck(
        "public bundle json",
        "pass" if payload.get("status") == "pass" else "watch",
        f"format={payload.get('format')}, status={payload.get('status')}, gates={gate_statuses}",
    )


def _public_bundle_markdown_check(path: Path) -> HandoffCheck:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return HandoffCheck("public bundle markdown", "fail", f"cannot read {path}: {exc}")
    required = ("# Ara Memory Public Bundle", "## GitHub Boundary", "Raw `.ara-memory`")
    missing = [item for item in required if item not in text]
    if missing:
        return HandoffCheck("public bundle markdown", "fail", "missing sections: " + ", ".join(missing))
    return HandoffCheck("public bundle markdown", "pass", "boundary and recall-evidence sections are present")


def _testing_gaps_check(path: Path) -> HandoffCheck:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return HandoffCheck("testing handoff checklist", "fail", f"cannot read {path}: {exc}")
    required = ("Cross-machine handoff verification", "Publication Boundary", "Restore drill repeatability")
    missing = [item for item in required if item not in text]
    if missing:
        return HandoffCheck("testing handoff checklist", "fail", "missing checklist entries: " + ", ".join(missing))
    return HandoffCheck("testing handoff checklist", "pass", "handoff, publication, and restore checks are documented")


def _gitignore_check(path: Path) -> HandoffCheck:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return HandoffCheck("private gitignore boundary", "fail", f"cannot read {path}: {exc}")
    missing = [pattern for pattern in PRIVATE_GITIGNORE_PATTERNS if pattern not in text]
    if missing:
        return HandoffCheck("private gitignore boundary", "fail", "missing patterns: " + ", ".join(missing))
    return HandoffCheck("private gitignore boundary", "pass", f"{len(PRIVATE_GITIGNORE_PATTERNS)} private patterns are ignored")


def _tracked_private_files_check(repo: Path) -> HandoffCheck:
    tracked = _git_lines(repo, ["ls-files"])
    if tracked is None:
        return HandoffCheck("tracked private files", "watch", "git unavailable; skipped tracked-file boundary check")
    offenders = [path for path in tracked if _is_private_tracked_path(path)]
    if offenders:
        preview = ", ".join(offenders[:8])
        if len(offenders) > 8:
            preview += f", +{len(offenders) - 8} more"
        return HandoffCheck("tracked private files", "fail", preview)
    return HandoffCheck("tracked private files", "pass", f"{len(tracked)} tracked files inspected")


def _privacy_check(repo: Path) -> HandoffCheck:
    report = run_privacy_pre_push_gate(repo=repo)
    if not report.passed:
        failures = [finding for finding in report.findings if finding.severity == "fail"]
        preview = ", ".join(f"{item.path}:{item.kind}" for item in failures[:6])
        return HandoffCheck("privacy pre-push", "fail", preview or "privacy gate failed")
    if report.findings:
        return HandoffCheck(
            "privacy pre-push",
            "watch",
            f"passed with {len(report.findings)} reviewed warning(s); scanned_files={report.scanned_files}",
        )
    return HandoffCheck("privacy pre-push", "pass", f"scanned_files={report.scanned_files}, findings=0")


def _clean_worktree_check(repo: Path) -> HandoffCheck:
    lines = _git_lines(repo, ["status", "--short"])
    if lines is None:
        return HandoffCheck("clean worktree", "watch", "git unavailable; skipped clean-worktree check")
    if lines:
        return HandoffCheck("clean worktree", "fail", f"{len(lines)} changed path(s)")
    return HandoffCheck("clean worktree", "pass", "git status --short is empty")


def _recommendations(checks: list[HandoffCheck], *, require_clean_worktree: bool) -> list[str]:
    if any(check.status == "fail" for check in checks):
        return [
            "Fix failing handoff checks before treating GitHub as the portable continuation point.",
            "Regenerate public-memory-bundle after any private memory milestone, then rerun handoff-doctor.",
        ]
    recommendations = [
        "On a second machine, clone the repository and run `python -m ara_memory handoff-doctor --json` before continuing.",
        "Keep raw `.ara-memory` backups local; publish only the public bundle and aggregate verification status.",
    ]
    if not require_clean_worktree:
        recommendations.append("Use `--require-clean-worktree` immediately before a release push.")
    return recommendations


def _is_private_tracked_path(path: str) -> bool:
    portable = path.replace("\\", "/")
    parts = [part for part in portable.split("/") if part]
    if parts and parts[0] in PRIVATE_TRACKED_ROOTS:
        return True
    lower = portable.lower()
    return lower.endswith(PRIVATE_TRACKED_SUFFIXES)


def _git_lines(repo: Path, args: list[str]) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
