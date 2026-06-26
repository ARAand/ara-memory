from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ara_memory.compressors import estimate_tokens
from ara_memory.hot import HotStateBuilder
from ara_memory.recall import RecallCompiler
from ara_memory.storage import MemoryStore, SCHEMA_VERSION, row_to_capsule


@dataclass(slots=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
        }


@dataclass(slots=True)
class DoctorReport:
    scope: str
    passed: bool
    checks: list[DoctorCheck]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_text(self) -> str:
        lines = [f"# Ara Memory Doctor: {self.scope}", f"status: {'pass' if self.passed else 'fail'}"]
        for check in self.checks:
            marker = "OK" if check.passed else "FAIL"
            lines.append(f"- [{marker}] {check.name}: {check.detail}")
        return "\n".join(lines)


class MemoryDoctor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def run(
        self,
        *,
        scope: str = "global",
        recall_query: str = "current memory state",
        recall_budget: int = 1600,
        hot_budget: int = 1200,
        include_global: bool = True,
    ) -> DoctorReport:
        self.store.init()
        checks = [
            self._check_schema(),
            self._check_audit(),
            self._check_artifacts(scope=scope),
            self._check_hot(scope=scope, budget=hot_budget),
            self._check_recall(
                query=recall_query,
                scope=scope,
                budget=recall_budget,
                include_global=include_global,
            ),
        ]
        passed = all(check.passed or check.severity == "warning" for check in checks)
        return DoctorReport(scope=scope, passed=passed, checks=checks)

    def _check_schema(self) -> DoctorCheck:
        actual = self.store.schema_version()
        return DoctorCheck(
            name="schema_version",
            passed=actual == SCHEMA_VERSION,
            detail=f"expected {SCHEMA_VERSION}, found {actual}",
        )

    def _check_audit(self) -> DoctorCheck:
        from ara_memory.audit import MemoryAuditor

        text = MemoryAuditor(self.store).audit()
        passed = text.startswith("No memory hygiene issues")
        return DoctorCheck(name="audit", passed=passed, detail=text.splitlines()[0] if text else "empty audit output")

    def _check_artifacts(self, *, scope: str) -> DoctorCheck:
        summaries = [
            row_to_capsule(row)
            for row in self.store.list_capsules(scope=scope, status=None, kind="summary", limit=500)
            if row["status"] == "stable"
        ]
        issues = []
        latest_by_artifact: dict[str, int] = {}
        for cap in summaries:
            title = cap["title"]
            if title == "Latest artifact memory: c":
                issues.append("drive-letter artifact summary")
            if title.startswith("Latest artifact memory: ") and ": #" in title:
                issues.append("inline payload in artifact title")
            artifact = _summary_artifact(title)
            if artifact:
                latest_by_artifact[artifact] = latest_by_artifact.get(artifact, 0) + 1
            for tag in cap["tags"]:
                if isinstance(tag, str) and tag.startswith("artifact:") and (" # " in tag or len(tag) > 180):
                    issues.append("malformed artifact tag")
        duplicates = [artifact for artifact, count in latest_by_artifact.items() if count > 1]
        if duplicates:
            issues.append(f"duplicate latest summaries: {', '.join(sorted(duplicates)[:3])}")
        detail = "clean" if not issues else "; ".join(sorted(set(issues)))
        return DoctorCheck(name="artifact_hygiene", passed=not issues, detail=detail)

    def _check_hot(self, *, scope: str, budget: int) -> DoctorCheck:
        state = HotStateBuilder(self.store).read(scope=scope)
        if state is None:
            return DoctorCheck(name="hot_memory", passed=False, detail="missing hot memory", severity="warning")
        issues = []
        if state.estimated_tokens > budget:
            issues.append(f"{state.estimated_tokens} tokens exceeds budget {budget}")
        if "# Ara Hot Memory Scope:" in state.text:
            issues.append("collapsed markdown header")
        if "Latest artifact memory: c" in state.text:
            issues.append("contains drive-letter artifact summary")
        detail = f"{state.estimated_tokens} tokens" if not issues else "; ".join(issues)
        return DoctorCheck(name="hot_memory", passed=not issues, detail=detail)

    def _check_recall(self, *, query: str, scope: str, budget: int, include_global: bool) -> DoctorCheck:
        hot = HotStateBuilder(self.store).read(scope=scope)
        result = RecallCompiler(self.store).recall(
            query,
            scope=scope,
            budget=budget,
            include_global=include_global,
            hot_state=hot.text if hot else None,
        )
        tokens = estimate_tokens(result.pack)
        issues = []
        if tokens > budget:
            issues.append(f"{tokens} tokens exceeds budget {budget}")
        for section in ("# Ara Memory Pack", "## Consolidated Memory", "## Supporting Episodes"):
            if section not in result.pack:
                issues.append(f"missing {section}")
        if "# Ara Memory Pack Query:" in result.pack:
            issues.append("collapsed recall markdown")
        detail = f"{tokens} tokens" if not issues else "; ".join(issues)
        return DoctorCheck(name="recall_pack", passed=not issues, detail=detail)


def _summary_artifact(title: str) -> str:
    prefix = "Latest artifact memory: "
    if not title.startswith(prefix):
        return ""
    return title[len(prefix) :].strip().lower()
