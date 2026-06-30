from __future__ import annotations

from ara_memory.risk import MemoryRiskAssessor
from ara_memory.storage import MemoryStore
from ara_memory.storage import row_to_capsule


class MemoryAuditor:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def audit(self) -> str:
        rows = self.store.audit_rows()
        risk = MemoryRiskAssessor(self.store)
        issues: list[str] = []
        for name, found in rows.items():
            if not found:
                continue
            if name == "instruction_like":
                found = [
                    row
                    for row in found
                    if row["status"] in {"candidate", "stable"}
                    if risk.assess_capsule(row_to_capsule(row)).should_quarantine
                ]
                if not found:
                    continue
            issues.append(f"## {name}")
            for row in found[:10]:
                issues.append(f"- {row['id']} [{row['kind']}/{row['status']}]: {row['title']}")
        risk_issues = []
        for verdict in risk.assess_scope(scope=None, limit=500):
            row = self.store.get_capsule(verdict.capsule_id)
            if row is None or row["status"] not in {"candidate", "stable"}:
                continue
            if verdict.should_quarantine or verdict.has_sensitive_text or verdict.has_self_serving_text:
                risk_issues.append((row_to_capsule(row), verdict))
        if risk_issues:
            issues.append("## deterministic_risk")
            for cap, verdict in risk_issues[:10]:
                reason = "; ".join(verdict.reasons[:3])
                issues.append(
                    f"- {cap['id']} [{cap['kind']}/{cap['status']}]: {cap['title']} "
                    f"(score={verdict.score:.2f}; {reason})"
                )
        if not issues:
            return "No memory hygiene issues found by the local audit rules."
        return "\n".join(["# Ara Memory Audit", *issues])
