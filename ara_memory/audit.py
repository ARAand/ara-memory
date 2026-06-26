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
                    if risk.assess_capsule(row_to_capsule(row)).should_quarantine
                ]
                if not found:
                    continue
            issues.append(f"## {name}")
            for row in found[:10]:
                issues.append(f"- {row['id']} [{row['kind']}/{row['status']}]: {row['title']}")
        if not issues:
            return "No memory hygiene issues found by the local audit rules."
        return "\n".join(["# Ara Memory Audit", *issues])
