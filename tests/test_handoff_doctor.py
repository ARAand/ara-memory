from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
import unittest

from ara_memory.handoff import run_handoff_doctor
from ara_memory.public_bundle import PUBLIC_BUNDLE_FORMAT


class HandoffDoctorTests(unittest.TestCase):
    def test_handoff_doctor_passes_public_artifact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_public_handoff_repo(repo, bundle_status="pass")
            _git(repo, "init")
            _git(repo, "add", ".")

            report = run_handoff_doctor(repo=repo, require_clean_worktree=False)

            self.assertEqual(report.status, "pass")
            self.assertTrue(report.passed)
            by_name = {check.name: check for check in report.checks}
            self.assertEqual(by_name["public bundle json"].status, "pass")
            self.assertEqual(by_name["tracked private files"].status, "pass")

    def test_handoff_doctor_fails_when_private_memory_is_tracked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write_public_handoff_repo(repo, bundle_status="pass")
            private_ledger = repo / ".ara-memory" / "ledger" / "events.jsonl"
            private_ledger.parent.mkdir(parents=True)
            private_ledger.write_text('{"text":"private raw memory"}\n', encoding="utf-8")
            _git(repo, "init")
            _git(repo, "add", ".")
            _git(repo, "add", "-f", ".ara-memory/ledger/events.jsonl")

            report = run_handoff_doctor(repo=repo)

            self.assertEqual(report.status, "fail")
            by_name = {check.name: check for check in report.checks}
            self.assertEqual(by_name["tracked private files"].status, "fail")
            self.assertIn(".ara-memory/ledger/events.jsonl", by_name["tracked private files"].evidence)


def _write_public_handoff_repo(repo: Path, *, bundle_status: str) -> None:
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (repo / "README.md").write_text("# Ara Memory OS\n\nRun `python -m ara_memory handoff-doctor`.\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "\n".join(
            [
                ".ara-memory/",
                ".ara-memory*/",
                "*.sqlite3",
                "*.zip",
                "ledger/",
                "spool/",
                "backups/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "public-memory-bundle.md").write_text(
        "# Ara Memory Public Bundle\n\n"
        "## GitHub Boundary\n\n"
        "- Raw `.ara-memory` ledgers, SQLite databases, backups, archives, hot packs, and spool files are private local evidence.\n",
        encoding="utf-8",
    )
    (docs / "TESTING_EVALUATION_GAPS.md").write_text(
        "# Ara Memory Testing And Evaluation Gaps\n\n"
        "## Missing Tests And Evaluation Methods\n\n"
        "4. Cross-machine handoff verification\n\n"
        "8. Restore drill repeatability\n\n"
        "## Publication Boundary\n\n",
        encoding="utf-8",
    )
    (docs / "ara-memory-status-architecture.html").write_text("<!doctype html><title>Ara Memory</title>\n", encoding="utf-8")
    (docs / "public-memory-bundle.json").write_text(
        json.dumps(
            {
                "format": PUBLIC_BUNDLE_FORMAT,
                "scope": "ara-memory",
                "status": bundle_status,
                "publication_policy": {
                    "summary": "Publish public handoff only.",
                    "raw_memory_included": False,
                    "private_paths_excluded": [".ara-memory"],
                },
                "memory_coverage": {
                    "stats": {
                        "events": 1,
                        "capsules": 1,
                        "stable_capsules": 1,
                        "source_event_links": 1,
                        "relation_nodes": 1,
                        "relation_edges": 1,
                        "long_run_stress_runs": 1,
                        "schema_version": 29,
                    }
                },
                "evaluation": {
                    "gates": [
                        {
                            "name": "health",
                            "status": bundle_status,
                            "evidence": "test",
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
