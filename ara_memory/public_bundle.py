from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ara_memory.compressors import compact_text
from ara_memory.risk import redact_sensitive_text


PUBLIC_BUNDLE_FORMAT = "ara-memory-public-bundle-v1"
DEFAULT_PUBLIC_QUERY = "current Ara natural memory architecture purpose recall safety and evaluation gaps"


@dataclass(slots=True)
class PublicMemoryBundle:
    scope: str
    status: str
    payload: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_markdown(self) -> str:
        stats = self.payload["memory_coverage"]["stats"]
        gates = self.payload["evaluation"]["gates"]
        gaps = self.payload["evaluation"]["missing_tests_and_methods"]
        recall = self.payload["redacted_recall_evidence"]
        lines = [
            "# Ara Memory Public Bundle",
            "",
            f"- format: `{self.payload['format']}`",
            f"- scope: `{self.scope}`",
            f"- status: `{self.status}`",
            f"- publication_policy: {self.payload['publication_policy']['summary']}",
            "",
            "## Memory Coverage",
            "",
            f"- source events: {stats['events']}",
            f"- capsules: {stats['capsules']}",
            f"- stable capsules: {stats['stable_capsules']}",
            f"- source event links: {stats['source_event_links']}",
            f"- relation nodes: {stats['relation_nodes']}",
            f"- relation edges: {stats['relation_edges']}",
            f"- long-run stress runs: {stats['long_run_stress_runs']}",
            f"- schema version: {stats['schema_version']}",
            f"- public coverage digest: `{self.payload['memory_coverage']['coverage_digest']}`",
            "",
            "## Evaluation Gates",
            "",
        ]
        for gate in gates:
            lines.append(f"- `{gate['name']}`: {gate['status']} - {gate['evidence']}")
        lines.extend(
            [
                "",
                "## Missing Tests And Evaluation Methods",
                "",
            ]
        )
        for item in gaps:
            lines.append(f"- [{item['status']}] {item['name']}: {item['method']}")
        lines.extend(
            [
                "",
                "## Redacted Recall Evidence",
                "",
                f"- query: {recall['query']}",
                f"- budget: {recall['budget']}",
                f"- estimated tokens: {recall['estimated_tokens']}",
                f"- capsules selected: {recall['capsules_selected']}",
                "",
                "```text",
                recall["preview"],
                "```",
                "",
                "## GitHub Boundary",
                "",
                "- Raw `.ara-memory` ledgers, SQLite databases, backups, archives, hot packs, and spool files are private local evidence.",
                "- This bundle is the publishable memory representation: redacted, bounded, and evidence-oriented.",
            ]
        )
        return "\n".join(lines) + "\n"


def build_public_memory_bundle(
    memory: Any,
    *,
    scope: str = "global",
    query: str = DEFAULT_PUBLIC_QUERY,
    budget: int = 1600,
    include_global: bool = True,
    regression_cases: list[Any] | None = None,
    regression_baseline: dict[str, Any] | None = None,
    repo: Path | None = None,
) -> PublicMemoryBundle:
    stats = memory.stats()
    health = memory.health(
        scope=scope,
        query="public memory bundle health",
        regression_cases=regression_cases,
        regression_baseline=regression_baseline,
    )
    stress = memory.long_run_stress_trend(scope=scope, include_global=include_global, min_samples=3)
    quality = memory.recall_quality(
        query,
        scope=scope,
        budgets=[800, budget, 2500],
        include_global=include_global,
        recall_budget=budget,
        min_evaluated=3,
    )
    recall = memory.recall_context(
        query,
        scope=scope,
        budgets=[budget],
        include_hot=True,
        include_global=include_global,
    )
    privacy = memory.privacy_pre_push(repo=repo or Path.cwd())
    gates = _evaluation_gates(health, stress, quality, privacy)
    gaps = _missing_tests_and_methods(stress, quality, privacy)
    status = _bundle_status(gates, gaps)
    payload = {
        "format": PUBLIC_BUNDLE_FORMAT,
        "scope": scope,
        "status": status,
        "publication_policy": {
            "summary": "Publish redacted coverage and evaluation evidence, never raw private memory stores.",
            "raw_memory_included": False,
            "private_paths_excluded": [
                ".ara-memory",
                ".ara-memory/events.jsonl",
                ".ara-memory/memory.sqlite3",
                ".ara-memory/backups",
                ".ara-memory/archive",
                ".ara-memory/hot",
                ".ara-memory/spool",
            ],
        },
        "memory_coverage": {
            "stats": _public_stats(stats),
            "coverage_digest": _coverage_digest(stats),
        },
        "evaluation": {
            "gates": gates,
            "missing_tests_and_methods": gaps,
        },
        "redacted_recall_evidence": _redacted_recall_evidence(recall, query=query, budget=budget),
    }
    return PublicMemoryBundle(scope=scope, status=status, payload=payload)


def write_public_memory_bundle(bundle: PublicMemoryBundle, output: Path, *, json_output: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if json_output:
        output.write_text(json.dumps(bundle.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        output.write_text(bundle.to_markdown(), encoding="utf-8")
    return output


def _public_stats(stats: dict[str, Any]) -> dict[str, int]:
    keys = [
        "events",
        "capsules",
        "stable_capsules",
        "source_event_links",
        "relation_nodes",
        "relation_edges",
        "long_run_stress_runs",
        "schema_version",
    ]
    return {key: int(stats.get(key, 0) or 0) for key in keys}


def _coverage_digest(stats: dict[str, Any]) -> str:
    material = json.dumps(_public_stats(stats), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _evaluation_gates(health: Any, stress: Any, quality: Any, privacy: Any) -> list[dict[str, str]]:
    return [
        {
            "name": "health",
            "status": health.status,
            "evidence": (
                f"score={health.score}/100, "
                f"signals={sum(1 for signal in health.signals if signal.passed)}/{len(health.signals)}"
            ),
        },
        {
            "name": "long-run-stress-trend",
            "status": stress.status,
            "evidence": (
                f"samples={stress.totals['runs']}, fail={stress.totals['fail']}, "
                f"score_delta={stress.score_delta}, token_growth_delta={stress.token_growth_delta}"
            ),
        },
        {
            "name": "recall-quality",
            "status": quality.status,
            "evidence": (
                f"critic={quality.critic_summary['status']}, "
                f"policy={quality.policy_summary['status']}, "
                f"stress={quality.stress_trend_summary['status']}"
            ),
        },
        {
            "name": "privacy-pre-push",
            "status": privacy.status,
            "evidence": f"passed={privacy.passed}, scanned_files={privacy.scanned_files}, findings={len(privacy.findings)}",
        },
    ]


def _missing_tests_and_methods(stress: Any, quality: Any, privacy: Any) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if stress.status != "pass":
        gaps.append(
            {
                "name": "long-run stress pass evidence",
                "status": "watch",
                "method": "Collect reviewed recall-critic-impact and working-memory-impact outcomes until stress runs stop returning watch.",
            }
        )
    if quality.critic_impact_summary["status"] != "pass":
        gaps.append(
            {
                "name": "recall critic outcome evaluation",
                "status": quality.critic_impact_summary["status"],
                "method": "Record at least three reviewed recall-critic-impact outcomes for real turns, then run recall-critic-impact-eval.",
            }
        )
    if quality.memory_impact_summary["status"] != "pass":
        gaps.append(
            {
                "name": "working memory outcome evaluation",
                "status": quality.memory_impact_summary["status"],
                "method": "Record at least three reviewed working-memory-impact outcomes for projected capsules, then run working-memory-impact-eval.",
            }
        )
    if privacy.status != "pass":
        gaps.append(
            {
                "name": "public repository privacy warnings",
                "status": privacy.status,
                "method": "Keep fixture warnings reviewed; fail the push if private memory roots, ledgers, databases, backups, or non-fixture secrets appear.",
            }
        )
    if not gaps:
        gaps.append(
            {
                "name": "continuous evaluation",
                "status": "pass",
                "method": "Keep health, recall-regression, privacy-pre-push, recall-quality, and long-run-stress-trend in the release gate.",
            }
        )
    return gaps


def _bundle_status(gates: list[dict[str, str]], gaps: list[dict[str, str]]) -> str:
    if any(item["status"] == "fail" for item in gates + gaps):
        return "fail"
    if any(item["status"] == "watch" for item in gates + gaps):
        return "watch"
    return "pass"


def _redacted_recall_evidence(recall: Any, *, query: str, budget: int) -> dict[str, Any]:
    text = redact_sensitive_text(getattr(recall, "pack", recall.to_text()))
    return {
        "query": redact_sensitive_text(query),
        "budget": budget,
        "estimated_tokens": int(recall.diagnostics.get("estimated_tokens_after", 0)),
        "capsules_selected": int(recall.diagnostics.get("capsules_selected", 0)),
        "preview": compact_text(text, limit=2400),
    }
