from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara_memory.lock import FileLock
from ara_memory.regression import load_recall_regression_baseline, load_recall_regression_cases
from ara_memory.relation_merge import (
    list_relation_merge_review_queue,
    record_relation_merge_review_queue,
    review_relation_merge_witnesses,
)
from ara_memory.reconsolidation import (
    list_reconsolidation_review_queue,
    record_reconsolidation_review_queue,
)


@dataclass(slots=True)
class WorkerStep:
    name: str
    passed: bool
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(slots=True)
class WorkerReport:
    scope: str
    passed: bool
    steps: list[WorkerStep] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "skipped": self.skipped,
            "reason": self.reason,
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(slots=True)
class WorkerLoopReport:
    scope: str
    passed: bool
    iterations_requested: int
    iterations_run: int
    interval_seconds: float
    reports: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "iterations_requested": self.iterations_requested,
            "iterations_run": self.iterations_run,
            "interval_seconds": self.interval_seconds,
            "reports": self.reports,
        }


def run_memory_worker(
    memory: Any,
    *,
    scope: str = "global",
    spool_limit: int = 25,
    processing_stale_seconds: int = 3600,
    quality_limit: int = 500,
    review_limit: int = 25,
    triage_limit: int = 500,
    apply_review: bool = False,
    episode_summary: bool = True,
    episode_summary_min_group_size: int = 5,
    episode_summary_session_min_group_size: int = 20,
    episode_summary_limit: int = 50,
    candidate_summary: bool = True,
    candidate_summary_min_group_size: int = 3,
    candidate_summary_limit: int = 80,
    governance_probe: bool = True,
    governance_query: str = "",
    governance_max_input_tokens: int = 0,
    governance_max_model_cost_usd: float = 0.0,
    doctor_query: str = "current memory state",
    recall_budget: int = 1600,
    hot_budget: int = 1200,
    regression_manifest: Path | None = None,
    regression_baseline: Path | None = None,
    regression_baseline_drift_warn_only: bool = False,
    run_maintenance_step: bool = True,
    vacuum: bool = True,
    report_item_limit: int = 20,
    use_lock: bool = True,
    lock_stale_seconds: int = 3600,
) -> WorkerReport:
    memory.init()
    steps: list[WorkerStep] = []
    if use_lock:
        lock = FileLock(memory.store.root, "worker", stale_seconds=lock_stale_seconds)
        lock_result = lock.acquire()
        steps.append(WorkerStep("worker_lock", True, lock_result.as_dict()))
        if not lock_result.acquired:
            return WorkerReport(
                scope=scope,
                passed=True,
                skipped=True,
                reason="worker_already_running",
                steps=steps,
            )
        try:
            return _run_locked_worker(
                memory,
                scope=scope,
                spool_limit=spool_limit,
                processing_stale_seconds=processing_stale_seconds,
                quality_limit=quality_limit,
                review_limit=review_limit,
                triage_limit=triage_limit,
                apply_review=apply_review,
                episode_summary=episode_summary,
                episode_summary_min_group_size=episode_summary_min_group_size,
                episode_summary_session_min_group_size=episode_summary_session_min_group_size,
                episode_summary_limit=episode_summary_limit,
                candidate_summary=candidate_summary,
                candidate_summary_min_group_size=candidate_summary_min_group_size,
                candidate_summary_limit=candidate_summary_limit,
                governance_probe=governance_probe,
                governance_query=governance_query,
                governance_max_input_tokens=governance_max_input_tokens,
                governance_max_model_cost_usd=governance_max_model_cost_usd,
                doctor_query=doctor_query,
                recall_budget=recall_budget,
                hot_budget=hot_budget,
                regression_manifest=regression_manifest,
                regression_baseline=regression_baseline,
                regression_baseline_drift_warn_only=regression_baseline_drift_warn_only,
                run_maintenance_step=run_maintenance_step,
                vacuum=vacuum,
                report_item_limit=report_item_limit,
                initial_steps=steps,
            )
        finally:
            lock.release()
    return _run_locked_worker(
        memory,
        scope=scope,
        spool_limit=spool_limit,
        processing_stale_seconds=processing_stale_seconds,
        quality_limit=quality_limit,
        review_limit=review_limit,
        triage_limit=triage_limit,
        apply_review=apply_review,
        episode_summary=episode_summary,
        episode_summary_min_group_size=episode_summary_min_group_size,
        episode_summary_session_min_group_size=episode_summary_session_min_group_size,
        episode_summary_limit=episode_summary_limit,
        candidate_summary=candidate_summary,
        candidate_summary_min_group_size=candidate_summary_min_group_size,
        candidate_summary_limit=candidate_summary_limit,
        governance_probe=governance_probe,
        governance_query=governance_query,
        governance_max_input_tokens=governance_max_input_tokens,
        governance_max_model_cost_usd=governance_max_model_cost_usd,
        doctor_query=doctor_query,
        recall_budget=recall_budget,
        hot_budget=hot_budget,
        regression_manifest=regression_manifest,
        regression_baseline=regression_baseline,
        regression_baseline_drift_warn_only=regression_baseline_drift_warn_only,
        run_maintenance_step=run_maintenance_step,
        vacuum=vacuum,
        report_item_limit=report_item_limit,
        initial_steps=steps,
    )


def run_worker_loop(
    memory: Any,
    *,
    iterations: int = 1,
    interval_seconds: float = 60.0,
    stop_on_failure: bool = True,
    **worker_kwargs: Any,
) -> WorkerLoopReport:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    reports: list[dict[str, Any]] = []
    scope = str(worker_kwargs.get("scope", "global"))
    for index in range(iterations):
        report = run_memory_worker(memory, **worker_kwargs)
        reports.append(_compact_worker_loop_report(report, iteration=index + 1))
        if not report.passed and stop_on_failure:
            break
        if index < iterations - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)
    return WorkerLoopReport(
        scope=scope,
        passed=all(item["passed"] for item in reports),
        iterations_requested=iterations,
        iterations_run=len(reports),
        interval_seconds=interval_seconds,
        reports=reports,
    )


def _run_locked_worker(
    memory: Any,
    *,
    scope: str,
    spool_limit: int,
    processing_stale_seconds: int,
    quality_limit: int,
    review_limit: int,
    triage_limit: int,
    apply_review: bool,
    episode_summary: bool,
    episode_summary_min_group_size: int,
    episode_summary_session_min_group_size: int,
    episode_summary_limit: int,
    candidate_summary: bool,
    candidate_summary_min_group_size: int,
    candidate_summary_limit: int,
    governance_probe: bool,
    governance_query: str,
    governance_max_input_tokens: int,
    governance_max_model_cost_usd: float,
    doctor_query: str,
    recall_budget: int,
    hot_budget: int,
    regression_manifest: Path | None,
    regression_baseline: Path | None,
    regression_baseline_drift_warn_only: bool,
    run_maintenance_step: bool,
    vacuum: bool,
    report_item_limit: int,
    initial_steps: list[WorkerStep],
) -> WorkerReport:
    steps = list(initial_steps)

    drain = memory.drain_spool(limit=spool_limit, scope=scope, processing_stale_seconds=processing_stale_seconds)
    steps.append(WorkerStep("drain_spool", drain.passed, drain.as_dict()))
    if not drain.passed:
        steps.append(
            WorkerStep(
                "worker_guard",
                False,
                {
                    "reason": "drain_spool failed; skipped behavior-changing worker steps",
                    "apply_review": apply_review,
                },
            )
        )
        return WorkerReport(scope=scope, passed=False, steps=steps)

    if episode_summary:
        episode_payload = {
            "patterns": [
                memory.episode_summary(
                    scope=scope,
                    pattern=pattern,
                    min_group_size=episode_summary_session_min_group_size
                    if pattern == "session"
                    else min(episode_summary_min_group_size, 3)
                    if pattern == "git_status"
                    else episode_summary_min_group_size,
                    limit=episode_summary_limit,
                    dry_run=False,
                ).as_dict()
                for pattern in ("command", "file_artifact", "git_status", "session")
            ]
        }
        episode_payload["summaries_created"] = sum(item["summaries_created"] for item in episode_payload["patterns"])
        episode_payload["superseded"] = sum(item["superseded"] for item in episode_payload["patterns"])
        steps.append(WorkerStep("episode_summary", True, _compact_episode_summary(episode_payload, limit=report_item_limit)))

    if candidate_summary:
        candidate_payload = memory.candidate_summary(
            scope=scope,
            pattern="all",
            min_group_size=candidate_summary_min_group_size,
            limit=candidate_summary_limit,
            dry_run=False,
        ).as_dict()
        steps.append(WorkerStep("candidate_summary", True, _compact_episode_summary(candidate_payload, limit=report_item_limit)))

    quality = memory.quality(scope=scope, limit=quality_limit, persist=True)
    quality_payload = _compact_quality(quality.as_dict(), limit=report_item_limit)
    steps.append(WorkerStep("quality", True, quality_payload))

    if governance_probe:
        governance_payload = _governance_probe_payload(
            memory,
            scope=scope,
            query=governance_query or doctor_query,
            recall_budget=recall_budget,
            hot_budget=hot_budget,
            max_input_tokens=governance_max_input_tokens,
            max_model_cost_usd=governance_max_model_cost_usd,
        )
        steps.append(WorkerStep("governance_probe", bool(governance_payload["passed"]), governance_payload))

    review = memory.review_worker(scope=scope, limit=review_limit, dry_run=not apply_review)
    review_payload = _limit_items(review.as_dict(), "items", limit=report_item_limit)
    review_passed = not apply_review or review.changed >= 0
    steps.append(WorkerStep("review_worker", review_passed, review_payload))

    triage = memory.review_triage(scope=scope, limit=triage_limit, examples_per_group=min(3, report_item_limit))
    triage_payload = _limit_items(triage.as_dict(), "groups", limit=report_item_limit)
    steps.append(WorkerStep("review_triage", True, triage_payload))

    relation_review_payload = _relation_merge_review_worker_payload(
        memory,
        scope=scope,
        review_limit=review_limit,
        regression_manifest=regression_manifest,
        regression_baseline=regression_baseline,
        regression_baseline_drift_warn_only=regression_baseline_drift_warn_only,
        report_item_limit=report_item_limit,
    )
    steps.append(
        WorkerStep(
            "relation_merge_review",
            bool(relation_review_payload["passed"]),
            relation_review_payload,
        )
    )

    relation_queue = list_relation_merge_review_queue(memory.store, scope=scope, status="open", limit=triage_limit)
    relation_queue_payload = _relation_review_queue_worker_payload(
        relation_queue.as_dict(),
        limit=report_item_limit,
    )
    steps.append(
        WorkerStep(
            "relation_review_queue",
            int(relation_queue_payload["fail_count"]) == 0,
            relation_queue_payload,
        )
    )

    reconsolidation_review_payload = _reconsolidation_review_worker_payload(
        memory,
        scope=scope,
        review_limit=review_limit,
        regression_manifest=regression_manifest,
        regression_baseline=regression_baseline,
        regression_baseline_drift_warn_only=regression_baseline_drift_warn_only,
        report_item_limit=report_item_limit,
    )
    steps.append(
        WorkerStep(
            "reconsolidation_review",
            bool(reconsolidation_review_payload["passed"]),
            reconsolidation_review_payload,
        )
    )

    reconsolidation_queue = list_reconsolidation_review_queue(memory, scope=scope, status="open", limit=triage_limit)
    reconsolidation_queue_payload = _reconsolidation_review_queue_worker_payload(
        reconsolidation_queue.as_dict(),
        limit=report_item_limit,
    )
    steps.append(
        WorkerStep(
            "reconsolidation_review_queue",
            int(reconsolidation_queue_payload["fail_count"]) == 0
            and int(reconsolidation_queue_payload["blockers"]) == 0,
            reconsolidation_queue_payload,
        )
    )

    doctor = memory.doctor(
        scope=scope,
        recall_query=doctor_query,
        recall_budget=recall_budget,
        hot_budget=hot_budget,
    )
    steps.append(WorkerStep("doctor", doctor.passed, doctor.as_dict()))

    if regression_manifest is not None:
        cases = load_recall_regression_cases(regression_manifest)
        baseline = load_recall_regression_baseline(regression_baseline)
        regression = memory.recall_regression(cases, baseline=baseline)
        regression_payload = _regression_worker_payload(
            regression.as_dict(),
            baseline_drift_warn_only=regression_baseline_drift_warn_only,
        )
        regression_passed = bool(
            regression_payload["cases_passed"]
            if regression_baseline_drift_warn_only
            else regression_payload["passed"]
        )
        steps.append(WorkerStep("recall_regression", regression_passed, regression_payload))

    if run_maintenance_step:
        maintenance = memory.maintenance(vacuum=vacuum)
        steps.append(
            WorkerStep(
                "maintenance",
                maintenance.sqlite_integrity == "ok",
                maintenance.as_dict(),
            )
        )

    passed = all(step.passed for step in steps)
    return WorkerReport(
        scope=scope,
        passed=passed,
        reason="" if passed else _worker_failure_reason(steps),
        steps=steps,
    )


def _compact_quality(payload: dict[str, Any], *, limit: int) -> dict[str, Any]:
    compact = dict(payload)
    items = list(compact.get("items", []))
    queue = list(compact.get("queue", []))
    compact["items"] = items[:limit]
    compact["items_truncated"] = max(0, len(items) - limit)
    compact["queue"] = queue[:limit]
    compact["queue_truncated"] = max(0, len(queue) - limit)
    return compact


def _limit_items(payload: dict[str, Any], key: str, *, limit: int) -> dict[str, Any]:
    compact = dict(payload)
    items = list(compact.get(key, []))
    compact[key] = items[:limit]
    compact[f"{key}_truncated"] = max(0, len(items) - limit)
    return compact


def _compact_episode_summary(payload: dict[str, Any], *, limit: int) -> dict[str, Any]:
    compact = dict(payload)
    patterns = []
    for item in compact.get("patterns", []):
        item = dict(item)
        groups = []
        for group in item.get("groups", []):
            group = dict(group)
            source_capsule_ids = list(group.get("source_capsule_ids", []))
            example_titles = list(group.get("example_titles", []))
            group["source_capsule_ids"] = source_capsule_ids[:limit]
            group["source_capsule_ids_truncated"] = max(0, len(source_capsule_ids) - limit)
            group["example_titles"] = example_titles[: min(5, limit)]
            groups.append(group)
        original_groups = list(item.get("groups", []))
        item["groups"] = groups[:limit]
        item["groups_truncated"] = max(0, len(original_groups) - limit)
        patterns.append(item)
    compact["patterns"] = patterns
    return compact


def _compact_worker_loop_report(report: WorkerReport, *, iteration: int) -> dict[str, Any]:
    by_name = {step.name: step for step in report.steps}
    drain = by_name.get("drain_spool")
    triage = by_name.get("review_triage")
    relation_review = by_name.get("relation_merge_review")
    relation_queue = by_name.get("relation_review_queue")
    reconsolidation_review = by_name.get("reconsolidation_review")
    reconsolidation_queue = by_name.get("reconsolidation_review_queue")
    doctor = by_name.get("doctor")
    governance = by_name.get("governance_probe")
    regression = by_name.get("recall_regression")
    maintenance = by_name.get("maintenance")
    episode = by_name.get("episode_summary")
    failed_steps = [step.name for step in report.steps if not step.passed]
    return {
        "iteration": iteration,
        "passed": report.passed,
        "skipped": report.skipped,
        "reason": report.reason,
        "failed_steps": failed_steps,
        "drain": {
            "processed": _detail_value(drain, "processed", 0),
            "succeeded": _detail_value(drain, "succeeded", 0),
            "failed": _detail_value(drain, "failed", 0),
            "recovered": _detail_value(drain, "recovered", 0),
        },
        "review_triage": {
            "total_open": _detail_value(triage, "total_open", 0),
            "groups": len((triage.detail.get("groups", []) if triage else [])),
            "groups_truncated": _detail_value(triage, "groups_truncated", 0),
        },
        "relation_merge_review": {
            "passed": bool(relation_review.detail.get("passed", False)) if relation_review else None,
            "reviewed": _detail_value(relation_review, "reviewed", 0),
            "fail_count": _detail_value(relation_review, "fail_count", 0),
            "watch_count": _detail_value(relation_review, "watch_count", 0),
            "recorded": _detail_value(relation_review, "queue_recorded", 0),
            "resolved": _detail_value(relation_review, "queue_resolved", 0),
        },
        "relation_review_queue": {
            "open": _detail_value(relation_queue, "total_open", 0),
            "fail_count": _detail_value(relation_queue, "fail_count", 0),
            "watch_count": _detail_value(relation_queue, "watch_count", 0),
            "open_items_truncated": _detail_value(relation_queue, "open_items_truncated", 0),
        },
        "reconsolidation_review": {
            "passed": bool(reconsolidation_review.detail.get("passed", False)) if reconsolidation_review else None,
            "reviewed": _detail_value(reconsolidation_review, "reviewed", 0),
            "fail_count": _detail_value(reconsolidation_review, "fail_count", 0),
            "watch_count": _detail_value(reconsolidation_review, "watch_count", 0),
            "recorded": _detail_value(reconsolidation_review, "queue_recorded", 0),
            "resolved": _detail_value(reconsolidation_review, "queue_resolved", 0),
        },
        "reconsolidation_review_queue": {
            "open": _detail_value(reconsolidation_queue, "total_open", 0),
            "fail_count": _detail_value(reconsolidation_queue, "fail_count", 0),
            "watch_count": _detail_value(reconsolidation_queue, "watch_count", 0),
            "blockers": _detail_value(reconsolidation_queue, "blockers", 0),
            "open_items_truncated": _detail_value(reconsolidation_queue, "open_items_truncated", 0),
        },
        "episode_summary": {
            "summaries_created": _detail_value(episode, "summaries_created", 0),
            "superseded": _detail_value(episode, "superseded", 0),
        },
        "doctor_passed": bool(doctor.detail.get("passed", False)) if doctor else None,
        "governance_probe": {
            "passed": bool(governance.detail.get("passed", False)) if governance else None,
            "budget_profile": _detail_value(governance, "budget_profile", ""),
            "fallback_strategy": _detail_value(governance, "fallback_strategy", ""),
            "fallback_model_context": _detail_value(governance, "fallback_model_context", ""),
            "recall_quality": _detail_value(governance, "recall_quality_status", ""),
            "cost_gate": _detail_value(governance, "cost_gate_status", ""),
            "risks": _detail_value(governance, "risks", []),
        },
        "recall_regression_passed": bool(regression.detail.get("passed", False)) if regression else None,
        "recall_regression_cases_passed": bool(regression.detail.get("cases_passed", False)) if regression else None,
        "recall_regression_baseline_passed": bool(regression.detail.get("baseline_passed", False)) if regression else None,
        "recall_regression_baseline_warn_only": (
            bool(regression.detail.get("baseline_drift_warn_only", False)) if regression else None
        ),
        "maintenance_integrity": maintenance.detail.get("sqlite_integrity") if maintenance else None,
    }


def _regression_worker_payload(payload: dict[str, Any], *, baseline_drift_warn_only: bool) -> dict[str, Any]:
    annotated = dict(payload)
    cases = [item for item in annotated.get("cases", []) if isinstance(item, dict)]
    baseline = [item for item in annotated.get("baseline_comparison", []) if isinstance(item, dict)]
    annotated["cases_passed"] = all(bool(item.get("passed", False)) for item in cases)
    annotated["baseline_passed"] = all(bool(item.get("passed", False)) for item in baseline)
    annotated["baseline_drift_warn_only"] = baseline_drift_warn_only
    return annotated


def _governance_probe_payload(
    memory: Any,
    *,
    scope: str,
    query: str,
    recall_budget: int,
    hot_budget: int,
    max_input_tokens: int,
    max_model_cost_usd: float,
) -> dict[str, Any]:
    report = memory.govern_turn(
        {
            "prompt": query,
            "assistant": "Scheduled memory worker governance probe.",
        },
        scope=scope,
        budgets=[recall_budget],
        working_budget=hot_budget,
        max_input_tokens=max_input_tokens,
        max_model_cost_usd=max_model_cost_usd,
    )
    payload = report.as_dict()
    actions = [
        {
            "name": str(action.get("name", "")),
            "status": str(action.get("status", "")),
            "reason": str(action.get("reason", "")),
        }
        for action in payload.get("actions", [])
        if isinstance(action, dict) and action.get("status") in {"block", "watch"}
    ]
    return {
        "passed": bool(payload["passed"]),
        "query": query,
        "budget_profile": payload["budget_profile"]["name"],
        "budget_profile_reason": payload["budget_profile"]["reason"],
        "budget_profile_budgets": list(payload["budget_profile"]["budgets"]),
        "fallback_strategy": payload["fallback_plan"]["strategy"],
        "fallback_model_context": payload["fallback_plan"]["model_context"],
        "recall_quality_status": payload["recall_quality"]["status"],
        "recall_quality_recommendations": list(payload["recall_quality"].get("recommendations", []))[:5],
        "cost_gate_status": payload["cost_gate"]["status"],
        "cost_gate_reasons": list(payload["cost_gate"].get("reasons", []))[:5],
        "projection_gate_status": payload["projection_gate"]["status"],
        "working_memory_items": payload["working_memory"]["items"],
        "working_memory_tokens": payload["working_memory"]["estimated_tokens"],
        "selected_input_tokens": payload["cost"]["selected_input_tokens"],
        "estimated_model_cost_usd": payload["cost"]["estimated_model_cost_usd"],
        "avoided_raw_tokens": payload["cost"]["avoided_raw_tokens"],
        "risks": list(payload.get("risks", []))[:8],
        "actions": actions[:8],
    }


def _relation_merge_review_worker_payload(
    memory: Any,
    *,
    scope: str,
    review_limit: int,
    regression_manifest: Path | None,
    regression_baseline: Path | None,
    regression_baseline_drift_warn_only: bool,
    report_item_limit: int,
) -> dict[str, Any]:
    review = review_relation_merge_witnesses(memory.store, scope=scope, limit=review_limit)
    payload = _limit_items(review.as_dict(), "items", limit=report_item_limit)
    regression_payload: dict[str, Any] | None = None
    relation_regression_passed = True
    if regression_manifest is not None:
        cases = load_recall_regression_cases(regression_manifest)
        baseline = load_recall_regression_baseline(regression_baseline)
        regression = memory.recall_regression(cases, baseline=baseline)
        regression_payload = _regression_worker_payload(
            regression.as_dict(),
            baseline_drift_warn_only=regression_baseline_drift_warn_only,
        )
        relation_regression_passed = bool(
            regression_payload["cases_passed"]
            if regression_baseline_drift_warn_only
            else regression_payload["passed"]
        )
        queue_regression_payload = dict(regression_payload)
        queue_regression_payload["passed"] = relation_regression_passed
    else:
        queue_regression_payload = None

    queue = record_relation_merge_review_queue(
        memory.store,
        review,
        regression=queue_regression_payload,
        limit=review_limit,
    )
    payload["regression"] = regression_payload
    payload["relation_regression_passed"] = relation_regression_passed
    payload["queue_recorded"] = queue.recorded
    payload["queue_resolved"] = queue.resolved
    payload["queue_open"] = len(queue.open_items)
    payload["queue"] = _relation_review_queue_worker_payload(queue.as_dict(), limit=report_item_limit)
    payload["passed"] = bool(review.passed and relation_regression_passed)
    return payload


def _relation_review_queue_worker_payload(payload: dict[str, Any], *, limit: int) -> dict[str, Any]:
    compact = _limit_items(payload, "open_items", limit=limit)
    items = list(payload.get("open_items", []))
    compact["total_open"] = len(items)
    compact["fail_count"] = sum(1 for item in items if item.get("review_status") == "fail")
    compact["watch_count"] = sum(1 for item in items if item.get("review_status") == "watch")
    return compact


def _reconsolidation_review_worker_payload(
    memory: Any,
    *,
    scope: str,
    review_limit: int,
    regression_manifest: Path | None,
    regression_baseline: Path | None,
    regression_baseline_drift_warn_only: bool,
    report_item_limit: int,
) -> dict[str, Any]:
    review = memory.review_reconsolidation(scope=scope, limit=review_limit)
    payload = _limit_items(review.as_dict(), "items", limit=report_item_limit)
    regression_payload: dict[str, Any] | None = None
    reconsolidation_regression_passed = True
    if regression_manifest is not None:
        cases = load_recall_regression_cases(regression_manifest)
        baseline = load_recall_regression_baseline(regression_baseline)
        regression = memory.recall_regression(cases, baseline=baseline)
        regression_payload = _regression_worker_payload(
            regression.as_dict(),
            baseline_drift_warn_only=regression_baseline_drift_warn_only,
        )
        reconsolidation_regression_passed = bool(
            regression_payload["cases_passed"]
            if regression_baseline_drift_warn_only
            else regression_payload["passed"]
        )
        queue_regression_payload = dict(regression_payload)
        queue_regression_payload["passed"] = reconsolidation_regression_passed
        review.regression = queue_regression_payload

    queue = record_reconsolidation_review_queue(
        memory,
        review,
        limit=review_limit,
    )
    payload["regression"] = regression_payload
    payload["reconsolidation_regression_passed"] = reconsolidation_regression_passed
    payload["queue_recorded"] = queue.recorded
    payload["queue_resolved"] = queue.resolved
    payload["queue_open"] = len(queue.open_items)
    payload["queue"] = _reconsolidation_review_queue_worker_payload(queue.as_dict(), limit=report_item_limit)
    payload["passed"] = bool(review.passed and reconsolidation_regression_passed)
    return payload


def _reconsolidation_review_queue_worker_payload(payload: dict[str, Any], *, limit: int) -> dict[str, Any]:
    compact = _limit_items(payload, "open_items", limit=limit)
    items = list(payload.get("open_items", []))
    compact["total_open"] = len(items)
    compact["fail_count"] = sum(1 for item in items if item.get("review_status") == "fail")
    compact["watch_count"] = sum(1 for item in items if item.get("review_status") == "watch")
    compact["blockers"] = sum(
        1 for item in items if str(item.get("action", "")).startswith("block-strong-reconsolidation")
    )
    return compact


def _worker_failure_reason(steps: list[WorkerStep]) -> str:
    failed = [step for step in steps if not step.passed]
    if not failed:
        return ""
    failed.sort(key=lambda step: 0 if step.name == "recall_regression" else 1)
    parts: list[str] = []
    for step in failed[:3]:
        detail = _step_failure_detail(step.detail)
        parts.append(f"{step.name}: {detail}" if detail else step.name)
    if len(failed) > 3:
        parts.append(f"+{len(failed) - 3} more failed steps")
    return "; ".join(parts)


def _step_failure_detail(detail: dict[str, Any]) -> str:
    reason = detail.get("reason")
    if reason:
        return str(reason)
    baseline_failures = _regression_failures(detail.get("baseline_comparison", []))
    if baseline_failures:
        return "; ".join(baseline_failures)
    case_failures = _regression_failures(detail.get("cases", []))
    if case_failures:
        return "; ".join(case_failures)
    if detail.get("passed") is False:
        return "passed=false"
    return ""


def _regression_failures(items: Any) -> list[str]:
    failures: list[str] = []
    if not isinstance(items, list):
        return failures
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "case"))
        details = item.get("details", {})
        if not isinstance(details, dict):
            continue
        raw_failures = details.get("failures", [])
        if isinstance(raw_failures, list) and raw_failures:
            joined = ", ".join(str(value) for value in raw_failures[:3])
            failures.append(f"{name} {joined}")
            continue
        if item.get("passed") is False:
            expected_hits = details.get("expected_hits", {})
            missing_terms = []
            if isinstance(expected_hits, dict):
                missing_terms = [str(term) for term, hit in expected_hits.items() if not hit]
            if missing_terms:
                failures.append(f"{name} missing expected terms: {', '.join(missing_terms[:3])}")
            else:
                failures.append(f"{name} failed")
    return failures


def _detail_value(step: WorkerStep | None, key: str, default: Any) -> Any:
    if step is None:
        return default
    return step.detail.get(key, default)
