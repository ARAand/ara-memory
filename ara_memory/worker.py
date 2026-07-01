from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara_memory.lock import FileLock
from ara_memory.regression import load_recall_regression_baseline, load_recall_regression_cases


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
    doctor_query: str = "current memory state",
    recall_budget: int = 1600,
    hot_budget: int = 1200,
    regression_manifest: Path | None = None,
    regression_baseline: Path | None = None,
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
                doctor_query=doctor_query,
                recall_budget=recall_budget,
                hot_budget=hot_budget,
                regression_manifest=regression_manifest,
                regression_baseline=regression_baseline,
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
        doctor_query=doctor_query,
        recall_budget=recall_budget,
        hot_budget=hot_budget,
        regression_manifest=regression_manifest,
        regression_baseline=regression_baseline,
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
    doctor_query: str,
    recall_budget: int,
    hot_budget: int,
    regression_manifest: Path | None,
    regression_baseline: Path | None,
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

    review = memory.review_worker(scope=scope, limit=review_limit, dry_run=not apply_review)
    review_payload = _limit_items(review.as_dict(), "items", limit=report_item_limit)
    review_passed = not apply_review or review.changed >= 0
    steps.append(WorkerStep("review_worker", review_passed, review_payload))

    triage = memory.review_triage(scope=scope, limit=triage_limit, examples_per_group=min(3, report_item_limit))
    triage_payload = _limit_items(triage.as_dict(), "groups", limit=report_item_limit)
    steps.append(WorkerStep("review_triage", True, triage_payload))

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
        steps.append(WorkerStep("recall_regression", regression.passed, regression.as_dict()))

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
    doctor = by_name.get("doctor")
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
        "episode_summary": {
            "summaries_created": _detail_value(episode, "summaries_created", 0),
            "superseded": _detail_value(episode, "superseded", 0),
        },
        "doctor_passed": bool(doctor.detail.get("passed", False)) if doctor else None,
        "recall_regression_passed": bool(regression.detail.get("passed", False)) if regression else None,
        "maintenance_integrity": maintenance.detail.get("sqlite_integrity") if maintenance else None,
    }


def _worker_failure_reason(steps: list[WorkerStep]) -> str:
    failed = [step for step in steps if not step.passed]
    if not failed:
        return ""
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
