from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ara_memory.backup_stewardship import DEFAULT_TARGET_BACKUP_BYTES
from ara_memory.costs import estimate_api_cost
from ara_memory.core import AraMemory
from ara_memory.ingest import ingest_file
from ara_memory.models import EventKind
from ara_memory.regression import (
    load_recall_regression_baseline,
    load_recall_regression_cases,
    write_recall_regression_baseline,
)
from ara_memory.relation_merge import (
    RELATION_MERGE_CONFIRMATION,
    apply_relation_merge_approval,
    list_relation_merge_review_queue,
    prepare_relation_merge_approval,
    record_relation_merge_review_queue,
    review_relation_merge_witnesses,
    run_relation_merge_dry_run,
)
from ara_memory.reconsolidation import (
    RECONSOLIDATION_APPLY_CONFIRMATION,
    RECONSOLIDATION_LIVE_ACTION_ROLLBACK_CONFIRMATION,
    RECONSOLIDATION_LIVE_ROLLBACK_CONFIRMATION,
    RECONSOLIDATION_STRONG_ACTION_PREPARE_CONFIRMATION,
    STRONG_RECONSOLIDATION_ACTIONS,
    backfill_legacy_reconsolidation_rollback_witnesses,
    list_reconsolidation_review_queue,
    record_reconsolidation_review_queue,
)
from ara_memory.turn import execute_turn_ingress, plan_turn_ingress, remember_turn
from ara_memory.worktree import capture_worktree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ara-memory")
    parser.add_argument("--root", type=Path, default=None, help="Memory root directory. Defaults to ARA_MEMORY_HOME or .ara-memory.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    retain = sub.add_parser("retain")
    retain.add_argument("--kind", choices=[k.value for k in EventKind], required=True)
    retain.add_argument("--text", default=None)
    retain.add_argument("--file", type=Path, default=None)
    retain.add_argument("--source", default="manual")
    retain.add_argument("--scope", default="global")
    retain.add_argument("--metadata", default="{}")
    retain.add_argument("--allow-raw-secret", action="store_true")

    ingest = sub.add_parser("ingest-file")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--scope", default="global")
    ingest.add_argument("--source", default="file-ingest")
    ingest.add_argument("--caption", default="")
    ingest.add_argument("--max-text-chars", type=int, default=12000)
    ingest.add_argument("--consolidate", action="store_true")

    remember = sub.add_parser("remember-turn")
    remember.add_argument("--file", type=Path, default=None, help="JSON turn envelope. Defaults to stdin.")
    remember.add_argument("--scope", default="global")
    remember.add_argument("--source", default="codex-turn")
    remember.add_argument("--no-consolidate", action="store_true")
    remember.add_argument("--sleep", action="store_true")
    remember.add_argument("--hot-budget", type=int, default=1200)
    remember.add_argument("--capture-cwd", type=Path, default=None)
    remember.add_argument("--include-untracked-content", action="store_true")
    remember.add_argument("--max-file-chars", type=int, default=8000)
    remember.add_argument("--max-text-chars", type=int, default=12000)
    remember.add_argument("--allow-raw-secret", action="store_true")

    plan_turn = sub.add_parser("plan-turn")
    plan_turn.add_argument("--file", type=Path, default=None, help="JSON turn envelope. Defaults to stdin.")
    plan_turn.add_argument("--capture-cwd", type=Path, default=None)
    plan_turn.add_argument("--max-file-chars", type=int, default=8000)
    plan_turn.add_argument("--max-text-chars", type=int, default=12000)
    plan_turn.add_argument("--direct-text-threshold", type=int, default=16000)

    ingress_turn = sub.add_parser("ingress-turn")
    ingress_turn.add_argument("--file", type=Path, default=None, help="JSON turn envelope. Defaults to stdin.")
    ingress_turn.add_argument("--scope", default="global")
    ingress_turn.add_argument("--source", default="codex-turn")
    ingress_turn.add_argument("--mode", choices=["auto", "remember-turn", "spool-turn"], default="auto")
    ingress_turn.add_argument("--dry-run", action="store_true")
    ingress_turn.add_argument("--no-consolidate", action="store_true")
    ingress_turn.add_argument("--sleep", action="store_true")
    ingress_turn.add_argument("--hot-budget", type=int, default=1200)
    ingress_turn.add_argument("--capture-cwd", type=Path, default=None)
    ingress_turn.add_argument("--include-untracked-content", action="store_true")
    ingress_turn.add_argument("--max-file-chars", type=int, default=8000)
    ingress_turn.add_argument("--max-text-chars", type=int, default=12000)
    ingress_turn.add_argument("--direct-text-threshold", type=int, default=16000)
    ingress_turn.add_argument("--allow-raw-secret", action="store_true")

    spool_turn = sub.add_parser("spool-turn")
    spool_turn.add_argument("--file", type=Path, default=None, help="JSON turn envelope. Defaults to stdin.")
    spool_turn.add_argument("--scope", default="global")
    spool_turn.add_argument("--source", default="codex-turn")
    spool_turn.add_argument("--no-consolidate", action="store_true")
    spool_turn.add_argument("--sleep", action="store_true")
    spool_turn.add_argument("--hot-budget", type=int, default=1200)
    spool_turn.add_argument("--capture-cwd", type=Path, default=None)
    spool_turn.add_argument("--include-untracked-content", action="store_true")
    spool_turn.add_argument("--max-file-chars", type=int, default=8000)
    spool_turn.add_argument("--max-text-chars", type=int, default=12000)
    spool_turn.add_argument("--allow-raw-secret", action="store_true")

    drain_spool = sub.add_parser("drain-spool")
    drain_spool.add_argument("--scope", default=None)
    drain_spool.add_argument("--limit", type=int, default=25)
    drain_spool.add_argument("--stop-on-error", action="store_true")
    drain_spool.add_argument("--processing-stale-seconds", type=int, default=3600)
    drain_spool.add_argument("--stabilize", action="store_true", help="After draining, fold repeated episode/candidate noise and rebuild hot memory.")
    drain_spool.add_argument("--stabilization-scope", default=None, help="Scope to stabilize even when no item was drained.")
    drain_spool.add_argument("--stabilization-episode-min-group-size", type=int, default=5)
    drain_spool.add_argument("--stabilization-session-min-group-size", type=int, default=20)
    drain_spool.add_argument("--stabilization-candidate-min-group-size", type=int, default=3)
    drain_spool.add_argument("--stabilization-limit", type=int, default=80)

    sub.add_parser("spool-stats")

    consolidate = sub.add_parser("consolidate")
    consolidate.add_argument("--limit", type=int, default=100)

    capture = sub.add_parser("capture-worktree")
    capture.add_argument("--cwd", type=Path, default=Path.cwd())
    capture.add_argument("--scope", default="project")
    capture.add_argument("--consolidate", action="store_true")
    capture.add_argument("--include-untracked-content", action="store_true")
    capture.add_argument("--max-file-chars", type=int, default=8000)

    recall = sub.add_parser("recall")
    recall.add_argument("query")
    recall.add_argument("--scope", default="global")
    recall.add_argument("--budget", type=int, default=4000)
    recall.add_argument("--no-global", action="store_true", help="Do not include global memories when recalling a scoped pack.")
    recall.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Print recall diagnostics after the pack, including rendered capsules, "
            "visible query terms, visible sections, truncation, and fallback status."
        ),
    )
    recall.add_argument("--hot", action="store_true", help="Prepend the scope hot memory file if it exists.")

    recall_candidates = sub.add_parser(
        "recall-candidates",
        help="Select ranked recall capsules without rendering a full memory pack.",
    )
    recall_candidates.add_argument("query")
    recall_candidates.add_argument("--scope", default="global")
    recall_candidates.add_argument("--budget", type=int, default=1600)
    recall_candidates.add_argument("--limit", type=int, default=18)
    recall_candidates.add_argument("--no-global", action="store_true")
    recall_candidates.add_argument("--hot", action="store_true")
    recall_candidates.add_argument("--include-capsules", action="store_true")

    recall_plan = sub.add_parser(
        "recall-plan",
        help="Compare recall budgets and choose the smallest pack with visible evidence quality.",
    )
    recall_plan.add_argument("query")
    recall_plan.add_argument("--scope", default="global")
    recall_plan.add_argument(
        "--budgets",
        default="800,1600,2500",
        help="Comma-separated recall token budgets to test before selecting a pack.",
    )
    recall_plan.add_argument("--no-global", action="store_true")
    recall_plan.add_argument("--no-hot", action="store_true")
    recall_plan.add_argument("--output-tokens", type=int, default=0)
    recall_plan.add_argument("--input-usd-per-million", type=float, default=0.0)
    recall_plan.add_argument("--output-usd-per-million", type=float, default=0.0)
    recall_plan.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print full plan diagnostics, including quality_score, visible_capsules, "
            "query_terms_visible_count, and fallback_used."
        ),
    )

    recall_context = sub.add_parser(
        "recall-context",
        help="Build the selected recall pack using recall-plan.",
    )
    recall_context.add_argument("query")
    recall_context.add_argument("--scope", default="global")
    recall_context.add_argument(
        "--budgets",
        default="800,1600,2500",
        help="Comma-separated recall token budgets to test before selecting a pack.",
    )
    recall_context.add_argument("--no-global", action="store_true")
    recall_context.add_argument("--no-hot", action="store_true")
    recall_context.add_argument("--output-tokens", type=int, default=0)
    recall_context.add_argument("--input-usd-per-million", type=float, default=0.0)
    recall_context.add_argument("--output-usd-per-million", type=float, default=0.0)
    recall_context.add_argument("--pack-only", action="store_true", help="Print only the selected pack, hiding the plan.")
    recall_context.add_argument("--json", action="store_true")

    recall_policy = sub.add_parser(
        "recall-policy",
        help="Choose the purpose-aware recall path before rendering memory context.",
    )
    recall_policy.add_argument("query")
    recall_policy.add_argument("--scope", default="global")
    recall_policy.add_argument(
        "--budgets",
        default="800,1600,2500",
        help="Comma-separated active recall budgets to test before choosing an action.",
    )
    recall_policy.add_argument("--cold-budget", type=int, default=700)
    recall_policy.add_argument("--cold-group-limit", type=int, default=3)
    recall_policy.add_argument("--no-global", action="store_true")
    recall_policy.add_argument("--no-hot", action="store_true")
    recall_policy.add_argument("--output-tokens", type=int, default=0)
    recall_policy.add_argument("--input-usd-per-million", type=float, default=0.0)
    recall_policy.add_argument("--output-usd-per-million", type=float, default=0.0)
    recall_policy.add_argument("--json", action="store_true")

    recall_quality = sub.add_parser(
        "recall-quality",
        help="Gate a recall route with policy and working-memory impact feedback.",
    )
    recall_quality.add_argument("query")
    recall_quality.add_argument("--scope", default="global")
    recall_quality.add_argument(
        "--budgets",
        default="800,1600,2500",
        help="Comma-separated active recall budgets to test before choosing an action.",
    )
    recall_quality.add_argument("--working-budget", type=int, default=900)
    recall_quality.add_argument("--recall-budget", type=int, default=1600)
    recall_quality.add_argument("--limit", type=int, default=500)
    recall_quality.add_argument("--min-evaluated", type=int, default=3)
    recall_quality.add_argument("--no-global", action="store_true")
    recall_quality.add_argument("--no-hot", action="store_true")
    recall_quality.add_argument("--json", action="store_true")

    reconsolidation_frame = sub.add_parser(
        "reconsolidation-frame",
        help="Build a read-only purpose/identity/decision/failure frame for the current query.",
    )
    reconsolidation_frame.add_argument("query", nargs="?", default=None)
    reconsolidation_frame.add_argument("--scope", default="global")
    reconsolidation_frame.add_argument("--budgets", default="800,1600,2500")
    reconsolidation_frame.add_argument("--working-budget", type=int, default=900)
    reconsolidation_frame.add_argument("--recall-budget", type=int, default=1600)
    reconsolidation_frame.add_argument("--no-global", action="store_true")
    reconsolidation_frame.add_argument("--no-hot", action="store_true")
    reconsolidation_frame.add_argument("--json", action="store_true")

    reconsolidation_prepare = sub.add_parser(
        "reconsolidation-prepare",
        help="Prepare a short-lived approval token for a reviewed reconsolidation frame snapshot.",
    )
    reconsolidation_prepare.add_argument("query", nargs="?", default=None)
    reconsolidation_prepare.add_argument("--scope", default="global")
    reconsolidation_prepare.add_argument("--budgets", default="800,1600,2500")
    reconsolidation_prepare.add_argument("--working-budget", type=int, default=900)
    reconsolidation_prepare.add_argument("--recall-budget", type=int, default=1600)
    reconsolidation_prepare.add_argument("--no-global", action="store_true")
    reconsolidation_prepare.add_argument("--no-hot", action="store_true")
    reconsolidation_prepare.add_argument("--ttl-minutes", type=int, default=60)
    reconsolidation_prepare.add_argument("--json", action="store_true")
    reconsolidation_apply = sub.add_parser(
        "reconsolidation-apply",
        help=(
            "Apply a prepared reconsolidation frame once as a candidate summary. "
            f"Requires exact confirmation: {RECONSOLIDATION_APPLY_CONFIRMATION!r}."
        ),
    )
    reconsolidation_apply.add_argument("--approval-token", required=True)
    reconsolidation_apply.add_argument("--confirm", required=True)
    reconsolidation_apply.add_argument("--json", action="store_true")
    reconsolidation_review = sub.add_parser(
        "reconsolidation-review",
        help="Review applied reconsolidation witnesses before stronger memory mutation is trusted.",
    )
    reconsolidation_review.add_argument("--scope", default=None)
    reconsolidation_review.add_argument("--approval-id", default=None)
    reconsolidation_review.add_argument("--limit", type=int, default=50)
    reconsolidation_review.add_argument("--regression-manifest", type=Path, default=None)
    reconsolidation_review.add_argument("--regression-baseline", type=Path, default=None)
    reconsolidation_review.add_argument("--record-queue", action="store_true")
    reconsolidation_review.add_argument("--json", action="store_true")
    reconsolidation_review_queue = sub.add_parser(
        "reconsolidation-review-queue",
        help="List persisted reconsolidation witness risks that block or warn stronger memory mutation.",
    )
    reconsolidation_review_queue.add_argument("--scope", default=None)
    reconsolidation_review_queue.add_argument("--status", default="open", choices=["open", "resolved"])
    reconsolidation_review_queue.add_argument("--limit", type=int, default=50)
    reconsolidation_review_queue.add_argument("--json", action="store_true")
    reconsolidation_backfill = sub.add_parser(
        "reconsolidation-backfill-rollback-witnesses",
        help="Review and optionally backfill legacy reconsolidation approvals that predate rollback witness previews.",
    )
    reconsolidation_backfill.add_argument("--scope", default=None)
    reconsolidation_backfill.add_argument("--approval-id", default=None)
    reconsolidation_backfill.add_argument("--limit", type=int, default=50)
    reconsolidation_backfill.add_argument("--apply", action="store_true")
    reconsolidation_backfill.add_argument("--json", action="store_true")
    reconsolidation_strong_preflight = sub.add_parser(
        "reconsolidation-strong-preflight",
        help="Run prepare/apply/review/recall-regression inside a restored backup shadow store before stronger reconsolidation design.",
    )
    reconsolidation_strong_preflight.add_argument("query", nargs="?", default=None)
    reconsolidation_strong_preflight.add_argument("--backup", type=Path, required=True)
    reconsolidation_strong_preflight.add_argument("--scope", default="global")
    reconsolidation_strong_preflight.add_argument("--budgets", default="800,1600,2500")
    reconsolidation_strong_preflight.add_argument("--working-budget", type=int, default=900)
    reconsolidation_strong_preflight.add_argument("--recall-budget", type=int, default=1600)
    reconsolidation_strong_preflight.add_argument("--no-global", action="store_true")
    reconsolidation_strong_preflight.add_argument("--no-hot", action="store_true")
    reconsolidation_strong_preflight.add_argument(
        "--action",
        choices=STRONG_RECONSOLIDATION_ACTIONS,
        action="append",
        default=[],
        help="Strong live capability to evaluate. Repeat for multiple actions. Defaults to all actions.",
    )
    reconsolidation_strong_preflight.add_argument("--regression-manifest", type=Path, default=None)
    reconsolidation_strong_preflight.add_argument("--regression-baseline", type=Path, default=None)
    reconsolidation_strong_preflight.add_argument("--json", action="store_true")
    prepare_live_recon_action = sub.add_parser(
        "prepare-live-reconsolidation-action",
        help=(
            "Prepare a short-lived one-use approval record for one design-ready "
            "strong reconsolidation action without mutating live memory."
        ),
    )
    prepare_live_recon_action.add_argument("query", nargs="?", default=None)
    prepare_live_recon_action.add_argument("--backup", type=Path, required=True)
    prepare_live_recon_action.add_argument("--action", choices=STRONG_RECONSOLIDATION_ACTIONS, required=True)
    prepare_live_recon_action.add_argument("--scope", default="global")
    prepare_live_recon_action.add_argument("--budgets", default="800,1600,2500")
    prepare_live_recon_action.add_argument("--working-budget", type=int, default=900)
    prepare_live_recon_action.add_argument("--recall-budget", type=int, default=1600)
    prepare_live_recon_action.add_argument("--ttl-minutes", type=int, default=30)
    prepare_live_recon_action.add_argument(
        "--confirm",
        required=True,
        help=f"Must exactly match: {RECONSOLIDATION_STRONG_ACTION_PREPARE_CONFIRMATION}",
    )
    prepare_live_recon_action.add_argument("--no-global", action="store_true")
    prepare_live_recon_action.add_argument("--no-hot", action="store_true")
    prepare_live_recon_action.add_argument("--regression-manifest", type=Path, default=None)
    prepare_live_recon_action.add_argument("--regression-baseline", type=Path, default=None)
    prepare_live_recon_action.add_argument("--json", action="store_true")
    live_recon_cool = sub.add_parser(
        "live-reconsolidation-cool",
        help="Consume one prepared cool action approval and supersede one non-core active capsule.",
    )
    live_recon_cool.add_argument("--approval-token", required=True)
    live_recon_cool.add_argument("--capsule-id", required=True)
    live_recon_cool.add_argument("--confirm", required=True)
    live_recon_cool.add_argument("--reason", default="approved strong reconsolidation cool action")
    live_recon_cool.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation cool",
    )
    live_recon_cool.add_argument("--recall-budget", type=int, default=1200)
    live_recon_cool.add_argument("--hot-budget", type=int, default=900)
    live_recon_cool.add_argument("--no-global", action="store_true")
    live_recon_cool.add_argument("--json", action="store_true")
    recon_action_witnesses = sub.add_parser(
        "reconsolidation-action-witnesses",
        help="Review live reconsolidation action witnesses against their allowed field-transition policy.",
    )
    recon_action_witnesses.add_argument("--scope", default=None)
    recon_action_witnesses.add_argument("--action", choices=STRONG_RECONSOLIDATION_ACTIONS, default=None)
    recon_action_witnesses.add_argument("--witness-id", default=None)
    recon_action_witnesses.add_argument("--limit", type=int, default=50)
    recon_action_witnesses.add_argument("--json", action="store_true")
    recon_action_shadow_rollback = sub.add_parser(
        "reconsolidation-action-shadow-rollback",
        help="Restore a backup into a temporary shadow store and prove live action witness rollback without touching live memory.",
    )
    recon_action_shadow_rollback.add_argument("--backup", type=Path, required=True)
    recon_action_shadow_rollback.add_argument("--scope", default=None)
    recon_action_shadow_rollback.add_argument("--action", choices=STRONG_RECONSOLIDATION_ACTIONS, default=None)
    recon_action_shadow_rollback.add_argument("--witness-id", default=None)
    recon_action_shadow_rollback.add_argument("--limit", type=int, default=20)
    recon_action_shadow_rollback.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation action rollback",
    )
    recon_action_shadow_rollback.add_argument("--recall-budget", type=int, default=1200)
    recon_action_shadow_rollback.add_argument("--hot-budget", type=int, default=900)
    recon_action_shadow_rollback.add_argument("--no-global", action="store_true")
    recon_action_shadow_rollback.add_argument("--json", action="store_true")
    prepare_live_recon_action_rollback = sub.add_parser(
        "prepare-live-reconsolidation-action-rollback",
        help=(
            "Prepare a short-lived one-use approval record for future live rollback "
            "of one reviewed reconsolidation action witness without mutating live memory."
        ),
    )
    prepare_live_recon_action_rollback.add_argument("--backup", type=Path, required=True)
    prepare_live_recon_action_rollback.add_argument("--witness-id", required=True)
    prepare_live_recon_action_rollback.add_argument("--scope", default=None)
    prepare_live_recon_action_rollback.add_argument("--action", choices=STRONG_RECONSOLIDATION_ACTIONS, default="cool")
    prepare_live_recon_action_rollback.add_argument("--ttl-minutes", type=int, default=30)
    prepare_live_recon_action_rollback.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation action rollback",
    )
    prepare_live_recon_action_rollback.add_argument("--recall-budget", type=int, default=1200)
    prepare_live_recon_action_rollback.add_argument("--hot-budget", type=int, default=900)
    prepare_live_recon_action_rollback.add_argument("--no-global", action="store_true")
    prepare_live_recon_action_rollback.add_argument("--json", action="store_true")
    recon_action_rollback_approvals = sub.add_parser(
        "reconsolidation-action-rollback-approvals",
        help=(
            "Review prepared reconsolidation action rollback approvals without consuming tokens "
            "or mutating live memory."
        ),
    )
    recon_action_rollback_approvals.add_argument("--scope", default=None)
    recon_action_rollback_approvals.add_argument("--action", choices=STRONG_RECONSOLIDATION_ACTIONS, default=None)
    recon_action_rollback_approvals.add_argument("--approval-id", default=None)
    recon_action_rollback_approvals.add_argument("--witness-id", default=None)
    recon_action_rollback_approvals.add_argument("--prepared-only", action="store_true")
    recon_action_rollback_approvals.add_argument("--limit", type=int, default=50)
    recon_action_rollback_approvals.add_argument("--json", action="store_true")
    live_recon_action_rollback = sub.add_parser(
        "live-reconsolidation-action-rollback",
        help="Consume one prepared action rollback approval and restore the exact recorded cool status transition.",
    )
    live_recon_action_rollback.add_argument("--approval-token", required=True)
    live_recon_action_rollback.add_argument(
        "--confirm",
        required=True,
        help=f"Must exactly match: {RECONSOLIDATION_LIVE_ACTION_ROLLBACK_CONFIRMATION}",
    )
    live_recon_action_rollback.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation action rollback",
    )
    live_recon_action_rollback.add_argument("--recall-budget", type=int, default=1200)
    live_recon_action_rollback.add_argument("--hot-budget", type=int, default=900)
    live_recon_action_rollback.add_argument("--no-global", action="store_true")
    live_recon_action_rollback.add_argument("--json", action="store_true")
    reconsolidation_shadow_rollback = sub.add_parser(
        "reconsolidation-shadow-rollback",
        help="Restore a backup into a temporary shadow store and prove reconsolidation candidate rollback without touching live memory.",
    )
    reconsolidation_shadow_rollback.add_argument("--backup", type=Path, required=True)
    reconsolidation_shadow_rollback.add_argument("--scope", default=None)
    reconsolidation_shadow_rollback.add_argument("--approval-id", default=None)
    reconsolidation_shadow_rollback.add_argument("--witness-id", default=None)
    reconsolidation_shadow_rollback.add_argument("--limit", type=int, default=20)
    reconsolidation_shadow_rollback.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation rollback",
    )
    reconsolidation_shadow_rollback.add_argument("--recall-budget", type=int, default=1200)
    reconsolidation_shadow_rollback.add_argument("--hot-budget", type=int, default=900)
    reconsolidation_shadow_rollback.add_argument("--no-global", action="store_true")
    reconsolidation_shadow_rollback.add_argument("--json", action="store_true")
    prepare_live_recon_rollback = sub.add_parser(
        "prepare-live-reconsolidation-rollback",
        help="Prepare a short-lived one-use token for live rollback of one reconsolidation candidate frame.",
    )
    prepare_live_recon_rollback.add_argument("--backup", type=Path, required=True)
    prepare_live_recon_rollback.add_argument("--witness-id", required=True)
    prepare_live_recon_rollback.add_argument("--scope", default=None)
    prepare_live_recon_rollback.add_argument("--ttl-minutes", type=int, default=30)
    prepare_live_recon_rollback.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation rollback",
    )
    prepare_live_recon_rollback.add_argument("--recall-budget", type=int, default=1200)
    prepare_live_recon_rollback.add_argument("--hot-budget", type=int, default=900)
    prepare_live_recon_rollback.add_argument("--no-global", action="store_true")
    prepare_live_recon_rollback.add_argument("--json", action="store_true")
    live_recon_rollback = sub.add_parser(
        "live-reconsolidation-rollback",
        help=(
            "Use a prepared one-use token to reject one reconsolidation candidate frame. "
            f"Requires exact confirmation: {RECONSOLIDATION_LIVE_ROLLBACK_CONFIRMATION!r}."
        ),
    )
    live_recon_rollback.add_argument("--approval-token", required=True)
    live_recon_rollback.add_argument("--confirm", required=True)
    live_recon_rollback.add_argument(
        "--doctor-query",
        default="current memory state after reconsolidation rollback",
    )
    live_recon_rollback.add_argument("--recall-budget", type=int, default=1200)
    live_recon_rollback.add_argument("--hot-budget", type=int, default=900)
    live_recon_rollback.add_argument("--no-global", action="store_true")
    live_recon_rollback.add_argument("--json", action="store_true")
    recon_exception = sub.add_parser(
        "reconsolidation-exception-witness",
        help="Record a reviewed exception for an exact reconsolidation capsule digest transition.",
    )
    recon_exception.add_argument("--witness-id", required=True)
    recon_exception.add_argument("--capsule-id", default=None)
    recon_exception.add_argument(
        "--field",
        required=True,
        choices=["title_digest", "body_digest", "source_event_ids"],
    )
    recon_exception.add_argument("--reason", required=True)
    recon_exception.add_argument("--evidence", default="{}")
    recon_exception.add_argument("--json", action="store_true")
    recon_exceptions = sub.add_parser(
        "reconsolidation-exception-witnesses",
        help="List active reconsolidation exception witnesses.",
    )
    recon_exceptions.add_argument("--scope", default=None)
    recon_exceptions.add_argument("--witness-id", default=None)
    recon_exceptions.add_argument("--status", default="active", choices=["active", "revoked"])
    recon_exceptions.add_argument("--limit", type=int, default=50)
    recon_exceptions.add_argument("--json", action="store_true")

    recall_policy_impact = sub.add_parser(
        "recall-policy-impact",
        help="Record whether a recall-policy route helped a real outcome.",
    )
    recall_policy_impact.add_argument("--scope", default="global")
    recall_policy_impact.add_argument("--query", required=True)
    recall_policy_impact.add_argument("--intent", required=True)
    recall_policy_impact.add_argument("--strategy", required=True)
    recall_policy_impact.add_argument("--action-name", action="append", required=True)
    recall_policy_impact.add_argument("--outcome", required=True)
    recall_policy_impact.add_argument("--helped", choices=["true", "false", "unknown"], default="unknown")
    recall_policy_impact.add_argument("--json", action="store_true")

    recall_policy_eval = sub.add_parser(
        "recall-policy-eval",
        help="Evaluate recorded recall-policy outcomes by intent and action.",
    )
    recall_policy_eval.add_argument("--scope", default="global")
    recall_policy_eval.add_argument("--limit", type=int, default=500)
    recall_policy_eval.add_argument("--min-evaluated", type=int, default=3)
    recall_policy_eval.add_argument("--no-global", action="store_true")
    recall_policy_eval.add_argument("--json", action="store_true")

    working_memory = sub.add_parser(
        "working-memory",
        help="Build a tiny cue-led working memory pack for the current situation.",
    )
    working_memory.add_argument("prompt", nargs="?", default=None)
    working_memory.add_argument("--scope", default="global")
    working_memory.add_argument("--active-file", action="append", default=[])
    working_memory.add_argument("--command-error", action="append", default=[])
    working_memory.add_argument("--budget", type=int, default=900)
    working_memory.add_argument("--recall-budget", type=int, default=1600)
    working_memory.add_argument("--no-global", action="store_true")
    working_memory.add_argument("--no-hot", action="store_true")
    working_memory.add_argument("--json", action="store_true")

    memory_impact = sub.add_parser(
        "working-memory-impact",
        help="Record which working-memory capsules influenced a turn.",
    )
    memory_impact.add_argument("--scope", default="global")
    memory_impact.add_argument("--cue", required=True)
    memory_impact.add_argument("--capsule-id", action="append", default=[])
    memory_impact.add_argument("--outcome", required=True)
    memory_impact.add_argument("--helped", choices=["true", "false", "unknown"], default="unknown")
    memory_impact.add_argument("--json", action="store_true")

    memory_impact_eval = sub.add_parser(
        "working-memory-impact-eval",
        help="Evaluate recorded working-memory impact outcomes by capsule and source.",
    )
    memory_impact_eval.add_argument("--scope", default="global")
    memory_impact_eval.add_argument("--limit", type=int, default=500)
    memory_impact_eval.add_argument("--min-evaluated", type=int, default=3)
    memory_impact_eval.add_argument("--no-global", action="store_true")
    memory_impact_eval.add_argument("--json", action="store_true")

    govern_turn = sub.add_parser(
        "govern-turn",
        help="Plan capture and recall actions for a turn without storing anything.",
    )
    govern_turn.add_argument("--file", type=Path, default=None, help="JSON turn envelope. Defaults to stdin.")
    govern_turn.add_argument("--scope", default="global")
    govern_turn.add_argument("--capture-cwd", type=Path, default=None)
    govern_turn.add_argument("--budgets", default="")
    govern_turn.add_argument("--working-budget", type=int, default=0)
    govern_turn.add_argument(
        "--budget-profile",
        choices=["auto", "quick", "standard", "deep", "debug", "mutation"],
        default="auto",
    )
    govern_turn.add_argument("--no-global", action="store_true")
    govern_turn.add_argument("--no-hot", action="store_true")
    govern_turn.add_argument("--direct-text-threshold", type=int, default=16000)
    govern_turn.add_argument("--max-file-chars", type=int, default=8000)
    govern_turn.add_argument("--max-text-chars", type=int, default=12000)
    govern_turn.add_argument("--output-tokens", type=int, default=0)
    govern_turn.add_argument("--input-usd-per-million", type=float, default=0.0)
    govern_turn.add_argument("--output-usd-per-million", type=float, default=0.0)
    govern_turn.add_argument("--max-input-tokens", type=int, default=0)
    govern_turn.add_argument("--max-model-cost-usd", type=float, default=0.0)
    govern_turn.add_argument(
        "--record-working-impact",
        action="store_true",
        help="Record projected working-memory capsule ids as bounded impact feedback after reviewing the turn outcome.",
    )
    govern_turn.add_argument("--impact-outcome", default="")
    govern_turn.add_argument("--impact-helped", choices=["true", "false", "unknown"], default="unknown")
    govern_turn.add_argument("--json", action="store_true")

    agency_review = sub.add_parser(
        "agency-review",
        help="Review a requested action against purpose, identity, recall policy, and working memory.",
    )
    agency_review.add_argument("prompt", nargs="?", default=None)
    agency_review.add_argument("--scope", default="global")
    agency_review.add_argument("--proposed-action", default="")
    agency_review.add_argument("--constraint", action="append", default=[])
    agency_review.add_argument("--active-file", action="append", default=[])
    agency_review.add_argument("--command-error", action="append", default=[])
    agency_review.add_argument(
        "--budgets",
        default="800,1600",
        help="Comma-separated active recall budgets for the underlying recall-policy check.",
    )
    agency_review.add_argument("--working-budget", type=int, default=700)
    agency_review.add_argument("--recall-budget", type=int, default=1200)
    agency_review.add_argument("--no-global", action="store_true")
    agency_review.add_argument("--no-hot", action="store_true")
    agency_review.add_argument("--health", action="store_true")
    agency_review.add_argument("--record", action="store_true")
    agency_review.add_argument(
        "--strict-action-exit",
        action="store_true",
        help="Exit non-zero unless the review explicitly allows proceeding.",
    )
    agency_review.add_argument("--json", action="store_true")

    purpose_check = sub.add_parser("purpose-check")
    purpose_check.add_argument("--scope", default="global")
    purpose_check.add_argument("--query", default="purpose of Ara natural memory and long-running goal")
    purpose_check.add_argument("--budget", type=int, default=1000)
    purpose_check.add_argument("--hot-budget", type=int, default=1200)
    purpose_check.add_argument("--no-global", action="store_true")
    purpose_check.add_argument("--repair-hot", action="store_true")
    purpose_check.add_argument("--json", action="store_true")

    identity_check = sub.add_parser("identity-check")
    identity_check.add_argument("--scope", default="global")
    identity_check.add_argument("--query", default="Ara identity judgment principles free will coding partner")
    identity_check.add_argument("--budget", type=int, default=1000)
    identity_check.add_argument("--hot-budget", type=int, default=1200)
    identity_check.add_argument("--no-global", action="store_true")
    identity_check.add_argument("--repair-hot", action="store_true")
    identity_check.add_argument("--json", action="store_true")

    milestone_check = sub.add_parser("milestone-check")
    milestone_check.add_argument("--scope", default="global")
    milestone_check.add_argument("--query", default="current Ara memory architecture and purpose")
    milestone_check.add_argument("--recall-budgets", default="800,1600,2500")
    milestone_check.add_argument("--health-query", default="current memory health")
    milestone_check.add_argument("--purpose-query", default="purpose of Ara natural memory and long-running goal")
    milestone_check.add_argument("--recall-budget", type=int, default=1600)
    milestone_check.add_argument("--hot-budget", type=int, default=1200)
    milestone_check.add_argument("--candidate-ratio-limit", type=float, default=0.5)
    milestone_check.add_argument("--no-repair-hot", action="store_true")
    milestone_check.add_argument("--regression-manifest", type=Path, default=None)
    milestone_check.add_argument("--regression-baseline", type=Path, default=None)
    milestone_check.add_argument("--json", action="store_true")

    failure_kind_audit = sub.add_parser("failure-kind-audit")
    failure_kind_audit.add_argument("--scope", default=None)
    failure_kind_audit.add_argument("--status", action="append", choices=["candidate", "stable", "quarantined", "superseded", "rejected"], default=None)
    failure_kind_audit.add_argument("--limit", type=int, default=200)
    failure_kind_audit.add_argument("--apply", action="store_true")
    failure_kind_audit.add_argument("--json", action="store_true")

    self_kind_audit = sub.add_parser("self-kind-audit")
    self_kind_audit.add_argument("--scope", default=None)
    self_kind_audit.add_argument("--status", action="append", choices=["candidate", "stable", "quarantined", "superseded", "rejected"], default=None)
    self_kind_audit.add_argument("--limit", type=int, default=200)
    self_kind_audit.add_argument("--apply", action="store_true")
    self_kind_audit.add_argument("--json", action="store_true")

    goal_roadmap = sub.add_parser("goal-roadmap")
    goal_roadmap.add_argument("--scope", default="global")
    goal_roadmap.add_argument("--regression-manifest", type=Path, default=None)
    goal_roadmap.add_argument("--regression-baseline", type=Path, default=None)
    goal_roadmap.add_argument(
        "--repair-hot",
        action="store_true",
        help="Rebuild hot memory if purpose/identity checks need it.",
    )
    goal_roadmap.add_argument("--json", action="store_true")

    graph_readiness = sub.add_parser("graph-readiness")
    graph_readiness.add_argument("--scope", default="global")
    graph_readiness.add_argument("--query", default=None)
    graph_readiness.add_argument("--budgets", default="800,1600")
    graph_readiness.add_argument("--no-global", action="store_true")
    graph_readiness.add_argument("--json", action="store_true")
    global_spreading = sub.add_parser("global-spreading-sandbox")
    global_spreading.add_argument("--scope", default="global")
    global_spreading.add_argument("--query", action="append", default=None)
    global_spreading.add_argument("--budgets", default="800,1600")
    global_spreading.add_argument("--max-edges", type=int, default=120)
    global_spreading.add_argument("--max-supplemented", type=int, default=8)
    global_spreading.add_argument("--max-depth", type=int, default=2)
    global_spreading.add_argument("--min-quality", type=int, default=35)
    global_spreading.add_argument("--json", action="store_true")
    privacy_pre_push = sub.add_parser("privacy-pre-push")
    privacy_pre_push.add_argument("--repo", type=Path, default=Path("."))
    privacy_pre_push.add_argument("--include-untracked", action="store_true")
    privacy_pre_push.add_argument("--include-direct-identifiers", action="store_true")
    privacy_pre_push.add_argument("--max-bytes", type=int, default=1_000_000)
    privacy_pre_push.add_argument("--json", action="store_true")
    relation_merge = sub.add_parser(
        "relation-merge",
        help="Dry-run semantic relation-node merge candidates without mutating the relation graph.",
    )
    relation_merge.add_argument("--scope", default="global")
    relation_merge.add_argument("--limit", type=int, default=20)
    relation_merge.add_argument("--threshold", type=float, default=0.72)
    relation_merge.add_argument("--node-limit", type=int, default=800)
    relation_merge.add_argument("--include-global", action="store_true")
    relation_merge.add_argument("--json", action="store_true")
    relation_merge_prepare = sub.add_parser(
        "relation-merge-prepare",
        help="Prepare a short-lived approval token and rollback witness for reviewed relation-node merge candidates.",
    )
    relation_merge_prepare.add_argument("--scope", default="global")
    relation_merge_prepare.add_argument("--limit", type=int, default=20)
    relation_merge_prepare.add_argument("--threshold", type=float, default=0.72)
    relation_merge_prepare.add_argument("--node-limit", type=int, default=800)
    relation_merge_prepare.add_argument("--include-global", action="store_true")
    relation_merge_prepare.add_argument("--ttl-minutes", type=int, default=60)
    relation_merge_prepare.add_argument("--json", action="store_true")
    relation_merge_apply = sub.add_parser(
        "relation-merge-apply",
        help=(
            "Apply a prepared relation-node merge once. "
            f"Requires exact confirmation: {RELATION_MERGE_CONFIRMATION!r}."
        ),
    )
    relation_merge_apply.add_argument("--approval-token", required=True)
    relation_merge_apply.add_argument("--confirm", required=True)
    relation_merge_apply.add_argument("--json", action="store_true")
    relation_merge_review = sub.add_parser(
        "relation-merge-review",
        help="Review applied relation-node merge witnesses for non-destructive recall-impact safety signals.",
    )
    relation_merge_review.add_argument("--scope", default=None)
    relation_merge_review.add_argument("--approval-id", default=None)
    relation_merge_review.add_argument("--limit", type=int, default=50)
    relation_merge_review.add_argument("--regression-manifest", type=Path, default=None)
    relation_merge_review.add_argument("--regression-baseline", type=Path, default=None)
    relation_merge_review.add_argument("--record-queue", action="store_true")
    relation_merge_review.add_argument("--json", action="store_true")
    relation_merge_review_queue = sub.add_parser(
        "relation-merge-review-queue",
        help="List persistent relation-merge review queue items created from witness/regression review gates.",
    )
    relation_merge_review_queue.add_argument("--scope", default=None)
    relation_merge_review_queue.add_argument("--status", default="open", choices=["open", "resolved"])
    relation_merge_review_queue.add_argument("--limit", type=int, default=50)
    relation_merge_review_queue.add_argument("--json", action="store_true")

    promote = sub.add_parser("promote")
    promote.add_argument("capsule_id")
    promote.add_argument("--reason", default="")
    promote.add_argument("--actor", default="manual")

    reject = sub.add_parser("reject")
    reject.add_argument("capsule_id")
    reject.add_argument("--reason", default="")
    reject.add_argument("--actor", default="manual")

    quarantine = sub.add_parser("quarantine")
    quarantine.add_argument("capsule_id")
    quarantine.add_argument("--reason", default="")
    quarantine.add_argument("--actor", default="manual")

    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--scope", default=None)
    list_cmd.add_argument("--status", choices=["candidate", "stable", "quarantined", "superseded", "rejected"], default=None)
    list_cmd.add_argument("--kind", default=None)
    list_cmd.add_argument("--limit", type=int, default=30)

    actions = sub.add_parser("actions")
    actions.add_argument("--limit", type=int, default=30)

    sleep = sub.add_parser("sleep")
    sleep.add_argument("--scope", default="global")
    sleep.add_argument("--dry-run", action="store_true")

    sleep_runs = sub.add_parser("sleep-runs")
    sleep_runs.add_argument("--limit", type=int, default=20)

    hot = sub.add_parser("hot")
    hot.add_argument("--scope", default="global")
    hot.add_argument("--budget", type=int, default=1200)

    show_hot = sub.add_parser("show-hot")
    show_hot.add_argument("--scope", default="global")

    cost = sub.add_parser("estimate-cost")
    cost.add_argument("query")
    cost.add_argument("--scope", default="global")
    cost.add_argument("--budget", type=int, default=4000)
    cost.add_argument("--no-global", action="store_true")
    cost.add_argument("--output-tokens", type=int, default=0)
    cost.add_argument("--input-usd-per-million", type=float, default=0.0)
    cost.add_argument("--output-usd-per-million", type=float, default=0.0)

    sub.add_parser("audit")
    risk = sub.add_parser("risk")
    risk.add_argument("--scope", default="global")
    risk.add_argument("--limit", type=int, default=200)
    review = sub.add_parser("review")
    review.add_argument("--scope", default="global")
    review.add_argument("--limit", type=int, default=50)
    quality = sub.add_parser("quality")
    quality.add_argument("--scope", default=None)
    quality.add_argument("--limit", type=int, default=500)
    quality.add_argument("--persist", action="store_true")
    quality.add_argument("--json", action="store_true")
    promotion_candidates = sub.add_parser("promotion-candidates")
    promotion_candidates.add_argument("--scope", default=None)
    promotion_candidates.add_argument("--limit", type=int, default=500)
    promotion_candidates.add_argument("--threshold", type=float, default=None)
    promotion_candidates.add_argument("--json", action="store_true")
    review_queue = sub.add_parser("review-queue")
    review_queue.add_argument("--scope", default=None)
    review_queue.add_argument("--status", default="open")
    review_queue.add_argument("--limit", type=int, default=50)
    review_triage = sub.add_parser("review-triage")
    review_triage.add_argument("--scope", default=None)
    review_triage.add_argument("--status", default="open")
    review_triage.add_argument("--limit", type=int, default=500)
    review_triage.add_argument("--examples-per-group", type=int, default=3)
    review_triage.add_argument("--json", action="store_true")
    resolve_review = sub.add_parser("resolve-review")
    resolve_review.add_argument("queue_id")
    review_witnesses = sub.add_parser("review-witnesses")
    review_witnesses.add_argument("--scope", default=None)
    review_witnesses.add_argument("--action", default=None)
    review_witnesses.add_argument("--limit", type=int, default=50)
    review_witnesses.add_argument("--json", action="store_true")
    prepare_review_rollback = sub.add_parser("prepare-review-rollback")
    prepare_review_rollback.add_argument("--witness-id", required=True)
    prepare_review_rollback.add_argument("--ttl-minutes", type=int, default=30)
    prepare_review_rollback.add_argument("--json", action="store_true")
    live_review_rollback = sub.add_parser("live-review-rollback")
    live_review_rollback.add_argument("--approval-token", required=True)
    live_review_rollback.add_argument("--confirm", required=True)
    live_review_rollback.add_argument("--json", action="store_true")
    mutation_preflight = sub.add_parser("mutation-preflight")
    mutation_preflight.add_argument("--capsule-id", required=True)
    mutation_preflight.add_argument("--action", required=True, choices=["rewrite", "delete"])
    mutation_preflight.add_argument("--title", default=None)
    mutation_preflight.add_argument("--body", default=None)
    mutation_preflight.add_argument("--tag", action="append", default=None)
    mutation_preflight.add_argument("--json", action="store_true")
    prepare_mutation = sub.add_parser("prepare-mutation")
    prepare_mutation.add_argument("--capsule-id", required=True)
    prepare_mutation.add_argument("--action", required=True, choices=["rewrite", "delete"])
    prepare_mutation.add_argument("--title", default=None)
    prepare_mutation.add_argument("--body", default=None)
    prepare_mutation.add_argument("--tag", action="append", default=None)
    prepare_mutation.add_argument("--ttl-minutes", type=int, default=30)
    prepare_mutation.add_argument("--json", action="store_true")
    live_mutation_apply = sub.add_parser("live-mutation-apply")
    live_mutation_apply.add_argument("--approval-token", required=True)
    live_mutation_apply.add_argument("--confirm", required=True)
    live_mutation_apply.add_argument("--json", action="store_true")
    prepare_mutation_rollback = sub.add_parser("prepare-mutation-rollback")
    prepare_mutation_rollback.add_argument("--witness-id", required=True)
    prepare_mutation_rollback.add_argument("--ttl-minutes", type=int, default=30)
    prepare_mutation_rollback.add_argument("--json", action="store_true")
    live_mutation_rollback = sub.add_parser("live-mutation-rollback")
    live_mutation_rollback.add_argument("--approval-token", required=True)
    live_mutation_rollback.add_argument("--confirm", required=True)
    live_mutation_rollback.add_argument("--json", action="store_true")
    review_worker = sub.add_parser("review-worker")
    review_worker.add_argument("--scope", default=None)
    review_worker.add_argument("--limit", type=int, default=25)
    review_worker.add_argument("--apply", action="store_true")
    review_worker.add_argument("--json", action="store_true")
    review_compact = sub.add_parser("review-compact")
    review_compact.add_argument("--scope", default=None)
    review_compact.add_argument("--limit", type=int, default=250)
    review_compact.add_argument("--apply", action="store_true")
    review_compact.add_argument("--compact", action="store_true")
    review_compact.add_argument("--json", action="store_true")
    review_redact = sub.add_parser("review-redact")
    review_redact.add_argument("--scope", default=None)
    review_redact.add_argument("--limit", type=int, default=50)
    review_redact.add_argument("--apply", action="store_true")
    review_redact.add_argument("--json", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("--scope", default="global")
    worker.add_argument("--spool-limit", type=int, default=25)
    worker.add_argument("--processing-stale-seconds", type=int, default=3600)
    worker.add_argument("--quality-limit", type=int, default=500)
    worker.add_argument("--review-limit", type=int, default=25)
    worker.add_argument("--triage-limit", type=int, default=500)
    worker.add_argument("--apply-review", action="store_true")
    worker.add_argument("--no-episode-summary", action="store_true")
    worker.add_argument("--episode-summary-min-group-size", type=int, default=5)
    worker.add_argument("--episode-summary-session-min-group-size", type=int, default=20)
    worker.add_argument("--episode-summary-limit", type=int, default=50)
    worker.add_argument("--no-candidate-summary", action="store_true")
    worker.add_argument("--candidate-summary-min-group-size", type=int, default=3)
    worker.add_argument("--candidate-summary-limit", type=int, default=80)
    worker.add_argument("--no-governance-probe", action="store_true")
    worker.add_argument("--governance-query", default="")
    worker.add_argument("--governance-max-input-tokens", type=int, default=0)
    worker.add_argument("--governance-max-model-cost-usd", type=float, default=0.0)
    worker.add_argument("--doctor-query", default="current memory state")
    worker.add_argument("--recall-budget", type=int, default=1600)
    worker.add_argument("--hot-budget", type=int, default=1200)
    worker.add_argument("--regression-manifest", type=Path, default=None)
    worker.add_argument("--regression-baseline", type=Path, default=None)
    worker.add_argument("--regression-baseline-warn-only", action="store_true")
    worker.add_argument("--no-maintenance", action="store_true")
    worker.add_argument("--no-vacuum", action="store_true")
    worker.add_argument("--report-item-limit", type=int, default=20)
    worker.add_argument("--no-lock", action="store_true")
    worker.add_argument("--lock-stale-seconds", type=int, default=3600)
    worker_loop = sub.add_parser("worker-loop")
    worker_loop.add_argument("--scope", default="global")
    worker_loop.add_argument("--iterations", type=int, default=1)
    worker_loop.add_argument("--interval-seconds", type=float, default=60.0)
    worker_loop.add_argument("--continue-on-failure", action="store_true")
    worker_loop.add_argument("--spool-limit", type=int, default=25)
    worker_loop.add_argument("--processing-stale-seconds", type=int, default=3600)
    worker_loop.add_argument("--quality-limit", type=int, default=500)
    worker_loop.add_argument("--review-limit", type=int, default=25)
    worker_loop.add_argument("--triage-limit", type=int, default=500)
    worker_loop.add_argument("--apply-review", action="store_true")
    worker_loop.add_argument("--no-episode-summary", action="store_true")
    worker_loop.add_argument("--episode-summary-min-group-size", type=int, default=5)
    worker_loop.add_argument("--episode-summary-session-min-group-size", type=int, default=20)
    worker_loop.add_argument("--episode-summary-limit", type=int, default=50)
    worker_loop.add_argument("--no-candidate-summary", action="store_true")
    worker_loop.add_argument("--candidate-summary-min-group-size", type=int, default=3)
    worker_loop.add_argument("--candidate-summary-limit", type=int, default=80)
    worker_loop.add_argument("--no-governance-probe", action="store_true")
    worker_loop.add_argument("--governance-query", default="")
    worker_loop.add_argument("--governance-max-input-tokens", type=int, default=0)
    worker_loop.add_argument("--governance-max-model-cost-usd", type=float, default=0.0)
    worker_loop.add_argument("--doctor-query", default="current memory state")
    worker_loop.add_argument("--recall-budget", type=int, default=1600)
    worker_loop.add_argument("--hot-budget", type=int, default=1200)
    worker_loop.add_argument("--regression-manifest", type=Path, default=None)
    worker_loop.add_argument("--regression-baseline", type=Path, default=None)
    worker_loop.add_argument("--regression-baseline-warn-only", action="store_true")
    worker_loop.add_argument("--no-maintenance", action="store_true")
    worker_loop.add_argument("--no-vacuum", action="store_true")
    worker_loop.add_argument("--report-item-limit", type=int, default=20)
    worker_loop.add_argument("--no-lock", action="store_true")
    worker_loop.add_argument("--lock-stale-seconds", type=int, default=3600)
    worker_schedule = sub.add_parser(
        "worker-schedule",
        description="Write reviewable Windows Task Scheduler scripts for a one-shot Ara Memory worker loop.",
    )
    worker_schedule.add_argument(
        "--output",
        type=Path,
        default=Path(".ara-memory") / "scripts" / "install-worker-task.ps1",
        help="Path for the generated install script.",
    )
    worker_schedule.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository working directory.")
    worker_schedule.add_argument("--task-name", default="AraMemoryWorker", help="Windows scheduled task name.")
    worker_schedule.add_argument("--interval-minutes", type=int, default=5, help="Scheduler repeat interval.")
    worker_schedule.add_argument("--scope", default="ara-memory", help="Memory scope for scheduled worker runs.")
    worker_schedule.add_argument("--python", type=Path, default=None, help="Python executable to run.")
    worker_schedule.add_argument("--regression-manifest", type=Path, default=None, help="Recall regression manifest.")
    worker_schedule.add_argument("--regression-baseline", type=Path, default=None, help="Recall regression baseline.")
    worker_schedule.add_argument("--log-path", type=Path, default=None, help="Scheduled worker log path.")
    worker_schedule_verify = sub.add_parser(
        "worker-schedule-verify",
        description="Statically verify generated scheduled-worker scripts before installation.",
    )
    worker_schedule_verify.add_argument(
        "--output",
        type=Path,
        default=Path(".ara-memory") / "scripts" / "install-worker-task.ps1",
        help="Generated install script to verify.",
    )
    worker_schedule_verify.add_argument("--scope", default="ara-memory", help="Expected worker scope.")
    worker_schedule_verify.add_argument(
        "--max-interval-minutes",
        type=int,
        default=15,
        help="Maximum allowed repetition interval in the generated scheduled task.",
    )
    worker_schedule_verify.add_argument(
        "--no-regression-gates",
        action="store_true",
        help="Do not require valid recall-regression manifest and baseline paths.",
    )
    worker_schedule_verify.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--scope", default="global")
    doctor.add_argument("--query", default="current memory state")
    doctor.add_argument("--recall-budget", type=int, default=1600)
    doctor.add_argument("--hot-budget", type=int, default=1200)
    doctor.add_argument("--no-global", action="store_true")
    doctor.add_argument("--json", action="store_true")
    health = sub.add_parser("health")
    health.add_argument("--scope", default="global")
    health.add_argument("--query", default="current memory health")
    health.add_argument("--recall-budget", type=int, default=1600)
    health.add_argument("--hot-budget", type=int, default=1200)
    health.add_argument("--review-limit", type=int, default=500)
    health.add_argument("--backup-max-age-hours", type=float, default=72.0)
    health.add_argument(
        "--backup-target-bytes",
        type=int,
        default=DEFAULT_TARGET_BACKUP_BYTES,
        help="Warn when backup bytes exceed this stewardship target. Use -1 to disable.",
    )
    health.add_argument("--retention-cycle-max-age-hours", type=float, default=72.0)
    health.add_argument("--regression-manifest", type=Path, default=None)
    health.add_argument("--regression-baseline", type=Path, default=None)
    health.add_argument("--compact", action="store_true")
    health.add_argument("--json", action="store_true")
    backup = sub.add_parser("backup")
    backup.add_argument("--output", type=Path, default=None)
    backup.add_argument("--no-archive", action="store_true")
    backup.add_argument(
        "--archive-mode",
        choices=["objects", "full", "none"],
        default=None,
        help=(
            "Archive payload to include. objects keeps encrypted artifact objects without recursive derived "
            "cold/export/failed-backup bundles; full includes the entire archive tree; none omits archive files."
        ),
    )
    backup_stewardship = sub.add_parser(
        "backup-stewardship",
        description=(
            "Review backup storage pressure. Dry-run by default; apply mode takes the worker lock "
            "before deleting verified backups or moving failed backups."
        ),
    )
    backup_stewardship.add_argument("--keep-latest", type=int, default=3)
    backup_stewardship.add_argument("--keep-retention-cycles", type=int, default=2)
    backup_stewardship.add_argument(
        "--target-backup-bytes",
        type=int,
        default=DEFAULT_TARGET_BACKUP_BYTES,
        help=f"Target backup bytes after deleting selected candidates. Defaults to {DEFAULT_TARGET_BACKUP_BYTES} bytes.",
    )
    backup_stewardship.add_argument(
        "--quarantine-failed",
        action="store_true",
        help="Mark failed-verification backup ZIPs for quarantine outside the live backup pool.",
    )
    backup_stewardship.add_argument(
        "--apply",
        action="store_true",
        help="Apply reviewed actions. Requires exact confirmation strings and takes the worker lock.",
    )
    backup_stewardship.add_argument(
        "--confirm",
        default="",
        help='Exact deletion confirmation for redundant verified backups: "DELETE OLD BACKUPS".',
    )
    backup_stewardship.add_argument(
        "--quarantine-confirm",
        default="",
        help='Exact quarantine confirmation for failed backups: "QUARANTINE FAILED BACKUPS".',
    )
    backup_stewardship.add_argument(
        "--no-lock",
        action="store_true",
        help="Do not take the worker lock in apply mode. Use only when the store is otherwise quiescent.",
    )
    backup_stewardship.add_argument(
        "--no-cache-write",
        action="store_true",
        help="Do not update the backup verification cache during dry-run review.",
    )
    backup_stewardship.add_argument("--lock-stale-seconds", type=int, default=3600)
    backup_stewardship.add_argument("--json", action="store_true")
    verify_backup = sub.add_parser("verify-backup")
    verify_backup.add_argument("path", type=Path)
    verify_backup.add_argument(
        "--trust-root",
        type=Path,
        default=None,
        help="Memory root whose .backup-signing-key should verify this archive. Defaults to --root.",
    )
    restore_backup = sub.add_parser("restore-backup")
    restore_backup.add_argument("path", type=Path)
    restore_backup.add_argument("--target-root", type=Path, required=True)
    restore_backup.add_argument("--force", action="store_true")
    restore_backup.add_argument(
        "--trust-root",
        type=Path,
        default=None,
        help="Memory root whose .backup-signing-key should verify this archive. Defaults to --root.",
    )
    restore_drill = sub.add_parser("restore-drill")
    restore_drill.add_argument("path", type=Path)
    restore_drill.add_argument("--scope", default="global")
    restore_drill.add_argument("--query", default=None)
    restore_drill.add_argument("--recall-budget", type=int, default=1200)
    restore_drill.add_argument(
        "--trust-root",
        type=Path,
        default=None,
        help="Memory root whose .backup-signing-key should verify this archive. Defaults to --root.",
    )
    cold_export = sub.add_parser("cold-export")
    cold_export.add_argument("--output", type=Path, default=None)
    cold_export.add_argument("--scope", default=None)
    cold_export.add_argument(
        "--status",
        action="append",
        choices=["superseded", "rejected", "quarantined"],
        default=None,
        help="Cold status to export. Repeat to include multiple statuses.",
    )
    cold_export.add_argument("--limit", type=int, default=None)
    cold_export.add_argument("--order", choices=["newest", "oldest"], default="newest")
    cold_export.add_argument("--no-events", action="store_true")
    verify_cold_export = sub.add_parser("verify-cold-export")
    verify_cold_export.add_argument("path", type=Path)
    verify_cold_export.add_argument(
        "--trust-root",
        type=Path,
        default=None,
        help="Memory root whose .backup-signing-key should verify this export. Defaults to --root.",
    )
    prune_plan = sub.add_parser("prune-plan")
    prune_plan.add_argument("--scope", default=None)
    prune_plan.add_argument("--limit", type=int, default=None)
    prune_plan.add_argument("--cold-export", type=Path, default=None)
    prune_plan.add_argument("--query", action="append", default=[])
    prune_plan.add_argument("--query-file", type=Path, default=None)
    prune_plan.add_argument("--recall-budget", type=int, default=1200)
    prune_plan.add_argument("--no-global", action="store_true")
    prune_plan.add_argument("--json", action="store_true")
    shadow_prune = sub.add_parser("shadow-prune")
    shadow_prune.add_argument("--backup", type=Path, required=True)
    shadow_prune.add_argument("--cold-export", type=Path, required=True)
    shadow_prune.add_argument("--scope", default=None)
    shadow_prune.add_argument("--limit", type=int, default=None)
    shadow_prune.add_argument("--query", action="append", default=[])
    shadow_prune.add_argument("--query-file", type=Path, default=None)
    shadow_prune.add_argument("--recall-budget", type=int, default=1200)
    shadow_prune.add_argument("--doctor-query", default="current memory state")
    shadow_prune.add_argument("--no-global", action="store_true")
    shadow_prune.add_argument("--json", action="store_true")
    prepare_live_prune = sub.add_parser("prepare-live-prune")
    prepare_live_prune.add_argument("--backup", type=Path, required=True)
    prepare_live_prune.add_argument("--cold-export", type=Path, required=True)
    prepare_live_prune.add_argument("--scope", default=None)
    prepare_live_prune.add_argument("--limit", type=int, default=None)
    prepare_live_prune.add_argument("--query", action="append", default=[])
    prepare_live_prune.add_argument("--query-file", type=Path, default=None)
    prepare_live_prune.add_argument("--recall-budget", type=int, default=1200)
    prepare_live_prune.add_argument("--doctor-query", default="current memory state")
    prepare_live_prune.add_argument("--ttl-minutes", type=int, default=30)
    prepare_live_prune.add_argument("--no-global", action="store_true")
    live_prune = sub.add_parser(
        "live-prune",
        description='Irreversibly delete approved cold capsules only. Requires exact confirmation: "DELETE COLD CAPSULES".',
    )
    live_prune.add_argument("--approval-token", required=True)
    live_prune.add_argument("--confirm", required=True, help='Exact confirmation text: "DELETE COLD CAPSULES".')
    live_prune.add_argument("--json", action="store_true")
    ops = sub.add_parser("irreversible-operations")
    ops.add_argument("--limit", type=int, default=20)
    maintenance = sub.add_parser("maintenance")
    maintenance.add_argument("--no-vacuum", action="store_true")
    archive_encrypt = sub.add_parser(
        "archive-encrypt",
        description="Encrypt plaintext archive/objects payloads in place. Dry-run by default.",
    )
    archive_encrypt.add_argument("--apply", action="store_true")
    archive_encrypt.add_argument("--limit", type=int, default=None)
    retention = sub.add_parser("retention")
    retention.add_argument("--scope", default=None)
    retention.add_argument("--cold-limit", type=int, default=20)
    retention.add_argument("--json", action="store_true")
    cold_stewardship = sub.add_parser("cold-stewardship")
    cold_stewardship.add_argument("--scope", default=None)
    cold_stewardship.add_argument("--group-limit", type=int, default=12)
    cold_stewardship.add_argument("--examples-per-group", type=int, default=2)
    cold_stewardship.add_argument("--max-cycle-age-hours", type=float, default=72.0)
    cold_stewardship.add_argument("--json", action="store_true")
    cold_map = sub.add_parser(
        "cold-map",
        help="Build a compact query-led navigation map over distant cold memory without rendering cold bodies.",
    )
    cold_map.add_argument("query", nargs="?", default="")
    cold_map.add_argument("--scope", default=None)
    cold_map.add_argument("--group-limit", type=int, default=10)
    cold_map.add_argument("--examples-per-group", type=int, default=2)
    cold_map.add_argument("--budget", type=int, default=900)
    cold_map.add_argument("--json", action="store_true")
    provenance_compact = sub.add_parser(
        "provenance-compact",
        help="Plan or apply guarded compaction of active capsule source-event links.",
    )
    provenance_compact.add_argument("--scope", default=None)
    provenance_compact.add_argument("--keep-events", type=int, default=8)
    provenance_compact.add_argument("--min-pinned-events", type=int, default=20)
    provenance_compact.add_argument("--limit", type=int, default=20)
    provenance_compact.add_argument("--include-non-summary", action="store_true")
    provenance_compact.add_argument("--apply", action="store_true")
    provenance_compact.add_argument("--confirm", default="")
    provenance_compact.add_argument("--recall-manifest", type=Path, default=None)
    provenance_compact.add_argument("--recall-baseline", type=Path, default=None)
    provenance_compact.add_argument("--max-token-growth", type=float, default=0.25)
    provenance_compact.add_argument("--min-overlap", type=float, default=0.35)
    provenance_compact.add_argument("--json", action="store_true")
    lifecycle = sub.add_parser(
        "lifecycle",
        help="Classify memory into purpose-aware lifecycle tiers for hot, working, guarded, and cold recall policy.",
    )
    lifecycle.add_argument("--scope", default=None)
    lifecycle.add_argument("--limit", type=int, default=1000)
    lifecycle.add_argument("--examples-per-tier", type=int, default=3)
    lifecycle.add_argument("--target-hot-tokens", type=int, default=1200)
    lifecycle.add_argument("--json", action="store_true")
    retention_cycle = sub.add_parser(
        "retention-cycle",
        description=(
            "Build non-destructive pruning-readiness evidence. Takes the worker lock by default; "
            "--no-shadow cannot pass pruning readiness."
        ),
    )
    retention_cycle.add_argument("--scope", default=None)
    retention_cycle.add_argument("--backup-output", type=Path, default=None)
    retention_cycle.add_argument(
        "--backup-archive-mode",
        choices=["objects", "full", "none"],
        default="objects",
        help="Archive payload for the retention-cycle backup. Defaults to objects to avoid recursive evidence bloat.",
    )
    retention_cycle.add_argument("--cold-output", type=Path, default=None)
    retention_cycle.add_argument("--report-output", type=Path, default=None)
    retention_cycle.add_argument("--limit", type=int, default=None)
    retention_cycle.add_argument("--query", action="append", default=[])
    retention_cycle.add_argument("--query-file", type=Path, default=None)
    retention_cycle.add_argument("--recall-budget", type=int, default=1200)
    retention_cycle.add_argument("--doctor-query", default="current memory state")
    retention_cycle.add_argument("--no-global", action="store_true")
    retention_cycle.add_argument(
        "--no-shadow",
        action="store_true",
        help="Collect partial evidence without shadow-prune; the cycle will not pass pruning readiness.",
    )
    retention_cycle.add_argument(
        "--no-lock",
        action="store_true",
        help="Do not take the worker lock. Use only when the store is otherwise quiescent.",
    )
    retention_cycle.add_argument(
        "--lock-stale-seconds",
        type=int,
        default=3600,
        help="Treat the worker lock as stale after this many seconds.",
    )
    retention_cycle.add_argument("--json", action="store_true")
    candidate_pressure = sub.add_parser("candidate-pressure")
    candidate_pressure.add_argument("--scope", default=None)
    candidate_pressure.add_argument("--limit", type=int, default=20)
    candidate_pressure.add_argument("--json", action="store_true")
    candidate_summary = sub.add_parser("candidate-summary")
    candidate_summary.add_argument("--scope", default=None)
    candidate_summary.add_argument(
        "--pattern",
        choices=[
            "all",
            "decision_memory_policy",
            "project_file_artifact",
            "failure_operational_update",
            "failure_taxonomy_discussion",
            "failure_success_command",
            "failure_worktree_evidence",
            "goal_purpose_update",
            "procedure_command",
            "procedure_memory_policy",
            "procedure_progress_update",
            "project_worktree_evidence",
        ],
        default="all",
    )
    candidate_summary.add_argument("--min-group-size", type=int, default=3)
    candidate_summary.add_argument("--limit", type=int, default=80)
    candidate_summary.add_argument("--apply", action="store_true")
    candidate_summary.add_argument("--json", action="store_true")
    conflict_adjudicate = sub.add_parser("conflict-adjudicate")
    conflict_adjudicate.add_argument("--scope", default=None)
    conflict_adjudicate.add_argument("--limit", type=int, default=500)
    conflict_adjudicate.add_argument("--apply", action="store_true")
    conflict_adjudicate.add_argument("--json", action="store_true")
    episode_summary = sub.add_parser("episode-summary")
    episode_summary.add_argument("--scope", default=None)
    episode_summary.add_argument("--pattern", choices=["command", "file_artifact", "git_status", "session"], default="command")
    episode_summary.add_argument("--min-group-size", type=int, default=5)
    episode_summary.add_argument("--limit", type=int, default=50)
    episode_summary.add_argument("--apply", action="store_true")
    episode_summary.add_argument("--json", action="store_true")
    sub.add_parser("eval")
    sub.add_parser("context-eval")
    recall_regression = sub.add_parser("recall-regression")
    recall_regression.add_argument("--manifest", type=Path, required=True)
    recall_regression.add_argument("--baseline", type=Path, default=None)
    recall_regression.add_argument("--write-baseline", type=Path, default=None)
    recall_regression.add_argument("--max-token-growth", type=float, default=0.25)
    recall_regression.add_argument("--min-overlap", type=float, default=0.35)
    sub.add_parser("stats")

    args = parser.parse_args(argv)
    memory = AraMemory(args.root)

    if args.cmd == "init":
        memory.init()
        print(f"Initialized Ara Memory OS at {memory.store.root.resolve()}")
        return 0

    if args.cmd == "retain":
        text = _read_text_arg(args.text, args.file)
        metadata = json.loads(args.metadata)
        event = memory.retain(
            kind=args.kind,
            text=text,
            source=args.source,
            scope=args.scope,
            metadata=metadata,
            allow_raw_private=args.allow_raw_secret,
        )
        print(json.dumps({"event_id": event.id, "created_at": event.created_at}, ensure_ascii=False))
        return 0

    if args.cmd == "ingest-file":
        event_id = ingest_file(
            memory,
            path=args.path,
            scope=args.scope,
            source=args.source,
            caption=args.caption,
            max_text_chars=args.max_text_chars,
        )
        result = {"event_id": event_id}
        if args.consolidate:
            result["capsules_created"] = memory.consolidate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "remember-turn":
        payload = _read_json_arg(args.file)
        result = remember_turn(
            memory,
            payload,
            scope=args.scope,
            source=args.source,
            consolidate=not args.no_consolidate,
            sleep=args.sleep,
            hot_budget=args.hot_budget,
            capture_cwd=args.capture_cwd,
            include_untracked_content=args.include_untracked_content,
            max_file_chars=args.max_file_chars,
            max_text_chars=args.max_text_chars,
            allow_raw_private=args.allow_raw_secret,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "plan-turn":
        payload = _read_json_arg(args.file)
        result = plan_turn_ingress(
            payload,
            capture_cwd=args.capture_cwd,
            max_file_chars=args.max_file_chars,
            max_text_chars=args.max_text_chars,
            direct_text_threshold=args.direct_text_threshold,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "ingress-turn":
        payload = _read_json_arg(args.file)
        result = execute_turn_ingress(
            memory,
            payload,
            scope=args.scope,
            source=args.source,
            consolidate=not args.no_consolidate,
            sleep=args.sleep,
            hot_budget=args.hot_budget,
            capture_cwd=args.capture_cwd,
            include_untracked_content=args.include_untracked_content,
            max_file_chars=args.max_file_chars,
            max_text_chars=args.max_text_chars,
            direct_text_threshold=args.direct_text_threshold,
            mode=args.mode,
            dry_run=args.dry_run,
            allow_raw_private=args.allow_raw_secret,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "spool-turn":
        payload = _read_json_arg(args.file)
        result = memory.spool_turn(
            payload,
            scope=args.scope,
            source=args.source,
            consolidate=not args.no_consolidate,
            sleep=args.sleep,
            hot_budget=args.hot_budget,
            capture_cwd=args.capture_cwd,
            include_untracked_content=args.include_untracked_content,
            max_file_chars=args.max_file_chars,
            max_text_chars=args.max_text_chars,
            allow_raw_private=args.allow_raw_secret,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "drain-spool":
        result = memory.drain_spool(
            limit=args.limit,
            scope=args.scope,
            stop_on_error=args.stop_on_error,
            processing_stale_seconds=args.processing_stale_seconds,
            stabilize=args.stabilize,
            stabilization_scope=args.stabilization_scope,
            stabilization_episode_min_group_size=args.stabilization_episode_min_group_size,
            stabilization_session_min_group_size=args.stabilization_session_min_group_size,
            stabilization_candidate_min_group_size=args.stabilization_candidate_min_group_size,
            stabilization_limit=args.stabilization_limit,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.passed else 1

    if args.cmd == "spool-stats":
        print(json.dumps(memory.spool_stats(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "consolidate":
        count = memory.consolidate(limit=args.limit)
        print(json.dumps({"capsules_created": count}, ensure_ascii=False))
        return 0

    if args.cmd == "capture-worktree":
        event_ids = capture_worktree(
            memory,
            cwd=args.cwd.resolve(),
            scope=args.scope,
            include_untracked_content=args.include_untracked_content,
            max_file_chars=args.max_file_chars,
        )
        result = {"events_retained": event_ids}
        if args.consolidate:
            result["capsules_created"] = memory.consolidate()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "recall":
        result = memory.recall_result(
            args.query,
            scope=args.scope,
            budget=args.budget,
            include_global=not args.no_global,
            include_hot=args.hot,
        )
        print(result.pack)
        if args.diagnostics:
            print("\n# Recall Diagnostics")
            print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "recall-candidates":
        result = memory.recall_candidates(
            args.query,
            scope=args.scope,
            budget=args.budget,
            include_global=not args.no_global,
            include_hot=args.hot,
            candidate_limit=args.limit,
        )
        print(json.dumps(result.as_dict(include_capsules=args.include_capsules), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "recall-plan":
        result = memory.recall_plan(
            args.query,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            output_tokens=args.output_tokens,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "recall-context":
        result = memory.recall_context(
            args.query,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            output_tokens=args.output_tokens,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text(include_plan=not args.pack_only))
        return 0

    if args.cmd == "recall-policy":
        result = memory.recall_policy(
            args.query,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            cold_budget=args.cold_budget,
            cold_group_limit=args.cold_group_limit,
            output_tokens=args.output_tokens,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "recall-quality":
        result = memory.recall_quality(
            args.query,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            working_budget=args.working_budget,
            recall_budget=args.recall_budget,
            limit=args.limit,
            min_evaluated=args.min_evaluated,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-frame":
        query = args.query if args.query is not None else sys.stdin.read().strip()
        result = memory.reconsolidation_frame(
            query,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            working_budget=args.working_budget,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.status in {"pass", "watch"} else 1

    if args.cmd == "reconsolidation-prepare":
        query = args.query if args.query is not None else sys.stdin.read().strip()
        result = memory.prepare_reconsolidation(
            query,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            working_budget=args.working_budget,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            ttl_minutes=args.ttl_minutes,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.prepared or result.frame.status in {"pass", "watch"} else 1

    if args.cmd == "reconsolidation-apply":
        result = memory.apply_reconsolidation(
            approval_token=args.approval_token,
            confirmation=args.confirm,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-review":
        cases = load_recall_regression_cases(args.regression_manifest) if args.regression_manifest else None
        baseline = load_recall_regression_baseline(args.regression_baseline) if args.regression_baseline else None
        result = memory.review_reconsolidation(
            scope=args.scope,
            approval_id=args.approval_id,
            limit=args.limit,
            regression_cases=cases,
            regression_baseline=baseline,
        )
        queue = None
        if args.record_queue:
            queue = record_reconsolidation_review_queue(memory, result)
        if args.json:
            payload = result.as_dict()
            if queue is not None:
                payload["queue"] = queue.as_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
            if queue is not None:
                print()
                print(queue.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-review-queue":
        result = list_reconsolidation_review_queue(
            memory,
            scope=args.scope,
            status=args.status,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "reconsolidation-backfill-rollback-witnesses":
        result = backfill_legacy_reconsolidation_rollback_witnesses(
            memory,
            scope=args.scope,
            approval_id=args.approval_id,
            limit=args.limit,
            apply=args.apply,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "reconsolidation-strong-preflight":
        query = args.query if args.query is not None else sys.stdin.read().strip()
        cases = load_recall_regression_cases(args.regression_manifest) if args.regression_manifest else None
        baseline = load_recall_regression_baseline(args.regression_baseline) if args.regression_baseline else None
        result = memory.strong_reconsolidation_preflight(
            query,
            backup_path=args.backup,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            working_budget=args.working_budget,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            regression_cases=cases,
            regression_baseline=baseline,
            actions=args.action,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "prepare-live-reconsolidation-action":
        query = args.query if args.query is not None else sys.stdin.read().strip()
        cases = load_recall_regression_cases(args.regression_manifest) if args.regression_manifest else None
        baseline = load_recall_regression_baseline(args.regression_baseline) if args.regression_baseline else None
        result = memory.prepare_live_reconsolidation_action(
            query,
            backup_path=args.backup,
            action=args.action,
            confirmation=args.confirm,
            scope=args.scope,
            budgets=_parse_budget_list(args.budgets),
            working_budget=args.working_budget,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            regression_cases=cases,
            regression_baseline=baseline,
            ttl_minutes=args.ttl_minutes,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "live-reconsolidation-cool":
        result = memory.live_reconsolidation_cool(
            approval_token=args.approval_token,
            capsule_id=args.capsule_id,
            confirmation=args.confirm,
            reason=args.reason,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-action-witnesses":
        result = memory.review_reconsolidation_action_witnesses(
            scope=args.scope,
            action=args.action,
            witness_id=args.witness_id,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-action-shadow-rollback":
        result = memory.shadow_reconsolidation_action_rollback(
            backup_path=args.backup,
            scope=args.scope,
            action=args.action,
            witness_id=args.witness_id,
            limit=args.limit,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "prepare-live-reconsolidation-action-rollback":
        try:
            result = memory.prepare_live_reconsolidation_action_rollback(
                backup_path=args.backup,
                witness_id=args.witness_id,
                scope=args.scope,
                action=args.action,
                ttl_minutes=args.ttl_minutes,
                doctor_query=args.doctor_query,
                recall_budget=args.recall_budget,
                hot_budget=args.hot_budget,
                include_global=not args.no_global,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "reconsolidation-action-rollback-approvals":
        result = memory.review_reconsolidation_action_rollback_approvals(
            scope=args.scope,
            action=args.action,
            approval_id=args.approval_id,
            witness_id=args.witness_id,
            include_non_prepared=not args.prepared_only,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "live-reconsolidation-action-rollback":
        result = memory.live_reconsolidation_action_rollback(
            approval_token=args.approval_token,
            confirmation=args.confirm,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-shadow-rollback":
        result = memory.shadow_reconsolidation_rollback(
            backup_path=args.backup,
            scope=args.scope,
            approval_id=args.approval_id,
            witness_id=args.witness_id,
            limit=args.limit,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "prepare-live-reconsolidation-rollback":
        try:
            result = memory.prepare_live_reconsolidation_rollback(
                backup_path=args.backup,
                witness_id=args.witness_id,
                scope=args.scope,
                ttl_minutes=args.ttl_minutes,
                doctor_query=args.doctor_query,
                recall_budget=args.recall_budget,
                hot_budget=args.hot_budget,
                include_global=not args.no_global,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "live-reconsolidation-rollback":
        result = memory.live_reconsolidation_rollback(
            approval_token=args.approval_token,
            confirmation=args.confirm,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-exception-witness":
        try:
            evidence = json.loads(args.evidence)
        except json.JSONDecodeError as exc:
            print(f"--evidence must be JSON: {exc}", file=sys.stderr)
            return 2
        result = memory.record_reconsolidation_exception_witness(
            witness_id=args.witness_id,
            capsule_id=args.capsule_id,
            field=args.field,
            reason=args.reason,
            evidence=evidence if isinstance(evidence, dict) else {"value": evidence},
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "reconsolidation-exception-witnesses":
        result = memory.list_reconsolidation_exception_witnesses(
            scope=args.scope,
            witness_id=args.witness_id,
            status=args.status,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "recall-policy-impact":
        helped = None if args.helped == "unknown" else args.helped == "true"
        intent = args.intent.strip()
        strategy = args.strategy.strip()
        action_names = [str(action).strip() for action in args.action_name if str(action).strip()]
        if not intent or not strategy or not action_names:
            print(
                "recall-policy-impact requires non-empty --intent, --strategy, and --action-name from the policy actually used.",
                file=sys.stderr,
            )
            return 2
        event = memory.record_recall_policy_impact(
            scope=args.scope,
            query=args.query,
            intent=intent,
            strategy=strategy,
            action_names=action_names,
            outcome=args.outcome,
            helped=helped,
        )
        payload = {
            "event_id": event.id,
            "scope": event.scope,
            "query": args.query,
            "intent": intent,
            "action_names": list(dict.fromkeys(action_names)),
            "helped": helped,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.cmd == "recall-policy-eval":
        result = memory.evaluate_recall_policy(
            scope=args.scope,
            include_global=not args.no_global,
            limit=args.limit,
            min_evaluated=args.min_evaluated,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "working-memory":
        prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
        result = memory.working_memory(
            prompt=prompt,
            scope=args.scope,
            active_files=args.active_file,
            command_errors=args.command_error,
            budget=args.budget,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "working-memory-impact":
        helped = None if args.helped == "unknown" else args.helped == "true"
        event = memory.record_memory_impact(
            scope=args.scope,
            cue=args.cue,
            capsule_ids=args.capsule_id,
            outcome=args.outcome,
            helped=helped,
        )
        payload = {"event_id": event.id, "scope": event.scope, "capsule_ids": list(dict.fromkeys(args.capsule_id))}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.cmd == "working-memory-impact-eval":
        result = memory.evaluate_memory_impact(
            scope=args.scope,
            include_global=not args.no_global,
            limit=args.limit,
            min_evaluated=args.min_evaluated,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "govern-turn":
        payload = _read_json_arg(args.file)
        impact_helped = None if args.impact_helped == "unknown" else args.impact_helped == "true"
        result = memory.govern_turn(
            payload,
            scope=args.scope,
            capture_cwd=args.capture_cwd,
            budgets=_parse_budget_list(args.budgets) if args.budgets else None,
            working_budget=args.working_budget if args.working_budget > 0 else None,
            budget_profile=args.budget_profile,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            direct_text_threshold=args.direct_text_threshold,
            max_file_chars=args.max_file_chars,
            max_text_chars=args.max_text_chars,
            output_tokens=args.output_tokens,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
            max_input_tokens=args.max_input_tokens,
            max_model_cost_usd=args.max_model_cost_usd,
            record_working_impact=args.record_working_impact,
            impact_outcome=args.impact_outcome,
            impact_helped=impact_helped,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "agency-review":
        prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
        result = memory.agency_review(
            prompt=prompt,
            scope=args.scope,
            proposed_action=args.proposed_action,
            constraints=args.constraint,
            active_files=args.active_file,
            command_errors=args.command_error,
            budgets=_parse_budget_list(args.budgets),
            working_budget=args.working_budget,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            include_hot=not args.no_hot,
            include_health=args.health,
            record=args.record,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        if args.strict_action_exit:
            return 0 if result.action_allowed else 1
        return 0 if result.passed or result.stance in {"refuse-or-reframe", "ask-before-acting", "ask-or-roadmap"} else 1

    if args.cmd == "purpose-check":
        result = memory.purpose_check(
            scope=args.scope,
            query=args.query,
            budget=args.budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
            repair_hot=args.repair_hot,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "identity-check":
        result = memory.identity_check(
            scope=args.scope,
            query=args.query,
            budget=args.budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
            repair_hot=args.repair_hot,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "milestone-check":
        cases = load_recall_regression_cases(args.regression_manifest) if args.regression_manifest else None
        baseline = load_recall_regression_baseline(args.regression_baseline) if args.regression_baseline else None
        result = memory.milestone_check(
            scope=args.scope,
            query=args.query,
            recall_budgets=_parse_budget_list(args.recall_budgets),
            health_query=args.health_query,
            purpose_query=args.purpose_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            candidate_ratio_limit=args.candidate_ratio_limit,
            repair_hot=not args.no_repair_hot,
            regression_cases=cases,
            regression_baseline=baseline,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "failure-kind-audit":
        result = memory.failure_kind_audit(
            scope=args.scope,
            statuses=args.status,
            limit=args.limit,
            dry_run=not args.apply,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "self-kind-audit":
        result = memory.self_kind_audit(
            scope=args.scope,
            statuses=args.status,
            limit=args.limit,
            dry_run=not args.apply,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "goal-roadmap":
        cases = load_recall_regression_cases(args.regression_manifest) if args.regression_manifest else None
        baseline = load_recall_regression_baseline(args.regression_baseline) if args.regression_baseline else None
        result = memory.goal_roadmap(
            scope=args.scope,
            regression_cases=cases,
            regression_baseline=baseline,
            repair_hot=args.repair_hot,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.status in {"pass", "watch"} else 1

    if args.cmd == "graph-readiness":
        result = memory.graph_activation_readiness(
            scope=args.scope,
            query=args.query,
            budgets=_parse_budget_list(args.budgets),
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.status in {"pass", "watch"} else 1

    if args.cmd == "global-spreading-sandbox":
        result = memory.global_spreading_sandbox(
            scope=args.scope,
            queries=args.query,
            budgets=_parse_budget_list(args.budgets),
            max_edges=args.max_edges,
            max_supplemented=args.max_supplemented,
            max_depth=args.max_depth,
            min_quality=args.min_quality,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.status in {"pass", "watch"} else 1

    if args.cmd == "privacy-pre-push":
        result = memory.privacy_pre_push(
            repo=args.repo,
            include_untracked=args.include_untracked,
            include_direct_identifiers=args.include_direct_identifiers,
            max_bytes=args.max_bytes,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "relation-merge":
        result = run_relation_merge_dry_run(
            memory.store,
            scope=args.scope,
            limit=args.limit,
            threshold=args.threshold,
            node_limit=args.node_limit,
            include_global=args.include_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "relation-merge-prepare":
        result = prepare_relation_merge_approval(
            memory.store,
            scope=args.scope,
            limit=args.limit,
            threshold=args.threshold,
            node_limit=args.node_limit,
            include_global=args.include_global,
            ttl_minutes=args.ttl_minutes,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "relation-merge-apply":
        result = apply_relation_merge_approval(
            memory.store,
            approval_token=args.approval_token,
            confirmation=args.confirm,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "relation-merge-review":
        result = review_relation_merge_witnesses(
            memory.store,
            scope=args.scope,
            approval_id=args.approval_id,
            limit=args.limit,
        )
        regression = None
        if args.regression_manifest:
            cases = load_recall_regression_cases(args.regression_manifest)
            baseline = load_recall_regression_baseline(args.regression_baseline)
            regression = memory.recall_regression(cases, baseline=baseline)
        passed = result.passed and (regression is None or regression.passed)
        queue = None
        if args.record_queue:
            queue = record_relation_merge_review_queue(
                memory.store,
                result,
                regression=regression.as_dict() if regression is not None else None,
            )
        if args.json:
            payload = result.as_dict()
            if regression is not None:
                payload["regression"] = regression.as_dict()
                payload["passed"] = passed
            if queue is not None:
                payload["queue"] = queue.as_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
            if regression is not None:
                regression_payload = regression.as_dict()
                print()
                print(f"## Recall Regression Sandbox")
                print(f"status: {'pass' if regression.passed else 'fail'}")
                print(f"cases: {_passed_count(regression_payload.get('cases', []))}")
                print(f"baseline: {_passed_count(regression_payload.get('baseline_comparison', []))}")
            if queue is not None:
                print()
                print(queue.to_text())
        return 0 if passed else 1

    if args.cmd == "relation-merge-review-queue":
        result = list_relation_merge_review_queue(
            memory.store,
            scope=args.scope,
            status=args.status,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "promote":
        if not memory.promote(args.capsule_id, actor=args.actor, reason=args.reason):
            raise SystemExit(f"Capsule not found: {args.capsule_id}")
        print(json.dumps({"promoted": args.capsule_id}, ensure_ascii=False))
        return 0

    if args.cmd == "reject":
        if not memory.reject(args.capsule_id, actor=args.actor, reason=args.reason):
            raise SystemExit(f"Capsule not found: {args.capsule_id}")
        print(json.dumps({"rejected": args.capsule_id}, ensure_ascii=False))
        return 0

    if args.cmd == "quarantine":
        if not memory.quarantine(args.capsule_id, actor=args.actor, reason=args.reason):
            raise SystemExit(f"Capsule not found: {args.capsule_id}")
        print(json.dumps({"quarantined": args.capsule_id}, ensure_ascii=False))
        return 0

    if args.cmd == "list":
        rows = memory.list_capsules(scope=args.scope, status=args.status, kind=args.kind, limit=args.limit)
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "actions":
        print(json.dumps(memory.list_actions(limit=args.limit), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "sleep":
        report = memory.sleep(scope=args.scope, dry_run=args.dry_run)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "sleep-runs":
        print(json.dumps(memory.list_sleep_runs(limit=args.limit), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "hot":
        state = memory.build_hot(scope=args.scope, budget=args.budget)
        print(json.dumps({"path": str(state.path), "estimated_tokens": state.estimated_tokens}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "show-hot":
        state = memory.read_hot(scope=args.scope)
        if state is None:
            raise SystemExit(f"No hot memory for scope: {args.scope}")
        print(state.text)
        return 0

    if args.cmd == "estimate-cost":
        result = memory.recall_result(
            args.query,
            scope=args.scope,
            budget=args.budget,
            include_global=not args.no_global,
            include_hot=False,
        )
        estimate = estimate_api_cost(
            input_tokens=int(result.diagnostics["estimated_tokens_after"]),
            output_tokens=args.output_tokens,
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        )
        stats = memory.stats()
        payload = {
            "recall": result.diagnostics,
            "api_cost": {
                "input_tokens": estimate.input_tokens,
                "output_tokens": estimate.output_tokens,
                "input_cost_usd": estimate.input_cost_usd,
                "output_cost_usd": estimate.output_cost_usd,
                "total_cost_usd": estimate.total_cost_usd,
            },
            "storage_bytes": stats["storage_bytes"],
            "live_storage_bytes": stats["live_storage_bytes"],
            "storage_breakdown": stats["storage_breakdown"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "audit":
        print(memory.audit())
        return 0

    if args.cmd == "risk":
        print(json.dumps(memory.risk_report(scope=args.scope, limit=args.limit), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "review":
        print(json.dumps(memory.review(scope=args.scope, limit=args.limit), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "quality":
        result = memory.quality(scope=args.scope, limit=args.limit, persist=args.persist)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "promotion-candidates":
        result = memory.promotion_candidates(scope=args.scope, limit=args.limit, threshold=args.threshold)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "review-queue":
        print(json.dumps(memory.review_queue(scope=args.scope, status=args.status, limit=args.limit), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "review-triage":
        result = memory.review_triage(
            scope=args.scope,
            status=args.status,
            limit=args.limit,
            examples_per_group=args.examples_per_group,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "resolve-review":
        if not memory.resolve_review(args.queue_id):
            raise SystemExit(f"Open review queue item not found: {args.queue_id}")
        print(json.dumps({"resolved": args.queue_id}, ensure_ascii=False))
        return 0

    if args.cmd == "review-witnesses":
        rows = memory.review_witnesses(scope=args.scope, action=args.action, limit=args.limit)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            for row in rows:
                print(
                    f"{row['id']} {row['requested_action']}->{row['applied_action']} "
                    f"{row['capsule_id']} {row['before_status']}->{row['after_status']}"
                )
        return 0

    if args.cmd == "prepare-review-rollback":
        result = memory.prepare_review_rollback(args.witness_id, ttl_minutes=args.ttl_minutes)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"approval_id: {result['approval_id']}")
            print(f"review_witness_id: {result['review_witness_id']}")
            print(f"confirmation_required: {result['confirmation_required']}")
            print(f"expires_at: {result['expires_at']}")
            print(f"token: {result['token']}")
        return 0

    if args.cmd == "live-review-rollback":
        result = memory.live_review_rollback(args.approval_token, confirm=args.confirm)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            status = "passed" if result.get("passed") else "blocked"
            print(f"{status}: {result.get('approval_id')}")
            for item in result.get("recommendations", []):
                print(f"- {item}")
        return 0 if result.get("passed") else 1

    if args.cmd == "mutation-preflight":
        result = memory.mutation_preflight(
            capsule_id=args.capsule_id,
            action=args.action,
            title=args.title,
            body=args.body,
            tags=args.tag,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "prepare-mutation":
        result = memory.prepare_mutation(
            capsule_id=args.capsule_id,
            action=args.action,
            title=args.title,
            body=args.body,
            tags=args.tag,
            ttl_minutes=args.ttl_minutes,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"approval_id: {result['approval_id']}")
            print(f"capsule_id: {result['capsule_id']}")
            print(f"action: {result['action']}")
            print(f"confirmation_required: {result['confirmation_required']}")
            print(f"expires_at: {result['expires_at']}")
            print(f"token: {result['token']}")
        return 0

    if args.cmd == "live-mutation-apply":
        result = memory.live_mutation_apply(args.approval_token, confirm=args.confirm)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            status = "passed" if result.get("passed") else "blocked"
            print(f"{status}: {result.get('approval_id')}")
            for item in result.get("recommendations", []):
                print(f"- {item}")
        return 0 if result.get("passed") else 1

    if args.cmd == "prepare-mutation-rollback":
        result = memory.prepare_mutation_rollback(args.witness_id, ttl_minutes=args.ttl_minutes)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"approval_id: {result['approval_id']}")
            print(f"mutation_witness_id: {result['mutation_witness_id']}")
            print(f"confirmation_required: {result['confirmation_required']}")
            print(f"expires_at: {result['expires_at']}")
            print(f"token: {result['token']}")
        return 0

    if args.cmd == "live-mutation-rollback":
        result = memory.live_mutation_rollback(args.approval_token, confirm=args.confirm)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            status = "passed" if result.get("passed") else "blocked"
            print(f"{status}: {result.get('approval_id')}")
            for item in result.get("recommendations", []):
                print(f"- {item}")
        return 0 if result.get("passed") else 1

    if args.cmd == "review-worker":
        result = memory.review_worker(scope=args.scope, limit=args.limit, dry_run=not args.apply)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "review-compact":
        result = memory.review_compact(scope=args.scope, limit=args.limit, dry_run=not args.apply)
        if args.json:
            payload = result.as_compact_dict() if args.compact else result.as_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        elif args.compact:
            print(result.to_compact_text())
        else:
            print(result.to_text())
        return 0

    if args.cmd == "review-redact":
        result = memory.review_redact(scope=args.scope, limit=args.limit, dry_run=not args.apply)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "worker":
        result = memory.worker(
            scope=args.scope,
            spool_limit=args.spool_limit,
            processing_stale_seconds=args.processing_stale_seconds,
            quality_limit=args.quality_limit,
            review_limit=args.review_limit,
            triage_limit=args.triage_limit,
            apply_review=args.apply_review,
            episode_summary=not args.no_episode_summary,
            episode_summary_min_group_size=args.episode_summary_min_group_size,
            episode_summary_session_min_group_size=args.episode_summary_session_min_group_size,
            episode_summary_limit=args.episode_summary_limit,
            candidate_summary=not args.no_candidate_summary,
            candidate_summary_min_group_size=args.candidate_summary_min_group_size,
            candidate_summary_limit=args.candidate_summary_limit,
            governance_probe=not args.no_governance_probe,
            governance_query=args.governance_query,
            governance_max_input_tokens=args.governance_max_input_tokens,
            governance_max_model_cost_usd=args.governance_max_model_cost_usd,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            regression_manifest=args.regression_manifest,
            regression_baseline=args.regression_baseline,
            regression_baseline_drift_warn_only=args.regression_baseline_warn_only,
            run_maintenance_step=not args.no_maintenance,
            vacuum=not args.no_vacuum,
            report_item_limit=args.report_item_limit,
            use_lock=not args.no_lock,
            lock_stale_seconds=args.lock_stale_seconds,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.passed else 1

    if args.cmd == "worker-loop":
        result = memory.worker_loop(
            iterations=args.iterations,
            interval_seconds=args.interval_seconds,
            stop_on_failure=not args.continue_on_failure,
            scope=args.scope,
            spool_limit=args.spool_limit,
            processing_stale_seconds=args.processing_stale_seconds,
            quality_limit=args.quality_limit,
            review_limit=args.review_limit,
            triage_limit=args.triage_limit,
            apply_review=args.apply_review,
            episode_summary=not args.no_episode_summary,
            episode_summary_min_group_size=args.episode_summary_min_group_size,
            episode_summary_session_min_group_size=args.episode_summary_session_min_group_size,
            episode_summary_limit=args.episode_summary_limit,
            candidate_summary=not args.no_candidate_summary,
            candidate_summary_min_group_size=args.candidate_summary_min_group_size,
            candidate_summary_limit=args.candidate_summary_limit,
            governance_probe=not args.no_governance_probe,
            governance_query=args.governance_query,
            governance_max_input_tokens=args.governance_max_input_tokens,
            governance_max_model_cost_usd=args.governance_max_model_cost_usd,
            doctor_query=args.doctor_query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            regression_manifest=args.regression_manifest,
            regression_baseline=args.regression_baseline,
            regression_baseline_drift_warn_only=args.regression_baseline_warn_only,
            run_maintenance_step=not args.no_maintenance,
            vacuum=not args.no_vacuum,
            report_item_limit=args.report_item_limit,
            use_lock=not args.no_lock,
            lock_stale_seconds=args.lock_stale_seconds,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.passed else 1

    if args.cmd == "worker-schedule":
        result = memory.write_worker_schedule(
            output=args.output,
            repo=args.repo.resolve(),
            task_name=args.task_name,
            interval_minutes=args.interval_minutes,
            scope=args.scope,
            python_executable=args.python,
            regression_manifest=args.regression_manifest,
            regression_baseline=args.regression_baseline,
            log_path=args.log_path,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "worker-schedule-verify":
        result = memory.verify_worker_schedule(
            output=args.output,
            scope=args.scope,
            max_interval_minutes=args.max_interval_minutes,
            require_regression_gates=not args.no_regression_gates,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "doctor":
        report = memory.doctor(
            scope=args.scope,
            recall_query=args.query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(report.to_text())
        return 0 if report.passed else 1

    if args.cmd == "health":
        cases = load_recall_regression_cases(args.regression_manifest) if args.regression_manifest else None
        baseline = load_recall_regression_baseline(args.regression_baseline) if args.regression_baseline else None
        report = memory.health(
            scope=args.scope,
            query=args.query,
            recall_budget=args.recall_budget,
            hot_budget=args.hot_budget,
            review_limit=args.review_limit,
            backup_max_age_hours=args.backup_max_age_hours,
            backup_target_bytes=None if args.backup_target_bytes < 0 else args.backup_target_bytes,
            retention_cycle_max_age_hours=args.retention_cycle_max_age_hours,
            regression_cases=cases,
            regression_baseline=baseline,
        )
        if args.json:
            payload = report.as_compact_dict() if args.compact else report.as_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        elif args.compact:
            print(report.to_compact_text())
        else:
            print(report.to_text())
        return 0 if report.passed else 1

    if args.cmd == "backup":
        if args.no_archive and args.archive_mode not in {None, "none"}:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "error": "--no-archive cannot be combined with --archive-mode other than none.",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        result = memory.backup(
            output=args.output,
            include_archive=not args.no_archive,
            archive_mode=args.archive_mode,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "backup-stewardship":
        if args.apply and args.no_cache_write:
            message = "--no-cache-write is only valid for dry-run backup-stewardship."
            if args.json:
                print(json.dumps({"passed": False, "error": message}, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(message, file=sys.stderr)
            return 2
        result = memory.backup_stewardship(
            keep_latest=args.keep_latest,
            keep_retention_cycles=args.keep_retention_cycles,
            target_backup_bytes=args.target_backup_bytes,
            quarantine_failed=args.quarantine_failed,
            apply=args.apply,
            confirm=args.confirm,
            quarantine_confirm=args.quarantine_confirm,
            use_lock=not args.no_lock,
            lock_stale_seconds=args.lock_stale_seconds,
            write_cache=not args.no_cache_write,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "verify-backup":
        result = memory.verify_backup(args.path, trust_root=args.trust_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.cmd == "restore-backup":
        result = memory.restore_backup(args.path, args.target_root, force=args.force, trust_root=args.trust_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.cmd == "restore-drill":
        result = memory.restore_drill(
            args.path,
            scope=args.scope,
            recall_query=args.query,
            recall_budget=args.recall_budget,
            trust_root=args.trust_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.cmd == "cold-export":
        result = memory.cold_export(
            output=args.output,
            scope=args.scope,
            statuses=args.status,
            limit=args.limit,
            include_events=not args.no_events,
            order=args.order,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "verify-cold-export":
        result = memory.verify_cold_export(args.path, trust_root=args.trust_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.cmd == "prune-plan":
        queries = list(args.query)
        if args.query_file:
            queries.extend(
                line.strip()
                for line in args.query_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        result = memory.prune_plan(
            scope=args.scope,
            limit=args.limit,
            export_path=args.cold_export,
            recall_queries=queries,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "shadow-prune":
        queries = list(args.query)
        if args.query_file:
            queries.extend(
                line.strip()
                for line in args.query_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        result = memory.shadow_prune(
            backup_path=args.backup,
            export_path=args.cold_export,
            scope=args.scope,
            limit=args.limit,
            recall_queries=queries,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            doctor_query=args.doctor_query,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "prepare-live-prune":
        queries = list(args.query)
        if args.query_file:
            queries.extend(
                line.strip()
                for line in args.query_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        try:
            result = memory.prepare_live_prune(
                backup_path=args.backup,
                export_path=args.cold_export,
                scope=args.scope,
                limit=args.limit,
                recall_queries=queries,
                recall_budget=args.recall_budget,
                include_global=not args.no_global,
                doctor_query=args.doctor_query,
                ttl_minutes=args.ttl_minutes,
            )
        except ValueError as exc:
            print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "live-prune":
        result = memory.live_prune(approval_token=args.approval_token, confirmation=args.confirm)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "irreversible-operations":
        print(json.dumps(memory.irreversible_operations(limit=args.limit), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.cmd == "maintenance":
        result = memory.maintenance(vacuum=not args.no_vacuum)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.sqlite_integrity == "ok" else 1

    if args.cmd == "archive-encrypt":
        result = memory.archive_encrypt(apply=args.apply, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.cmd == "retention":
        result = memory.retention(scope=args.scope, cold_limit=args.cold_limit)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "cold-stewardship":
        result = memory.cold_stewardship(
            scope=args.scope,
            group_limit=args.group_limit,
            examples_per_group=args.examples_per_group,
            max_cycle_age_hours=args.max_cycle_age_hours,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "cold-map":
        result = memory.cold_map(
            scope=args.scope,
            query=args.query,
            group_limit=args.group_limit,
            examples_per_group=args.examples_per_group,
            budget=args.budget,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "provenance-compact":
        try:
            recall_cases = load_recall_regression_cases(args.recall_manifest) if args.recall_manifest else None
            recall_baseline = (
                load_recall_regression_baseline(args.recall_baseline)
                if args.recall_baseline
                else None
            )
            result = memory.provenance_compaction(
                scope=args.scope,
                keep_events=args.keep_events,
                min_pinned_events=args.min_pinned_events,
                limit=args.limit,
                summary_only=not args.include_non_summary,
                dry_run=not args.apply,
                confirm=args.confirm,
                recall_cases=recall_cases,
                recall_baseline=recall_baseline,
                recall_max_token_growth=args.max_token_growth,
                recall_min_overlap=args.min_overlap,
            )
        except ValueError as exc:
            if args.json:
                print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(f"error: {exc}")
            return 1
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "lifecycle":
        result = memory.lifecycle(
            scope=args.scope,
            limit=args.limit,
            examples_per_tier=args.examples_per_tier,
            target_hot_tokens=args.target_hot_tokens,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "retention-cycle":
        queries = list(args.query)
        if args.query_file:
            queries.extend(
                line.strip()
                for line in args.query_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        result = memory.retention_cycle(
            scope=args.scope,
            backup_output=args.backup_output,
            cold_output=args.cold_output,
            report_output=args.report_output,
            limit=args.limit,
            recall_queries=queries,
            recall_budget=args.recall_budget,
            include_global=not args.no_global,
            doctor_query=args.doctor_query,
            shadow=not args.no_shadow,
            backup_archive_mode=args.backup_archive_mode,
            use_lock=not args.no_lock,
            lock_stale_seconds=args.lock_stale_seconds,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0 if result.passed else 1

    if args.cmd == "candidate-pressure":
        result = memory.candidate_pressure(scope=args.scope, limit=args.limit)
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "candidate-summary":
        result = memory.candidate_summary(
            scope=args.scope,
            pattern=args.pattern,
            min_group_size=args.min_group_size,
            limit=args.limit,
            dry_run=not args.apply,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "conflict-adjudicate":
        result = memory.conflict_adjudicate(
            scope=args.scope,
            limit=args.limit,
            dry_run=not args.apply,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "episode-summary":
        result = memory.episode_summary(
            scope=args.scope,
            pattern=args.pattern,
            min_group_size=args.min_group_size,
            limit=args.limit,
            dry_run=not args.apply,
        )
        if args.json:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(result.to_text())
        return 0

    if args.cmd == "eval":
        report = memory.evaluate()
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.passed else 1

    if args.cmd == "context-eval":
        report = memory.contextual_evaluate()
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.passed else 1

    if args.cmd == "recall-regression":
        cases = load_recall_regression_cases(args.manifest)
        baseline = load_recall_regression_baseline(args.baseline)
        report = memory.recall_regression(
            cases,
            baseline=baseline,
            max_token_growth=args.max_token_growth,
            min_overlap=args.min_overlap,
        )
        if args.write_baseline:
            write_recall_regression_baseline(report, args.write_baseline)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.passed else 1

    if args.cmd == "stats":
        print(json.dumps(memory.stats(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.cmd}")
    return 2


def _read_text_arg(text: str | None, file: Path | None) -> str:
    if text is not None and file is not None:
        raise SystemExit("Use either --text or --file, not both.")
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is not None:
        return text
    if sys.stdin.isatty():
        raise SystemExit("Provide --text, --file, or stdin.")
    return sys.stdin.read()


def _read_json_arg(file: Path | None) -> dict:
    if file is not None:
        raw = file.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            raise SystemExit("Provide --file or JSON on stdin.")
        raw = sys.stdin.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("Turn envelope must be a JSON object.")
    return payload


def _parse_budget_list(value: str) -> list[int]:
    budgets = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        budget = int(item)
        if budget <= 0:
            raise SystemExit("Recall budgets must be positive integers.")
        budgets.append(budget)
    if not budgets:
        raise SystemExit("Provide at least one recall budget.")
    return budgets


def _passed_count(items: object) -> str:
    if not isinstance(items, list):
        return "0/0"
    passed = sum(1 for item in items if isinstance(item, dict) and item.get("passed"))
    return f"{passed}/{len(items)}"


if __name__ == "__main__":
    raise SystemExit(main())
