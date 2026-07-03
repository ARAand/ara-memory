# Ara Memory Testing And Evaluation Gaps

This document is the public handoff checklist for the current Ara Memory OS state.
It intentionally does not include raw `.ara-memory` ledgers, databases, backups,
spool files, hot packs, or archived private evidence.

## Current Gate Status

- Health gate: pass. Run `python -m ara_memory health --scope ara-memory --compact --json --regression-manifest examples\recall_regression_manifest.json --regression-baseline .ara-memory\archive\recall-regression-baseline.json`.
- Recall regression: pass. Run `python -m ara_memory recall-regression --manifest examples\recall_regression_manifest.json --baseline .ara-memory\archive\recall-regression-baseline.json`.
- Long-run stress trend: pass. Six recorded samples exist; latest worker-loop run passed with score 100, no leaks, no critic failures, harmful impact ratio 0, and no token growth.
- Recall quality: pass. Recall critic, policy route, reviewed impact feedback, and long-run stress trend are all passing for the current purpose-recall gate.
- Review queue hygiene: pass. `review-compact` resolved 123 non-destructive low-quality review markers, leaving open review queue count at 0 without promoting, deleting, or decaying capsules.
- Privacy pre-push: watch/pass. Existing warnings are fixture-like test strings; any private memory root, database, ledger, backup, spool, hot pack, archive, or non-fixture secret must fail publication.

## Missing Tests And Evaluation Methods

1. Reviewed recall-critic outcomes

   Current status: first pass threshold reached locally. Continue after real turns where recall was used by running `recall-quality-impact --apply` with the actual query, reviewed outcome, and helped flag. Gate with `recall-critic-impact-eval --min-evaluated 3`.

2. Reviewed working-memory outcomes

   Current status: first pass threshold reached locally. Continue after real turns where working memory projected capsule IDs by running `recall-quality-impact --apply` with the cue, reviewed outcome, and helped flag. Gate with `working-memory-impact-eval --min-evaluated 3`.

3. Long-run stress pass evidence

   Current status: pass after reviewed impact outcome gates became available. Keep recording `long-run-stress --record` from worker rehearsal; do not tune thresholds from this alone.

4. Cross-machine handoff verification

   Method: clone the GitHub repository on a clean machine, run the test suite, generate `public-memory-bundle`, and verify that the bundle explains current memory state without needing raw local `.ara-memory` evidence.

5. Public bundle privacy regression

   Method: run `privacy-pre-push`, scan `docs/public-memory-bundle.md` and `.json` for secret-like strings, and verify the bundle contains only redacted recall evidence plus aggregate coverage metrics.

6. Goal continuity after handoff

   Method: on the new machine, run `recall-quality "current Ara natural memory purpose recall" --scope ara-memory`. It should preserve purpose, identity, and evaluation gaps. It may return watch on a fresh local store until reviewed outcome evidence accumulates there.

7. Outcome capture ergonomics

   Method: first run `recall-quality-impact QUERY --scope ara-memory --outcome "reviewed result" --helped unknown` as a dry-run. After the result is reviewed, rerun with `--apply` and `--helped true` or `--helped false`. This avoids rewarding the system before the real outcome is known.

## Publication Boundary

Raw local memory is private evidence. Publish these instead:

- `docs/public-memory-bundle.md`
- `docs/public-memory-bundle.json`
- `docs/TESTING_EVALUATION_GAPS.md`
- implementation code and tests

Do not publish these:

- `.ara-memory/`
- SQLite databases
- ledgers
- backups
- cold exports
- hot memory files
- spool envelopes
- archive object metadata
- logs containing private local evidence
