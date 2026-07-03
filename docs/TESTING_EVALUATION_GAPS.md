# Ara Memory Testing And Evaluation Gaps

This document is the public handoff checklist for the current Ara Memory OS state.
It intentionally does not include raw `.ara-memory` ledgers, databases, backups,
spool files, hot packs, or archived private evidence.

## Current Gate Status

- Health gate: pass. Run `python -m ara_memory health --scope ara-memory --compact --json --regression-manifest examples\recall_regression_manifest.json --regression-baseline .ara-memory\archive\recall-regression-baseline.json`.
- Recall regression: pass. Run `python -m ara_memory recall-regression --manifest examples\recall_regression_manifest.json --baseline .ara-memory\archive\recall-regression-baseline.json`.
- Long-run stress trend: watch. Three recorded samples exist, with no leaks, no critic failures, score delta 0, and token growth delta 0.
- Recall quality: watch. Recall critic and policy route are usable, but reviewed recall-critic and working-memory outcome evidence is still too thin.
- Privacy pre-push: watch/pass. Existing warnings are fixture-like test strings; any private memory root, database, ledger, backup, spool, hot pack, archive, or non-fixture secret must fail publication.

## Missing Tests And Evaluation Methods

1. Reviewed recall-critic outcomes

   Method: after real turns where recall was used, run `recall-critic-impact` with the actual query, critic status, decision labels, outcome, and helped flag. Gate with `recall-critic-impact-eval --min-evaluated 3`.

2. Reviewed working-memory outcomes

   Method: after real turns where working memory projected capsule IDs, run `working-memory-impact` with the cue, projected capsule IDs, outcome, and helped flag. Gate with `working-memory-impact-eval --min-evaluated 3`.

3. Long-run stress pass evidence

   Method: keep recording `long-run-stress --record` from worker rehearsal, but do not treat repeated watch samples as pass. The stress trend should move to pass only after the underlying impact outcome gates pass.

4. Cross-machine handoff verification

   Method: clone the GitHub repository on a clean machine, run the test suite, generate `public-memory-bundle`, and verify that the bundle explains current memory state without needing raw local `.ara-memory` evidence.

5. Public bundle privacy regression

   Method: run `privacy-pre-push`, scan `docs/public-memory-bundle.md` and `.json` for secret-like strings, and verify the bundle contains only redacted recall evidence plus aggregate coverage metrics.

6. Goal continuity after handoff

   Method: on the new machine, run `recall-quality "current Ara natural memory purpose recall" --scope ara-memory`. It may remain watch until local reviewed outcomes accumulate, but it must preserve purpose, identity, and evaluation gaps.

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
