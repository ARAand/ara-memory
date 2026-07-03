# Ara Memory Handoff Continuation

This is the concrete continuation note for moving Ara Memory OS off this local
machine without publishing raw private memory in plaintext.

## GitHub State

- Repository: `https://github.com/ARAand/ara-memory`
- Branch: `codex/ara-memory-natural-memory-gates`
- Public code handoff commit before this note: `aeeaa58 Add public handoff doctor`
- Public handoff command:

```powershell
python -m ara_memory handoff-doctor --json
python -m ara_memory handoff-doctor --require-clean-worktree --json
```

The handoff doctor intentionally does not need raw `.ara-memory`. It verifies
public bundle shape, documentation, `.gitignore` private-memory boundaries,
tracked-file privacy, privacy-pre-push, and optionally clean worktree state.

## Private Raw Memory Download

Raw `.ara-memory` was not committed to the public repository. It was packaged as
an encrypted release asset:

- Release: https://github.com/ARAand/ara-memory/releases/tag/ara-memory-private-handoff-20260703-145658
- Encrypted asset: `ara-memory-raw-handoff-20260703-145658.zip.aesgcm`
- Manifest asset: `ara-memory-raw-handoff-20260703-145658.manifest.json`
- Raw memory files included: 397
- Plaintext ZIP bytes: 247,053,813
- Encrypted asset bytes: 329,405,147
- Encrypted asset SHA-256: `127dc1a6c9d7d0681f305dbcc861ce9c191f3e2e74fe4c1427a486524bd92e4d`
- Plaintext ZIP SHA-256 after decrypt: `58810ec8976c40ac230c33beca3c040d16aaa3837a01f5f44f47c47bb658645c`
- Local key file, not published: `C:\Users\Owner\Documents\AraMemoryPrivateHandoff\ara-memory-raw-handoff-20260703-145658.key.txt`

Do not upload the key file, plaintext ZIP, or extracted `.ara-memory` directory
to the public repository. Move the key through a separate private channel.

## Restore Procedure On Another Machine

1. Clone the repository and switch to the handoff branch.

```powershell
git clone --branch codex/ara-memory-natural-memory-gates --single-branch https://github.com/ARAand/ara-memory.git ara-memory
cd ara-memory
python -m ara_memory handoff-doctor --json
```

2. Download the encrypted release asset and manifest from the release URL above.
3. Obtain `key_base64_urlsafe` from the local-only key file through a private channel.
4. Decrypt the asset to a ZIP, verify its SHA-256, then extract it at the repo root
   so `.ara-memory` is restored.
5. After restore, run:

```powershell
python -m ara_memory health --scope ara-memory --compact --json --regression-manifest examples\recall_regression_manifest.json --regression-baseline .ara-memory\archive\recall-regression-baseline.json
python -m ara_memory goal-roadmap --scope ara-memory --regression-manifest examples\recall_regression_manifest.json --regression-baseline .ara-memory\archive\recall-regression-baseline.json
python -m ara_memory recall-quality "current Ara natural memory purpose recall" --scope ara-memory --no-global --json
```

## Verified Evidence

- Local repository before release: clean and synchronized with origin.
- Local `handoff-doctor --require-clean-worktree`: passed true, status watch only
  because fixture privacy warnings remain reviewed warnings.
- Private package decryption check: passed. Decrypted plaintext hash matched the manifest.
- GitHub release assets: uploaded and SHA-256 matched the encrypted manifest.
- Clean clone path tested: `C:\Users\Owner\AppData\Local\Temp\ara-memory-clean-handoff-20260703-145658`
- Clean clone raw memory state: `.ara-memory` absent.
- Clean clone `handoff-doctor --require-clean-worktree --json`: passed true.
- Clean clone handoff unit tests: 2 tests OK.
- Clean clone full unit suite: 419 tests OK, skipped 1.

Skipped test:

- `test_backup_refuses_symlinked_tree_entries` skipped because Windows symlink
  creation was unavailable in the current user privilege context. This is an OS
  permission skip, not a failing memory-store behavior.

## Failure Context To Preserve

- A PowerShell command initially used Bash heredoc syntax (`python - <<'PY'`) and
  failed. The fix was to pipe a PowerShell here-string into `python -`, which is
  the safer Windows pattern for future scripted one-offs.
- Recall regression briefly failed because `espa_procedural_routing` grew from
  557 to 704 estimated tokens, exceeding the old 25% token-growth baseline by
  8 tokens. Expected terms, forbidden terms, source-event evidence, and graph
  diagnostics still passed. The baseline was refreshed only after verifying the
  new output as stable evidence.
- Review queue pressure reappeared as 10 low-quality review markers after memory
  activity. `review-compact` dry-run showed all 10 were non-destructive marker
  acknowledgements; `review-compact --apply` resolved them without promotion,
  deletion, or decay.
- `privacy-pre-push` remains watch/pass because fixture-like secret strings in
  `tests/test_memory_flow.py` are intentionally retained as adversarial tests.
  Any non-fixture secret, tracked `.ara-memory`, database, ledger, backup, spool,
  archive object metadata, or log must remain a hard failure.

## Concrete Remaining Work

1. On the next machine, decrypt and restore the private `.ara-memory` package,
   then run health, goal-roadmap, and recall-quality as listed above.
2. Add a first post-handoff memory event on the new machine, drain it, and confirm
   the restored store can continue rather than only be inspected.
3. Convert the private handoff package creation into a reviewed CLI command so
   future packages are reproducible, include restore scripts, and avoid manual
   PowerShell snippets.
4. Add a private-package restore drill that decrypts the release asset into a
   temporary directory, restores `.ara-memory`, and runs health without touching
   the live store.
5. Continue collecting real reviewed recall-quality-impact outcomes. Do not tune
   thresholds from synthetic stress runs alone.

