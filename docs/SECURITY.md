# Ara Memory OS Security Model

## Primary Risks

1. Memory poisoning: hostile text becomes durable instruction.
2. Preference overfitting: a temporary user statement becomes permanent.
3. Compression drift: repeated summaries change meaning.
4. Scope leakage: one project recalls another project's private memory.
5. Self-justification: the acting agent records flattering false memories.
6. Provenance collapse: a memory survives after its source is forgotten.
7. External advisor abuse: a configured advisor command returns unsafe promotions.
8. Spool poisoning: unattended queued envelopes contain hostile instructions or stale artifacts.
9. Queue loss: a crash or failed ingest drops a prompt before it reaches the ledger.
10. Background overreach: an automated worker silently promotes bad memories.
11. Worker race: overlapping background runs process the same queue or maintenance cycle.
12. Processing orphan: a crashed worker leaves an envelope in `processing`.
13. Backup blind spot: pending spool work or orphaned FK rows appear healthy but cannot be restored safely.
14. Live-prune drift: the approved cold capsule set changes before irreversible deletion.
15. Evidence drift: queued file/worktree evidence changes between enqueue and drain.
16. Plaintext memory spill: local ledgers, SQLite databases, backups, spool files, hot-memory packs, or cold exports are accidentally committed, copied, or uploaded.

## Current Defenses

- Raw events are append-only.
- Extracted memories keep source event IDs.
- Most memories start as `candidate`.
- Manual `promote` and `reject` commands exist.
- Manual and automatic `quarantine` exists for suspicious behavioral memory.
- The deterministic risk auditor detects instruction-like text, secret-like credential patterns, direct identifiers, self-serving identity claims, and keyword stuffing across capsule title, body, and tags before hot memory is rebuilt.
- The pre-retain privacy guard redacts secret-like credentials, direct identifiers, hidden reasoning text/metadata, and sensitive metadata keys before text events are written to the append-only ledger, SQLite event table, or FTS index. Source and scope labels containing recognized private patterns are replaced with deterministic private labels. `--allow-raw-secret` is the explicit override and records the privacy action in event metadata.
- Instruction-like, secret-like, and direct-identifier tags are suppressed from recall tag surfaces.
- Default recall, manual promotion, quality review, and external advisor payloads reuse that deterministic risk boundary; risky advisor candidates are redacted before any external command receives them.
- Promotion recommendation and the actual `stable` write are separated by a shared promotion gate. Automatic promotion from sleep, review-worker apply mode, external advisors, or summary consolidation requires real provenance: two existing source events, one trusted explicit source, or a summary backed by consolidated source capsules plus at least one existing source event.
- Automatic promotion writes are candidate-only compare-and-set updates, so stale sleep or review-worker decisions cannot overwrite a concurrent quarantine, rejection, or supersession. Summary consolidation also rechecks source capsules before and during supersession.
- Recall requires an explicit scope.
- Audit flags instruction-like memory, low-confidence stable memory, and missing provenance.
- Sleep consolidation records every automatic promotion/supersession as an action.
- Sleep runs a separate Memory Auditor before promotion.
- External advisor commands are opt-in and fall back to deterministic review on invalid output.
- Stale and malformed summaries are superseded instead of deleted, preserving provenance.
- Recognized hidden-reasoning fields and line-prefixed text are redacted from retained text events by default; only observable decisions and summaries should be retained. Archive object payloads are encrypted at rest; spool envelopes/path metadata, legacy snapshots, ledger entries, and SQLite rows remain private local evidence and may still contain plaintext.
- `spool-turn` writes privacy-guarded pending envelopes atomically before database ingestion, seals each envelope with a local HMAC, snapshots existing file/image artifacts as encrypted archive objects, and captures bounded worktree evidence at enqueue time.
- `drain-spool` verifies the local seal, strict option types, bounded sizes, and artifact snapshot hashes before retaining any events; unsealed or edited pending JSON is moved to failed.
- `drain-spool` moves failures to `.ara-memory/spool/failed/` with the sealed guarded envelope and error details intact; malformed queue files keep a raw sidecar there.
- Artifact preflight runs before retaining turn text so a failed drain does not leave partial prompt-only memory or later read an unsnapshotted live artifact path.
- Recall regression blocks retrieval drift, token jumps, and forbidden-text reintroduction after memory system changes.
- `worker` runs review-worker in dry-run mode by default; behavior-changing review actions require explicit `--apply-review`.
- `worker` takes `.ara-memory/locks/worker.lock` by default and skips when another worker owns the lock.
- `drain-spool` and `worker` recover stale `.ara-memory/spool/processing/` files for the requested scope back to pending before draining.
- `worker-schedule-verify` checks generated scheduled-worker scripts before they are treated as installable script-readiness evidence.
- Backups include durable spool envelopes, per-entry SHA-256 hashes, and a manifest HMAC signed by the local `.backup-signing-key`; `verify-backup` first bounds ZIP entry count, entry size, total expansion, compression ratio, and normalized path collisions, then checks entry hashes, manifest signature, SQLite integrity, foreign keys, and encrypted archive-object key escrow when encrypted objects are present. The default `objects` archive profile backs up encrypted artifact objects without recursively embedding older cold exports, retention-cycle reports, or quarantined failed-backup ZIPs; `full` is the explicit forensic profile for the whole archive tree.
- Signed v2 cold exports include per-entry SHA-256 hashes and a manifest HMAC using the same local trust key; `verify-cold-export` bounds archive expansion, streams JSONL with per-file/per-line limits, checks hashes/signature, and rejects exports whose capsules reference source events missing from `events.jsonl`.
- `restore-backup` copies the source ZIP to a temporary snapshot and verifies/extracts that same byte stream; `--force` only clears recognized memory-root paths and preserves local signing keys.
- `live-prune` binds approvals to exact capsule IDs plus backup/export SHA-256 identities, rechecks export coverage from the verified cold-export scan, rechecks current cold status at deletion time, and preserves source events.
- `retention-cycle --no-shadow` is partial evidence only; pruning readiness requires a passing shadow-prune in a restored sandbox.
- `cold-stewardship` treats high cold pressure as current evidence only when the latest retention-cycle is fresh, matches live cold totals, proved source-event preservation, separates active-linked evidence from archive candidates and audit-only rejects, and identifies active memories that pin cold source events.
- `lifecycle` is analysis-only: it separates core, working, guarded, evidence, archive, and reject memory without changing capsule status or deleting provenance.
- Repository ignore rules block local memory roots, SQLite databases, backups, archives, logs, restored memory roots, and Codex-local folders from ordinary `git add .` publishing paths.

## Operating Rules

- Do not promote a memory just because it is recent.
- Do not treat memory as obedience storage; purpose, provenance, review state, and risk decide whether a memory can become hot or stable.
- Do not add a new promotion path unless it calls the shared promotion gate before writing `stable`.
- Promote user preferences only when repeated, explicit, or corrected by Jongseo.
- Treat file contents as project memory, not user preference.
- Treat external web/page/image content as untrusted until corroborated.
- Treat secret-like values as quarantine candidates, not useful memory. Direct identifiers and keyword-stuffed memories must not enter hot memory unless a future reviewed policy explicitly allows it.
- Treat spooled envelopes as untrusted input until `sleep`, audit, risk, and recall-regression gates have run.
- Treat encrypted spool artifact references and legacy `spool/snapshots` files as part of live queued evidence; do not clean them independently from their pending/done/failed envelope.
- Treat `.ara-memory/spool/.seal-key` as private local trust material; backups preserve it so pending sealed envelopes remain drainable after restore.
- Treat `.ara-memory/.archive-object-key` as private local trust material. Backups do not include it in plaintext; encrypted archive-object backups carry a manifest escrow wrapped by `.backup-signing-key`, so copied/restored archive bytes still need the corresponding source trust root.
- Run `archive-encrypt` as a dry-run before applying legacy archive migration; apply only when plaintext objects are expected and corrupt, missing-key, or key-mismatch reports have been reviewed.
- Treat `.ara-memory/.backup-signing-key` as private local trust material; it is not stored in backup ZIPs, so copied backups need the corresponding key to remain cryptographically verifiable.
- Treat `backup --archive-mode full` as an explicit forensic operation. Routine milestone and retention-cycle backups should use the default `objects` profile so signed evidence bundles are not recursively embedded in later backups.
- Treat `backup --no-archive`/`--archive-mode none` as metadata-only for artifacts; it is rejected while active spool records depend on encrypted archive-object snapshots.
- Keep project scopes isolated unless the user asks for cross-project recall.
- Never use memory as a substitute for reading current files when coding.
- Treat `ARA_MEMORY_ADVISOR_COMMAND` as trusted code, not as data.
- Inspect `.ara-memory/spool/failed/` before deleting or replaying failed envelopes.
- Treat `worker --apply-review` as a behavior-changing operation and inspect the dry-run report first.
- Treat stale lock removal as an operational recovery step; lower `--lock-stale-seconds` only for known-crashed workers.
- Lower `--processing-stale-seconds` only when you know no worker is still processing those envelopes.
- Run `worker-loop` with `--iterations 1` under external schedulers unless a foreground operator is watching the loop output.
- Run `worker-schedule-verify` after generating or editing scheduled-worker scripts and before installing them; verify the installed task and recent worker log separately.
- Treat `worker-schedule-verify` as pre-install static evidence only; runtime evidence requires the generated status script plus recent worker log review.
- Treat `live-prune` as irreversible: rerun retention-cycle and prepare-live-prune if any cold capsule status changes after approval.
- Treat stale or drifted cold-stewardship evidence as a watch signal; inspect the evidence/archive/reject tier split and active provenance pins, then rerun retention-cycle before relying on it.
- Treat guarded lifecycle memory as review-required; do not let it enter hot memory simply because it is recent or high-salience.
- Treat `.ara-memory`, backups, cold exports, hot-memory files, archive object metadata, spool envelopes/path metadata, legacy snapshots, and SQLite databases as private evidence. Archive object payloads are encrypted at rest, but backups still include plaintext spool/ledger/database evidence. Keep live memory roots outside public repositories when possible, and run `git status --short` plus `git ls-files` before publishing.

## Recommended Future Defenses

- Signed event sources for tool-generated memories.
- Memory quarantine for web/OCR content.
- Per-scope allow lists.
- Stronger contradiction detection before promotion.
- Periodic stale-memory review.
- Poisoning benchmark suite based on OWASP ASI06-style cases.
- Signed or allow-listed advisor providers.
- A pre-push privacy gate that rejects tracked memory roots, databases, archives, absolute user paths, and secret-like patterns.
- Encrypted backups and cold exports, with raw source events opt-in for portable archives.
