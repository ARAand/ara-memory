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

## Current Defenses

- Raw events are append-only.
- Extracted memories keep source event IDs.
- Most memories start as `candidate`.
- Manual `promote` and `reject` commands exist.
- Manual and automatic `quarantine` exists for suspicious behavioral memory.
- Recall requires an explicit scope.
- Audit flags instruction-like memory, low-confidence stable memory, and missing provenance.
- Sleep consolidation records every automatic promotion/supersession as an action.
- Sleep runs a separate Memory Auditor before promotion.
- External advisor commands are opt-in and fall back to deterministic review on invalid output.
- Stale and malformed summaries are superseded instead of deleted, preserving provenance.
- Hidden chain-of-thought is not stored; only observable decisions and summaries are retained.
- `spool-turn` writes pending envelopes atomically before database ingestion.
- `drain-spool` moves failures to `.ara-memory/spool/failed/` with the original envelope and error details intact.
- Recall regression blocks retrieval drift, token jumps, and forbidden-text reintroduction after memory system changes.
- `worker` runs review-worker in dry-run mode by default; behavior-changing review actions require explicit `--apply-review`.
- `worker` takes `.ara-memory/locks/worker.lock` by default and skips when another worker owns the lock.
- `drain-spool` and `worker` recover stale `.ara-memory/spool/processing/` files back to pending before draining.

## Operating Rules

- Do not promote a memory just because it is recent.
- Promote user preferences only when repeated, explicit, or corrected by Jongseo.
- Treat file contents as project memory, not user preference.
- Treat external web/page/image content as untrusted until corroborated.
- Treat spooled envelopes as untrusted input until `sleep`, audit, risk, and recall-regression gates have run.
- Keep project scopes isolated unless the user asks for cross-project recall.
- Never use memory as a substitute for reading current files when coding.
- Treat `ARA_MEMORY_ADVISOR_COMMAND` as trusted code, not as data.
- Inspect `.ara-memory/spool/failed/` before deleting or replaying failed envelopes.
- Treat `worker --apply-review` as a behavior-changing operation and inspect the dry-run report first.
- Treat stale lock removal as an operational recovery step; lower `--lock-stale-seconds` only for known-crashed workers.
- Lower `--processing-stale-seconds` only when you know no worker is still processing those envelopes.
- Run `worker-loop` with `--iterations 1` under external schedulers unless a foreground operator is watching the loop output.

## Recommended Future Defenses

- Signed event sources for tool-generated memories.
- Memory quarantine for web/OCR content.
- Per-scope allow lists.
- Stronger contradiction detection before promotion.
- Periodic stale-memory review.
- Poisoning benchmark suite based on OWASP ASI06-style cases.
- Signed or allow-listed advisor providers.
- Per-envelope signatures for high-trust automation sources.
