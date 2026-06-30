# Ara Memory OS Architecture

## Core Claim

Ara Memory OS treats memory as a typed, temporal, auditable substrate.
It deliberately avoids a vectorDB-first design. Similarity search can be added
later, but it is not the center of recall.

## Memory Flow

```text
retain(event)
  -> content fingerprint dedupe
  -> append-only JSONL ledger
  -> SQLite event index + FTS

remember_turn(envelope)
  -> prompt / assistant / command / decision events
  -> sha256-addressed file and image archive
  -> optional worktree capture
  -> consolidation + optional sleep + hot refresh

spool_turn(envelope)
  -> atomic JSON write under .ara-memory/spool/pending
  -> later drain_spool()
  -> recover stale .ara-memory/spool/processing records
  -> remember_turn(envelope)
  -> done/failed archive with original envelope preserved

consolidate()
  -> episode capsule
  -> optional decision/preference/procedure/failure/project/self capsule
  -> temporal edges with provenance

recall(query, scope, budget)
  -> optional hot memory prelude
  -> keyword extraction
  -> graph neighbor lookup
  -> FTS/BM25 capsule search
  -> typed context pack
  -> token budget trimming

audit()
  -> poisoning heuristics
  -> low-confidence stable memories
  -> missing provenance

eval()
  -> synthetic recall quality check
  -> scope isolation check
  -> poisoning quarantine check
  -> hot memory budget check

recall_regression(manifest, baseline)
  -> representative recall cases
  -> expected/forbidden term checks
  -> token growth checks
  -> selected capsule overlap checks

worker(scope)
  -> acquire .ara-memory/locks/worker.lock
  -> drain_spool()
  -> quality(persist=True)
  -> review_worker(dry_run=True by default)
  -> review_triage()
  -> doctor()
  -> optional recall_regression()
  -> maintenance()

worker_loop(iterations, interval)
  -> repeat worker(scope)
  -> compact per-iteration report

sleep()
  -> review candidate memories through advisor interface
  -> promote high-confidence operational memories
  -> merge repeated candidates into summaries
  -> consolidate repeated artifact memories by file/hash identity
  -> supersede stale summary generations
  -> flag possible contradictions
  -> record consolidation run metrics
```

## Storage

- `.ara-memory/ledger/events.jsonl`: append-only source of truth.
- `.ara-memory/memory.db`: SQLite tables, FTS indexes, temporal edges.
- `.ara-memory/archive/`: reserved for cold files, images, and future Memvid-style archives.
- `.ara-memory/archive/objects/`: sha256-addressed raw file and image artifacts.
- `.ara-memory/hot/`: tiny always-on Markdown state files compiled from stable capsules.
- `.ara-memory/spool/`: durable pending/done/failed turn envelopes and enqueue-time snapshots for crash-safe ingress.

## Ingress Boundary

`remember-turn` is the synchronous automation boundary. A Codex skill or wrapper
does not need to understand the database schema; it emits one JSON turn envelope
containing the user prompt, assistant result, files, images, commands, and
explicit decisions. Ara Memory OS then preserves raw evidence first and performs
compression later.

`spool-turn` is the unattended boundary. It writes the same envelope to an
atomic local file queue before touching the memory database, copies existing
file/image artifacts into `spool/snapshots`, and stores a bounded worktree
snapshot inside the envelope. `drain-spool` processes pending envelopes through
`remember-turn`; successful records move to `done`, and failed records move to
`failed` with the original envelope and error details intact. This is the
preferred bridge for always-on capture because a Codex crash, DB lock, missing
artifact, later file edit/delete, or later worker failure does not erase or
rewrite the user's prompt, files, or worktree evidence. If a worker dies after
moving a file into `processing`, the next drain/worker recovers stale processing
records back to `pending` before continuing.

`worker` is the separate memory processor. Codex can keep acting as the live
reasoning agent while the worker drains queued turns and runs quality, review,
triage, doctor, recall-regression, and maintenance gates. The default review
worker mode is dry-run, so background operation can inspect and score memories
without silently rewriting long-term behavior. Review triage groups large queues
by action, reason, status, and kind so humans/Ara can inspect the queue without
reading hundreds of repeated rows. The worker also takes a filesystem lock by
default, so overlapping scheduler invocations skip safely instead of racing over
the same spool and SQLite store.

`worker-loop` is the scheduler-friendly wrapper around `worker`. In production,
an OS scheduler can call it with `--iterations 1`; in a foreground session it can
run multiple iterations with a fixed interval and compact reports.

## Memory Organs

- `self`: Ara principles and durable operating constraints.
- `preference`: Jongseo preferences; should usually start as candidate.
- `project`: repo and worktree context.
- `procedure`: reusable ways of working.
- `failure`: prior broken approaches and hazards.
- `decision`: explicit design or product decisions.
- `episode`: source-level event memory.
- `fact`: durable factual statements.

## Why This Differs From RAG

RAG usually optimizes for "find similar chunks." Ara Memory OS optimizes for:

- what is currently true,
- what used to be true,
- where a claim came from,
- whether a memory is candidate or stable,
- which scope may read it,
- whether using it would help the current task.

## Efficiency Rules

1. Write raw once, read raw rarely.
2. Ingest a whole turn through one envelope instead of many ad hoc commands.
3. Do expensive understanding after the session.
4. Keep hot memory tiny.
5. Compile recall packs by budget before calling a large model.
6. Treat code, prose, terminal output, and images as different compression domains.
7. Put only stable, compressed state into hot memory; keep raw history cold.
8. Queue unattended captures before analysis so ingestion failure does not lose raw evidence.
9. Run recall regression after retrieval, compression, sleep, pruning, or quality changes.
10. Keep the background worker conservative: score and verify by default, mutate only with explicit review.
11. Serialize background workers with a lock; concurrency belongs at the queue boundary, not inside maintenance.
12. Recover stale processing records before draining pending work.
13. Triage review queues by groups before asking a human or model to inspect individual items.
14. Treat cold-memory stewardship as current only when the latest retention-cycle is fresh, matches live cold totals, and proved source-event preservation in shadow-prune.

## Future Extension Points

- Optional local embeddings for intent matching.
- Cross-encoder reranking for high-value recall.
- OCR/caption ingestion for images.
- Memvid-style cold archives for large old sessions.
- Small local curator model for better consolidation.
- Separate auditor model for poisoning and drift checks.
- External-command advisor provider for local models or API wrappers.
- Example advisor wrappers under `examples/`.
