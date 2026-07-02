# Ara Memory OS Architecture

## Core Claim

Ara Memory OS treats memory as a typed, temporal, auditable substrate.
It deliberately avoids a vectorDB-first design. Similarity search can be added
later, but it is not the center of recall.

The working-memory model is an associative substrate rather than a finished
human-like memory. Recall combines explicit purpose, scope, temporal edges,
graph neighbors, FTS/BM25 matches, and budget selection instead of treating
memory as one flat similarity search space. It is still partly
lexical/salience-heavy, but search/render projection split is now the first
compression boundary, deterministic ESPA activation is the first axis router,
and bounded two-hop temporal-edge spreading activation is now part of recall
ranking. There is no normalized relation-node layer, global spreading layer, or
reconsolidation frame yet.

Natural memory means Ara recalls the desired purpose, identity, and task
context without rereading raw history or turning every stored event into
always-on instruction.

The contract is stricter than storage: raw evidence may be retained, but only
reviewed purpose, identity, and preference anchors can become always-on. Working
memory must be selected by query, scope, provenance, recency, risk gates, and
budget. Ara Memory OS is not a secret vault, not source control, not obedience
storage, and not a substitute for reading the current workspace.

## Memory Flow

```text
retain(event)
  -> pre-retain privacy redaction for secrets, direct identifiers, hidden reasoning, metadata keys, and source/scope labels containing recognized private patterns
  -> content fingerprint dedupe
  -> append-only JSONL ledger
  -> SQLite event index + FTS

remember_turn(envelope)
  -> prompt / assistant / command / decision events
  -> sha256-addressed file and image archive
  -> optional worktree capture
  -> bounded synthetic turn episode with evidence ids
  -> consolidation + optional sleep + hot refresh

govern_turn(envelope)
  -> model-free capture plan
  -> agency_review gate for action_allowed and refusal/ask/repair stances
  -> recall_candidates probe over small budgets
  -> working_memory projection only when visible evidence exists
  -> action recommendation: capture, agency-review, working-memory, recall-context, or current evidence
  -> no event retention and no AI API call

spool_turn(envelope)
  -> privacy-guarded atomic JSON write under .ara-memory/spool/pending
  -> local HMAC seal over the pending envelope
  -> later drain_spool()
  -> verify seal and strict option bounds before retention
  -> recover stale .ara-memory/spool/processing records
  -> remember_turn(envelope)
  -> optional post-drain stabilization summaries
  -> done/failed archive with sealed guarded envelope preserved

consolidate()
  -> episode capsule
  -> optional decision/preference/procedure/failure/project/self capsule
  -> temporal edges with provenance

recall(query, scope, budget)
  -> recall_candidates(query, scope)
  -> optional hot memory prelude
  -> typed context pack
  -> render bounded capsule projections instead of raw body text
  -> token budget trimming with soft-budget compaction after enough evidence
  -> count rendered capsules, visible sections, and visible query-term coverage

recall_candidates(query, scope)
  -> keyword extraction
  -> graph neighbor lookup
  -> FTS/BM25 search over compact search projections, not raw capsule bodies
  -> recent-context supplement for explicit temporal queries
  -> deterministic risk filter for quarantined and hot-excluded candidates
  -> deterministic ESPA activation over episodic, semantic, procedural, and affective axes
  -> bounded two-hop temporal-edge spreading activation with query-overlap, derived expansion terms, depth decay, path diagnostics, and graph-source capsule supplementation
  -> cue-matched working-memory-impact boost/penalty with bounded magnitude
  -> mark direct FTS/BM25 matches versus salience fallback/supplements
  -> suppress no-evidence salience fallback bodies
  -> ranked capsule ids and diagnostics without rendering a pack

graph_activation_readiness(scope)
  -> run bounded recall-plans over representative graph-readiness probes
  -> inspect alternatives for live spreading activation, graph edges, boosted capsules, and visible evidence
  -> return pass when at least one probe proves temporal-edge activation actually contributed, otherwise watch with repair guidance
  -> feed milestone-check and goal-roadmap without exposing activation path text as model context

recall_policy(query, scope)
  -> classify recall intent: purpose continuity, working context, distant memory, retention safety, or balanced recall
  -> run bounded recall-plan for active memory evidence
  -> inspect purpose-aware lifecycle tiers for hot/core availability
  -> run cold-map only for distant or retention-safety intents
  -> emit actions such as purpose-check, working-memory, recall-context, cold-map, or retention-cycle
  -> never render cold capsule bodies

recall_policy_impact(query, intent, actions, outcome)
  -> require intent, strategy, and action names from the policy decision actually used
  -> append a note event with recall_policy_impact metadata
  -> project one row per action into recall_policy_impacts
  -> evaluate scope-local outcomes by intent and action
  -> produce review recommendations without automatically mutating policy routing

working_memory(prompt, scope, files, errors)
  -> cue frame from prompt, active files, command errors, constraints, temporal hints
  -> associative recall_candidates over hot memory plus a bounded cold pack
  -> suppress unrelated fallback capsules when there is no direct evidence
  -> project only budget-visible items into Keep / Risk / Next Action
  -> optional working-memory-impact event for outcome feedback

agency_review(prompt, proposed_action, scope)
  -> inspect purpose-check and identity-check before accepting the frame
  -> run recall-policy and working-memory without reading the raw ledger
  -> return a stance: proceed, ask-before-acting, repair-memory-first, ask-or-roadmap, or refuse-or-reframe
  -> split review completion from action permission with action_allowed
  -> optionally append an agency_review audit note
  -> never execute actions or mutate routing automatically

govern_turn(turn)
  -> plan turn ingress without storing the turn
  -> run agency_review before recommending capture, recall, or working-memory actions
  -> block anti-judgment, destructive, or safety-deferred frames from being treated as normal clearance
  -> still keep the operation local and model-free

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
  -> rendered-evidence expected term checks
  -> full-pack forbidden term checks
  -> token growth checks
  -> visible capsule and bounded source-event lineage overlap checks
     (full-set digests plus local provenance recompute when truncated)

health(scope)
  -> doctor + spool + review pressure
  -> candidate/cold retention ratios
  -> latest backup age + read-only backup stewardship dry-run pressure
  -> cold_stewardship when cold pressure is high
  -> retention-cycle freshness and live cold-total drift checks
  -> optional recall_regression

worker(scope)
  -> acquire .ara-memory/locks/worker.lock
  -> drain_spool(scope)
  -> episode_summary() + candidate_summary()
  -> quality(persist=True)
  -> review_worker(dry_run=True by default)
  -> review_triage()
  -> doctor()
  -> optional recall_regression()
  -> maintenance()

review_compact(scope)
  -> resolve missing-capsule and stale queue rows
  -> acknowledge low-quality review markers
  -> acknowledge deterministic artifact-exclusion markers
  -> never promote, quarantine, decay, delete, or otherwise mutate capsules

review_redact(scope)
  -> redact sensitive capsule title/body/tags projections only
  -> preserve source-event links and write digest witnesses
  -> resolve sensitive review markers only after risk re-score clears

cold_map(query, scope)
  -> scan distant cold capsules locally
  -> group by lifecycle tier, status, kind, and title pattern
  -> render redacted examples, counts, source digests, and matched terms
  -> never promote, restore, or print cold capsule bodies

review_worker(scope)
  -> promote/quarantine only gated queue actions
  -> acknowledge only low-risk review markers
  -> keep sensitive-data and identity-policy reviews open

worker_loop(iterations, interval)
  -> repeat worker(scope)
  -> compact per-iteration report

sleep()
  -> review candidate memories through advisor interface
  -> promote high-confidence operational memories
  -> merge repeated candidates into semantic Remember/Use when/Evidence memories
  -> preserve core goal/self/preference kinds during homogeneous merges
  -> consolidate repeated artifact memories by file/hash identity
  -> supersede stale summary generations
  -> flag possible contradictions
  -> record consolidation run metrics
```

## Storage

- `.ara-memory/ledger/events.jsonl`: append-only source of truth.
- `.ara-memory/memory.db`: SQLite tables, active-capsule FTS indexes, temporal edges.
- `.ara-memory/archive/`: derived evidence area; subdirectories have distinct lifecycle policy.
- `.ara-memory/archive/objects/`: sha256-addressed file and image artifacts
  encrypted at rest with the local `.archive-object-key`; default backups include
  encrypted objects plus a manifest key escrow wrapped by the backup signing key,
  never the plaintext archive key.
- `.ara-memory/archive/cold/`: signed cold capsule exports used as pruning evidence.
- `.ara-memory/archive/retention-cycles/`: compact retention-cycle reports.
- `.ara-memory/archive/failed-backups/`: quarantined failed backup ZIPs kept for manual inspection.
- Default backups include `archive/objects` but not derived archive bundles such
  as older cold exports, retention-cycle reports, or quarantined failed-backup
  ZIPs. Use `backup --archive-mode full` only for an explicit forensic snapshot
  of the whole archive tree.
- `.ara-memory/hot/`: tiny always-on Markdown state files compiled from stable capsules.
- `.ara-memory/spool/`: durable pending/done/failed turn envelopes, encrypted
  archive-object snapshot references, and legacy enqueue-time snapshots for
  crash-safe ingress.
- `.ara-memory/.backup-signing-key`: local HMAC trust key for backup manifests;
  it is intentionally excluded from backup ZIPs.
- `.ara-memory/.archive-object-key`: local AES-GCM archive object key; it is
  not written to backup ZIPs in plaintext. Backups that include encrypted
  archive objects carry an encrypted escrow in `manifest.json` so verified
  restores can recover artifact bytes with the source trust root.
- Large provenance event lookups are de-duplicated and chunked so cold export,
  risk review, provenance compaction, and restore/prune preparation do not hit
  SQLite variable limits.
- Temporal edges are also a bounded recall substrate. `source_capsule_id`,
  subject/predicate/object text, confidence, and scope can supplement and boost
  candidate capsules when edge text overlaps the current query. A second hop can
  derive capped expansion terms from first-hop edge text and fetch one additional
  edge layer with depth decay. This is capped, risk-filtered, and diagnostic;
  normal pack output exposes aggregate activation diagnostics rather than raw
  edge path text. It does not mutate capsule status or replace future normalized
  graph nodes.

Cold memory is intentionally outside the active recall FTS index. `cold-map`
provides the distant-memory layer between active recall and archival
maintenance: it can search cold titles/tags/bodies locally, but the rendered
output is only a small navigation map with tier labels, redacted examples,
source-event counts, and digests. A cold-map hit is a cue to inspect, export, or
audit a narrow group; it is not permission to treat cold text as active memory.

## Ingress Boundary

`remember-turn` is the synchronous automation boundary. A Codex skill or wrapper
does not need to understand the database schema; it emits one JSON turn envelope
containing the user prompt, assistant result, files, images, commands, and
explicit decisions. Ara Memory OS applies the pre-retain privacy guard before
text events reach the ledger, SQLite, or FTS, then preserves bounded local
evidence and performs compression later.

`spool-turn` is the unattended boundary. It writes a privacy-guarded copy of the
envelope to an atomic local file queue before touching the memory database,
stores existing file/image artifacts as encrypted `archive/objects`, and stores
a bounded worktree snapshot inside the envelope. It also seals the pending JSON
with a local HMAC key stored under the memory root. `drain-spool` verifies the
seal, strict option types, integer bounds, and enqueue-time artifact snapshots
before decrypting encrypted artifact bytes in memory for `remember-turn`;
successful records move to `done`, and failed records move to `failed` with the
sealed guarded envelope and result/error details intact. Spool envelopes,
snapshot path metadata, explicit raw overrides, and legacy `spool/snapshots`
files remain private local evidence. This is the
preferred bridge for always-on capture because a Codex crash, DB lock, missing
artifact, later file edit/delete, or later worker failure does not erase or
rewrite the user's prompt, files, or worktree evidence. If a worker dies after
moving a file into `processing`, the next drain/worker recovers stale processing
records for that same scope back to `pending` before continuing.

When the foreground session needs to recall immediately after draining, use
`drain-spool --stabilize`. It performs the same conservative episode/candidate
summary folds that the worker uses for repeated command, file, git-status,
bounded session narrative, and memory-policy decision noise, then refreshes hot
memory for the drained scopes. This keeps queued turn preservation crash-safe while
preventing fresh operational evidence from destabilizing the next recall pack.

`worker` is the separate memory processor. Codex can keep acting as the live
reasoning agent while the worker drains queued turns for its scope and runs quality, review,
triage, doctor, recall-regression, and maintenance gates. The default review
worker mode is dry-run, so background operation can inspect and score memories
without silently rewriting long-term behavior. Review triage groups large queues
by action, reason, status, and kind so humans/Ara can inspect the queue without
reading hundreds of repeated rows. The worker also takes a filesystem lock by
default, so overlapping scheduler invocations skip safely instead of racing over
the same spool and SQLite store.

External advisor providers receive redacted candidate projections when the
deterministic auditor has already marked a memory as unsafe for promotion or hot
recall.

`worker-loop` is the scheduler-friendly wrapper around `worker`. In production,
an OS scheduler can call it with `--iterations 1`; in a foreground session it can
run multiple iterations with a fixed interval and compact reports.
`worker-schedule` writes reviewable Windows Task Scheduler install/status/
uninstall scripts, and `worker-schedule-verify` checks that those scripts use
structured one-shot worker-loop arguments, prevent overlapping scheduled
instances, keep valid recall regression gates attached, and stay inside the
configured interval budget. This is script-readiness evidence, not proof that
the task is installed or recently succeeded.

## Promotion Boundary

`stable` is the first memory state that can directly steer future behavior, so
candidate promotion is a boundary, not just a score threshold.

```text
promotion recommendation
  -> reload capsule and source events
  -> deterministic risk gate
  -> automatic provenance gate
  -> candidate-only compare-and-set write
  -> memory_actions actor/reason log + hot invalidation
```

Manual promotion uses the shared deterministic risk gate but remains an
explicit operator decision. Automatic promotion from sleep, review-worker apply,
or an external advisor additionally requires either two existing source events,
one trusted explicit source, or a summary backed by consolidated source
capsules plus at least one existing source event. Source IDs that do not
resolve to event rows block automatic promotion, so provenance cannot be forged
by editing a capsule payload.

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
7. Put only stable, compressed, low-risk state into hot memory; keep raw history cold.
8. Queue unattended captures before analysis so ingestion failure does not lose raw evidence.
9. Run recall regression after retrieval, compression, sleep, pruning, or quality changes.
10. Keep the background worker conservative: score and verify by default, mutate only with explicit review.
11. Serialize background workers with a lock; concurrency belongs at the queue boundary, not inside maintenance.
12. Recover stale processing records before draining pending work.
13. Stabilize drained turns before immediate recall when the background worker has not run yet.
14. Keep full cold evidence in SQLite/ledger/archive, but keep FTS recall indexes limited to candidate and stable capsules.
15. Triage review queues by groups before asking a human or model to inspect individual items.
16. Treat cold-memory stewardship as current only when the latest retention-cycle is fresh, matches live cold totals and the cold identity fingerprint, proved source-event preservation in shadow-prune, and shows which cold capsules are active-linked evidence, archive candidates, or audit-only rejects, plus which active memories are pinning cold source events.
16a. Do not force a new retention-cycle for protected-only drift. If fresh capture creates additional active-linked cold evidence but the prunable source-event fingerprint is unchanged, keep the retention signal green and report the drift explicitly; same-count identity drift or prunable-set drift must still warn.
17. Rotate redundant verified backups with `backup-stewardship` against a target backup-byte budget before touching live memory or cold evidence; keep latest backups, retention-cycle evidence, failed-verification backups, and active live-prune approval backups.
18. Use purpose-aware lifecycle tiers before recall or pruning: long-running purpose, identity, and preference anchors may enter hot memory, working memories require query selection, guarded memories require review, and cold evidence requires export/prune gates.
19. Use `cold-map` as distant-memory navigation, not cold recall. It may scan cold text locally, but it renders only redacted group cues, tier/source counts, and source digests so old evidence can be located without reactivating or printing cold bodies.
20. Use `recall-policy` as the purpose-aware recall controller before spending context. It should classify intent, choose hot/core, working-memory, recall-context, cold-map, or retention-cycle actions, and keep distant cold bodies out of model context unless a separate audited export path is chosen.
21. Record `recall-policy-impact` as bounded audit feedback, not reward optimization. It must use the intent, strategy, and actions from the policy decision actually used in the turn, not a later recomputation. Helpful/harmful/unknown outcomes can explain routing quality and create review recommendations, but they must not automatically mutate intent classification, hot eligibility, pruning gates, or policy commands. Roadmap gates evaluate this evidence per scope and require multiple reviewed outcomes before pass/fail trust.
22. Verify scheduled-worker scripts before installation; treat installed always-on maintenance as a separate operational gate.
23. Compact active summary provenance before expecting cold pressure to fall, but only from a reviewed dry-run. The retained source sample is deterministic and quality-aware: it prefers decision, verification, regression, health, backup, retention, identity, purpose, and risk evidence while preserving temporal coverage. Apply uses one transaction with compare-and-set checks on capsule status, source links, event existence, and provenance witness creation, so a stale or malformed plan rolls back without partial link rewrites. Witness rows preserve the original source-event set and digest while active direct links are shortened. When recall-regression manifest/baseline inputs are supplied, baseline-visible and baseline-selected capsules are protected, then apply simulates the remaining rewrite in a shadow store. If the simulation shifts a reviewed recall path, unstable candidates are elided and retried; live mutation is blocked when no recall-stable plan remains. This operation changes active provenance links, not memory text or promotion status, and must be followed by cold-stewardship, health, and recall-regression before retention-cycle or pruning decisions.
24. Redact sensitive active capsule projections through reviewed `review-redact` dry-runs, not by deleting source evidence. The operation updates only title/body/tags with compare-and-set checks, records original/redacted digests in `capsule_redaction_witnesses`, keeps source-event links intact, and resolves the review marker only after a fresh risk score no longer detects sensitive projection text.
25. Treat temporal recall as explicit: only temporal words or phrases should
    trigger recency boosts, while ordinary substrings such as `knowledge` or
    `blast` must stay lexical.
26. Treat working memory as a cue-led action pack, not another cache: it should
    change the next step, record whether it helped, and stay empty when recall
    has no visible evidence.
27. Treat feedback as bounded evidence, not reward maximization: working-memory
    impact rows may nudge ranking only when their cue overlaps the current
    query, and the boost or penalty is capped. Recall-policy-impact rows are
    audit evidence only and must not nudge routing automatically.
28. Treat self-directed agency as a review gate, not an executor. `agency-review`
    may refuse or reframe an anti-judgment prompt, ask before irreversible work,
    or require purpose/identity repair before proceeding, but it must not launch
    actions, mutate policy, or claim free will without visible evidence.
    `passed` means the review had enough evidence to complete; `action_allowed`
    is the automation gate and is true only for proceed.
29. Store agency-review records as audit notes, not decision memories. They may
    explain a turn but must not become self-certifying evidence that future
    agency reviews use to prove Ara's judgment.
30. Prefer projection and routing over rereading raw memory. Raw body, search
    projection, and render projection are now split; deterministic ESPA
    activation routes recall through episodic, semantic, procedural, and
    affective axes before ranking, and bounded temporal-edge spreading activation
    can add evidence-backed graph-source candidates without rendering raw memory.
    Affective signals are caution/context signals, not reward or promotion scores.

## Future Extension Points

- Global spreading activation beyond the bounded two-hop prototype, normalized
  relation nodes, and ESPA-aware path weighting.
- Optional local embeddings for intent matching.
- Cross-encoder reranking for high-value recall.
- Impact feedback analytics for drift, overfitting, and stale helpfulness.
- OCR/caption ingestion for images.
- Memvid-style cold archives for large old sessions.
- Small local curator model for better consolidation.
- Separate auditor model for poisoning and drift checks.
- External-command advisor provider for local models or API wrappers.
- Example advisor wrappers under `examples/`.
