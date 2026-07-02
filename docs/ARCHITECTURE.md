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
and bounded two-hop relation spreading activation is now part of recall ranking.
A conservative semantic relation merge dry-run gate can now report likely
relation aliases without mutating the graph. A prepare gate can freeze reviewed
candidates behind a short-lived approval token, relation fingerprint, and
rollback witness preview. A one-use apply gate can consume that token only when
the live dry-run fingerprint still matches, merge the approved relation-node
alias in one transaction, and preserve rollback witnesses. A review gate can
now audit applied merge witnesses for candidate remnants, remaining edge
references, self-loops, duplicate coalescing, and evidence-count preservation
without mutating memory; with regression inputs, that same review command runs
the reviewed recall-regression sandbox before returning success. Watch/fail
items can now be persisted to a relation-specific review queue so later workers
do not need to infer relation risk from capsule review rows. A read-only global
spreading sandbox now probes cross-scope graph activation for bounded fanout,
supplementation, depth, visible evidence, quality, and risk-filter pressure
before broader global spreading is trusted. A reconsolidation frame now
recontextualizes a query into purpose/identity anchors, settled decisions,
failure/conflict checks, working context, and forgetting boundaries. The first
apply path is candidate-only: prepare freezes a reviewed frame behind a
fingerprint and short-lived token, apply consumes it once, writes one summary
candidate, and records a witness for review before any stronger memory rewrite
exists.

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
  -> projection_gate verifies visible/projected overlap, action coverage, and token budget
  -> optional reviewed working-memory-impact event for projected capsule ids
  -> action recommendation: capture, agency-review, working-memory, recall-context, or current evidence
  -> no event retention by default and no AI API call

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
  -> explicit Decision: evidence stays decision context, not failure memory just because it names blocked gates or recall regression
  -> successful wrapped Turn episode assistant outcomes stay progress evidence even when Prompt cue text precedes the outcome
  -> successful command markers such as -> passed, => passed, changed=0, and fail=0 do not become failure memory
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
  -> bounded two-hop relation spreading activation with query-overlap, derived expansion terms, depth decay, path diagnostics, and graph-source capsule supplementation
  -> cue-matched working-memory-impact boost/penalty with bounded magnitude
  -> mark direct FTS/BM25 matches versus salience fallback/supplements
  -> suppress no-evidence salience fallback bodies
  -> ranked capsule ids and diagnostics without rendering a pack

graph_activation_readiness(scope)
  -> run bounded recall-plans over representative graph-readiness probes
  -> inspect alternatives for live spreading activation, graph edges, boosted capsules, and visible evidence
  -> return pass when at least one probe proves temporal-edge activation actually contributed, otherwise watch with repair guidance
  -> feed milestone-check and goal-roadmap without exposing activation path text as model context

global_spreading_sandbox(scope)
  -> run bounded recall-plans with include_global=True over representative cross-scope probes
  -> inspect fanout, graph supplementation, max depth, visible evidence, quality, fallback, and risk-filter diagnostics
  -> fail only when global spreading becomes unsafe, such as excess fanout, excess supplementation, or paths beyond the configured depth
  -> return watch when global evidence is absent or low-confidence so roadmap can show capability pressure without inventing safety

recall_policy(query, scope)
  -> classify recall intent: purpose continuity, working context, distant memory, retention safety, or balanced recall
  -> run bounded recall-plan for active memory evidence
  -> inspect purpose-aware lifecycle tiers for hot/core availability
  -> run cold-map only for distant or retention-safety intents
  -> emit actions such as purpose-check, working-memory, recall-context, cold-map, or retention-cycle
  -> attach prior recall-policy-impact rows as Policy Feedback for the proposed actions
     (helpful explains confidence, harmful lowers the route to watch, unknown remains diagnostic)
  -> never render cold capsule bodies

recall_policy_impact(query, intent, actions, outcome)
  -> require intent, strategy, and action names from the policy decision actually used
  -> append a note event with recall_policy_impact metadata
  -> project one row per action into recall_policy_impacts
  -> evaluate scope-local outcomes by intent and action
  -> produce review recommendations without automatically mutating policy commands,
     hot eligibility, pruning gates, or intent classification

working_memory_impact_eval(scope)
  -> read working_memory_impacts joined to source events
  -> group reviewed outcomes by capsule id and source
  -> report pass/watch/fail for helpful versus harmful cue-matched habits
  -> recommend review for capsules with harmful >= helpful
  -> never mutate ranking, memory status, policy routes, or review queues

reconsolidation_review(scope)
  -> review candidate-only reconsolidation witnesses for unchanged evidence capsules
  -> compare the approval rollback witness preview against the approved frame fingerprint
  -> compare the created frame capsule against the witness after-snapshot for kind, status, scope, source links, and content digests
  -> keep the created frame capsule candidate-only and fail review if it drifted after witness creation
  -> optionally run recall-regression as a sandbox over representative recall cases
  -> optionally persist fail/watch items into reconsolidation_review_queue with --record-queue
  -> store failed rows as block-strong-reconsolidation so health and goal-roadmap can gate stronger live mutation
  -> resolve open queue rows when the same witness later passes review
  -> block stronger reconsolidation when witness review, recall regression, or open blocker queue rows fail

reconsolidation_strong_preflight(backup, query, scope)
  -> verify backup and restore into a temporary shadow memory root
  -> run prepare/apply/review and optional recall-regression only in the restored copy
  -> redact the shadow approval token from output
  -> evaluate promote/rewrite/delete/cool as explicit action gates
  -> report design_ready separately from live_authorized for each requested action
  -> never mutate the live memory store
  -> treat pass as design evidence for a future stronger gate, not live mutation approval

prepare_live_reconsolidation_action(backup, query, action, confirm)
  -> require exact PREPARE STRONG RECONSOLIDATION ACTION confirmation
  -> rerun strong preflight for exactly one promote/rewrite/delete/cool action
  -> require design_ready=true and live_authorized=false
  -> store a short-lived one-use approval token plus backup identity, preflight, and action gate evidence
  -> never mutate the live memory store and never consume the token itself
  -> leave actual live execution closed until a future executor rechecks compare-and-set and rollback/exception witnesses

live_reconsolidation_cool(approval_token, capsule_id, confirm)
  -> consume only a prepared cool action approval
  -> require the exact ENABLE STRONG RECONSOLIDATION COOL confirmation recorded by the action gate
  -> recheck approval status, expiry, backup identity, backup verification, and action_gate invariants
  -> block core lifecycle anchors
  -> compare-and-set exactly one active target capsule to superseded
  -> preserve capsule text and source events and write reconsolidation_action_witnesses

reconsolidation_action_witnesses(scope, action, witness_id)
  -> review live action witnesses against an explicit per-action field-transition policy
  -> require current cool witnesses to show only status: candidate|stable -> superseded
  -> fail if target capsule state drifts after the witness or if approval/token evidence is missing
  -> act as the common witness-review floor before any future rewrite, promote, or delete executor

reconsolidation_action_shadow_rollback(backup, scope, action, witness_id)
  -> restore a verified backup into a temporary shadow store
  -> require action witness review to pass inside the shadow store first
  -> roll back only the exact recorded cool status transition: superseded -> prior candidate/stable
  -> run doctor after rollback and leave the live memory store unchanged
  -> provide rollback evidence, not live rollback authorization

prepare_live_reconsolidation_action_rollback(backup, action_witness_id)
  -> rerun action shadow rollback for exactly one witness and require it to pass
  -> recheck the live action witness review and backup identity after the shadow proof
  -> store a short-lived one-use approval token plus the frozen capsule snapshot
  -> never mutate live capsules and never consume the token itself

reconsolidation_action_rollback_approvals(scope, action, approval_id, witness_id)
  -> review prepared action rollback approvals without consuming their tokens
  -> recheck approval status/expiry, backup identity, backup verification, live witness review, and target snapshot drift
  -> fail stale approvals before live action rollback can consume them
  -> never mutate live capsules and never write rollback witnesses

live_reconsolidation_action_rollback(approval_token, confirm)
  -> require exact confirmation and one prepared action rollback approval token
  -> rerun reconsolidation_action_rollback_approvals for that exact approval before mutation
  -> compare-and-set only the recorded cool status transition: superseded -> prior candidate/stable
  -> write reconsolidation_action_rollback_witnesses and memory_actions evidence
  -> mark the rollback approval used, invalidate hot memory, and run doctor

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
  -> run projection_gate so working memory must be visible, action-bearing, and within budget before being treated as clean next-action context
  -> optionally record projection outcome as working-memory-impact only when an explicit reviewed outcome is supplied
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
  -> relation-merge and reconsolidation review queue pressure
  -> candidate/cold retention ratios
  -> latest backup age + read-only backup stewardship dry-run pressure
  -> cold_stewardship when cold pressure is high
  -> retention-cycle freshness and live cold-total drift checks
  -> optional recall_regression
  -> compact projection for worker logs and low-token status handoffs

worker(scope)
  -> acquire .ara-memory/locks/worker.lock
  -> drain_spool(scope)
  -> episode_summary() + candidate_summary()
  -> quality(persist=True)
  -> review_worker(dry_run=True by default)
  -> review_triage()
  -> relation_merge_review(record_queue=True)
  -> relation_review_queue(open pressure)
  -> reconsolidation_review(record_queue=True)
  -> reconsolidation_review_queue(open pressure)
  -> doctor()
  -> optional recall_regression()
  -> maintenance()

review_compact(scope)
  -> resolve missing-capsule and stale queue rows
  -> acknowledge low-quality review markers
  -> acknowledge deterministic artifact-exclusion markers
  -> compact projection groups changes by action/reason for low-token review
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
- `.ara-memory/memory.db`: SQLite tables, active-capsule FTS indexes, temporal edges, and normalized relation graph tables.
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
- Temporal edges remain provenance-like relation evidence. On write and schema
  migration, their subject/object/predicate strings are normalized into
  `relation_nodes` and `relation_edges`. Recall prefers those relation edges for
  graph activation, then falls back to raw temporal edges if the relation graph
  has no match. A second hop can derive capped expansion terms from first-hop
  edge text and fetch one additional edge layer with depth decay. This is capped,
  risk-filtered, and diagnostic; normal pack output exposes aggregate activation
  diagnostics rather than raw edge path text. `global-spreading-sandbox` reuses
  those aggregate diagnostics with global memory enabled and blocks unsafe
  global fanout before broader cross-scope spreading policy is trusted. The `relation-merge` command can
  inspect likely relation-node aliases as a dry-run only. The
  `relation-merge-prepare` command stores a hashed approval token, candidate
  snapshot, relation fingerprint, and rollback witness preview for future
  reviewed apply work. The `relation-merge-apply` command requires exact
  confirmation, rejects expired/reused/drifted approvals, rewires only approved
  relation edges, coalesces duplicate relation edges, deletes the merged alias
  node, and records `relation_merge_witnesses` in the same transaction. The
  `relation-merge-review` command reads those witnesses as a non-destructive
  impact audit and can run recall-regression manifest/baseline checks as a
  sandbox before broader review-worker automation is trusted. With
  `--record-queue`, watch/fail witness items and regression failures are written
  to `relation_merge_review_queue`, while passing re-reviews resolve stale open
  rows for the same witness or regression gate. Health includes the same queue
  as `relation_review_pressure`, so failed open rows stop the health gate before
  further graph merges, open watch rows stay visible as warnings, and an empty
  queue is explicitly OK. None of these commands mutate capsule status or source
  events.

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

`privacy-pre-push` is the publication boundary. It does not inspect the live
memory store directly; it inspects the Git repository about to be published. The
gate fails when tracked or staged paths include private memory roots, ledgers,
databases, backups, archives, logs, raw hidden-reasoning markers, or secret-like
text outside approved test/docs fixtures. This keeps local-first evidence and
public implementation code on different sides of the architecture.

When the foreground session needs to recall immediately after draining, use
`drain-spool --stabilize`. It performs the same conservative episode/candidate
summary folds that the worker uses for repeated command, file, git-status,
bounded session narrative, and memory-policy decision noise, then refreshes hot
memory for the drained scopes. This keeps queued turn preservation crash-safe while
preventing fresh operational evidence from destabilizing the next recall pack.

`worker` is the separate memory processor. Codex can keep acting as the live
reasoning agent while the worker drains queued turns for its scope and runs quality, review,
triage, relation-merge witness review, reconsolidation witness review, doctor, recall-regression, and maintenance
gates. The default review worker mode is dry-run, so background operation can
inspect and score memories without silently rewriting long-term behavior. Review
triage groups large queues by action, reason, status, and kind so humans/Ara can
inspect the queue without reading hundreds of repeated rows. Relation-merge
review is narrower: the worker does not apply new graph merges, but it reruns the
applied witness audit, records watch/fail rows in `relation_merge_review_queue`,
resolves stale passing rows, and fails the worker when an open failed relation
review remains. Reconsolidation review follows the same conservative shape: the
worker does not open stronger live mutation, but it reruns candidate-frame witness
review, records watch/fail rows in `reconsolidation_review_queue`, resolves stale
passing rows, and fails when an open blocker would make stronger reconsolidation
unsafe. The worker also takes a filesystem lock by default, so overlapping
scheduler invocations skip safely instead of racing over
the same spool and SQLite store.

`reconsolidation-shadow-rollback` is the first rollback executor. It verifies a
backup, restores it into a temporary memory root, selects applied
reconsolidation witnesses, checks that evidence capsule digests still match the
rollback witness, then compare-and-set rejects only the candidate frame capsule
created by the witness. Source events, approvals, evidence capsules, and witness
rows remain intact for audit. The restored shadow store must still pass doctor
before the rollback report passes. `prepare-live-reconsolidation-rollback` then
turns exactly one passing shadow rollback for exactly one witness into a
short-lived one-use token. `live-reconsolidation-rollback` rechecks backup
identity, backup verification, witness invariants, and the approved candidate
capsule digest before it rejects that single live candidate frame and writes a
`reconsolidation_rollback_witnesses` row. This opens rollback for bad candidate
frames without opening promotion, rewrite, deletion, or broad reconsolidation
mutation. `reconsolidation-exception-witness` is the narrow escape hatch for
intended drift: it records the witness id, capsule id, field, reason, and exact
before/after digest transition. Review and rollback checks only accept that
specific transition, so another edit to the same capsule must produce another
reviewed witness instead of inheriting blanket trust.

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
promotion_candidates(scope)
  -> quality score candidate capsules
  -> rerun shared deterministic promotion gate
  -> split ready versus blocked high-quality candidates
  -> read-only report; no capsule status mutation

promotion recommendation
  -> reload capsule and source events
  -> deterministic risk gate
  -> automatic provenance gate
  -> candidate-only compare-and-set write
  -> memory_actions actor/reason log + hot invalidation

review_worker_apply(scope)
  -> reload open memory_review_queue row and capsule
  -> recompute quality/risk action
  -> promotion provenance gate or quarantine risk gate
  -> compare-and-set status write
  -> memory_review_witnesses before/after capsule snapshot + digest
  -> resolve queue row only after witness-backed mutation succeeds

prepare_review_rollback(witness)
  -> reload memory_review_witnesses row
  -> require live capsule still equals witness after snapshot
  -> write one-use rollback approval with token hash and capsule snapshot

live_review_rollback(token)
  -> require exact confirmation and prepared, unexpired token
  -> require witness and capsule snapshots have not drifted
  -> compare-and-set capsule status back to witness before_status
  -> write memory_review_rollback_witnesses and consume approval

mutation_preflight(capsule, action)
  -> read current capsule projection
  -> build before, after, and rollback snapshots
  -> run deterministic risk checks on proposed rewrite projection
  -> map delete to reversible rejected status only
  -> block stable delete and all physical deletion
  -> return digest-bound plan without live mutation

prepare_mutation(rewrite)
  -> rerun mutation_preflight
  -> require live capsule still equals preflight before projection
  -> write one-use approval with token hash, preflight JSON, and capsule snapshot

live_mutation_apply(token)
  -> require exact confirmation and prepared, unexpired rewrite token
  -> require approval capsule snapshot and preflight digest have not drifted
  -> compare-and-set title/body/tags only
  -> write memory_mutation_witnesses and consume approval
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
26a. Treat working-memory projection as auditable evidence, not ambient context:
     govern-turn must report whether projected capsule IDs overlap visible
     recall evidence, whether a Next Action section exists, and whether the
     projection stayed inside its token budget.
26b. Record govern-turn projection impact only after the turn outcome has been
     reviewed. The record may populate working-memory-impact rows for projected
     capsule IDs, but it must not auto-promote memories, mutate policy routing,
     or claim that the projection was useful without an explicit outcome.
26c. Evaluate working-memory impact as a read-only habit audit. Helpful or
     harmful projection outcomes may explain future ranking behavior, but the
     eval command must only report and recommend review; it must not directly
     change ranking, status, lifecycle tier, or review queue state.
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
31. Treat global spreading as sandboxed until proven safe. Missing global
    activation is a watch signal, but fanout, supplementation, or depth beyond
    configured limits is a failure because cross-scope recall can otherwise turn
    old memories into accidental always-on context.
32. Treat reconsolidation apply as candidate-only until stronger apply safety
    exists. `reconsolidation-frame` and `reconsolidation-prepare` are read-only.
    `reconsolidation-prepare` stores a rollback witness preview for the evidence
    capsule identities, statuses, source links, and content digests.
    `reconsolidation-apply` may create only one candidate summary capsule and a
    witness after matching the approved fingerprint and prepared rollback
    witness fingerprint. It must not promote,
    supersede, rewrite, delete, or cool existing capsules. Any future stronger
    apply path must first pass `reconsolidation-strong-preflight` in a restored
    backup shadow store, prove recall-regression stability, and preserve
    source-event provenance with an explicit live rollback or exception witness.

## Future Extension Points

- Global spreading policy beyond the read-only sandbox and ESPA-aware path
  weighting.
- Stronger reviewed reconsolidation apply paths after shadow preflight and
  candidate-only witnesses prove useful.
- Optional local embeddings for intent matching.
- Cross-encoder reranking for high-value recall.
- Impact feedback analytics for drift, overfitting, and stale helpfulness.
- OCR/caption ingestion for images.
- Memvid-style cold archives for large old sessions.
- Small local curator model for better consolidation.
- Separate auditor model for poisoning and drift checks.
- External-command advisor provider for local models or API wrappers.
- Example advisor wrappers under `examples/`.
