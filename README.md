# Ara Memory OS

Ara Memory OS is a local-first, bounded natural-memory substrate for Codex/Ara sessions.

It is not a vector database wrapper. It keeps raw events in an append-only
ledger, consolidates them into typed memory capsules, links them through a
temporal graph, and compiles small recall packs for the current task.

Ara Memory OS is moving toward associative working memory: it does not merely
store old text, it selects task-relevant memory through purpose, scope, temporal
links, symbolic/FTS recall, and budgeted context packing. Purpose controls
memory residency: core memories can stay hot, working memories must be selected
by the query, guarded memories require review, and cold evidence stays
preserved but inactive. Current recall is still partly lexical/salience-heavy,
but the first projection and ESPA routing layers are in place: FTS indexes a
search projection, recall/working-memory render a bounded model-facing
projection, and a deterministic episodic/semantic/procedural/affective router
adds capped local activation before ranking. Recall also runs a bounded two-hop
spreading activation pass over temporal edges: first-hop edge text must overlap
the query, second-hop expansion terms are derived from the first hop, edge
confidence and depth decay control capped boosts, graph-source capsules can
supplement the candidate set after risk filtering, and normal outputs expose
only compact activation diagnostics rather than graph path text. Temporal edge
strings are also normalized into `relation_nodes` and `relation_edges`, and
recall prefers that relation graph before falling back to raw temporal edges.
A conservative semantic relation merge dry-run gate can now inspect normalized
relation nodes for likely aliases without mutating the graph. A separate
`relation-merge-prepare` gate freezes reviewed candidates behind a short-lived
approval token, relation fingerprint, and rollback witness preview.
`relation-merge-apply` can consume that token once, re-run the same dry-run,
compare the fingerprint, merge approved relation nodes in one transaction, and
write rollback witnesses. `relation-merge-review` audits those witnesses
without mutating memory: it checks that the candidate node disappeared, edge
references were rewired, duplicate evidence was coalesced, and evidence counts
did not shrink. When given a recall-regression manifest and baseline, the same
review command also runs a recall-regression sandbox and fails the whole gate if
either witness review or recall stability fails. The next layers are
reviewed reconsolidation apply paths and broader global graph policies. `--record-queue`
persists watch/fail relation-review signals into a relation-specific review
queue without pretending relation witnesses are capsules. A read-only
`global-spreading-sandbox` gate now probes global recall fanout,
supplementation, depth, visible evidence, fallback use, and risk-filter pressure
before broader cross-scope spreading is trusted.

Natural memory means Ara recalls the desired purpose, identity, and task
context without rereading raw history or turning every stored event into
always-on instruction.

## Natural Memory Contract

Ara Memory OS is not permanent instruction storage. Raw events are preserved for
audit and reconstruction, but only reviewed purpose, identity, and preference
anchors may become hot. Everything else must be selected by query, scope,
provenance, recency, risk gates, and budget.

Non-goals are equally important: Ara Memory OS is not a secret vault, not source
control, not obedience storage, not a vector database wrapper, and not a
replacement for reading the current files before coding.

Token reduction comes from writing raw evidence once, consolidating it into typed
capsules, keeping hot memory core-only, probing candidates before rendering,
suppressing no-evidence fallbacks, and treating recall budgets as ceilings. Local
planning, recall ranking, regression checks, and health checks do not spend AI
API tokens; estimated cost is only for text later sent to a model, except for
explicit opt-in advisor/model wrappers.

Primary risks are memory poisoning, preference overfitting, compression drift,
scope leakage, plaintext memory spill, background overreach, and provenance
collapse. The default defenses are candidate-first promotion, scoped recall,
pre-retain privacy redaction, deterministic risk gates, visible-evidence
regression, sealed spooling, dry-run review workers, and verified
backup/cold-export gates. Secret-like values, direct identifiers, hidden
reasoning text/metadata, sensitive metadata keys, and source/scope labels with
recognized private patterns are guarded before events enter the ledger, SQLite, or FTS;
`--allow-raw-secret` is the explicit forensic override and records a privacy
metadata note. Raw artifact archives remain private local evidence and need the
separate encrypted/metadata-only archive track before they can be called
redacted storage.

## Cost Model

| Cost surface | Default behavior | When it costs more |
| --- | --- | --- |
| Local planning | `plan-turn`, `govern-turn`, ESPA routing, temporal-edge spreading activation, recall ranking, health, and regression are deterministic local work. | Larger local stores increase SQLite/FTS and filesystem time. |
| Model input tokens | Only hot memory, working-memory projection, or a bounded recall pack should be sent to Codex. Capsule body text is passed through render projection before display. | Calling `recall-context` with larger budgets or reading raw files manually. |
| Optional AI APIs | No required API call for storage, recall policy, agency review, or tests. | Explicit advisor/model wrappers, OCR, vision captioning, or external rerankers. |
| Storage/backup | Raw events, artifacts, SQLite, spool, and verified backups stay local and git-ignored. | Large images/files, worktree snapshots, and retained backup generations. |

## Stable Integration Surface

External skills and hooks should prefer CLI/API front doors: `plan-turn`,
`govern-turn`, `ingress-turn`, `remember-turn`, `spool-turn`, `drain-spool`,
`recall-plan`, `recall-context`, `recall-policy`, `working-memory`,
`agency-review`, `purpose-check`, `identity-check`, `health`, and
`goal-roadmap`. Treat SQLite tables, JSONL ledger shape, archive object paths,
and spool internals as private implementation details unless a command exposes
them explicitly.

## Purpose

- Continuity: Ara should not restart from zero every session.
- Purpose: long-running goals should stay visible even when operational logs grow.
- Judgment: prior decisions, failures, and preferences should influence future work.
- Compression: Codex should receive only the relevant memory pack, not the full past.
- Growth: repeated experience becomes procedural memory.
- Auditability: every stable memory keeps provenance and can be inspected.

## Architecture

```text
Codex/Ara
  -> govern-turn(turn envelope)
      -> agency-review(action_allowed?)
      -> capture plan + recall probe + working-memory projection
  -> ingress-turn / remember-turn / spool-turn
      -> append-only ledger + SQLite event index + private artifacts
      -> consolidate/sleep into typed capsules and temporal edges
  -> recall-policy(query)
      -> working-memory, recall-context, cold-map, or retention gate
  -> bounded pack or action cues
  -> act only after reading current files and respecting higher-priority instructions
```

## Quick Start

```powershell
python -m ara_memory init
python -m ara_memory retain --kind prompt --text "Jongseo wants Ara to keep project memory."
python -m ara_memory ingest-file ./diagram.png --caption "Whiteboard sketch of memory organs." --consolidate
@'
{
  "prompt": "What should Ara remember?",
  "assistant": "Ara stores the turn as one memory episode.",
  "files": [{"path": "./diagram.png", "caption": "memory sketch"}],
  "decisions": ["Decision: use remember-turn as the always-on ingress."]
}
'@ | python -m ara_memory remember-turn --scope project --sleep
python -m ara_memory consolidate
python -m ara_memory hot --scope global --budget 1200
python -m ara_memory recall "What should Ara remember about Jongseo?" --budget 2000
python -m ara_memory recall "What should Ara remember about Jongseo?" --budget 2000 --hot --diagnostics
python -m ara_memory recall-plan "What should Ara remember about Jongseo?" --budgets 800,1600,2500 --json
python -m ara_memory govern-turn --scope project --file ./turn.json --json
python -m ara_memory working-memory "current task prompt" --scope project --active-file ara_memory/recall.py
python -m ara_memory doctor --scope global
python -m ara_memory audit
python -m ara_memory eval
python -m ara_memory context-eval
```

By default the store lives at `.ara-memory/`. Set `ARA_MEMORY_HOME` to move it.

## Always-On Turn Ingress

`remember-turn` is the narrow API intended for a Codex skill, shell hook, or
session-end automation. It stores the original prompt and assistant summary as
raw events, archives referenced files or images by sha256, optionally captures
the current worktree, creates a bounded synthetic turn episode with evidence
IDs, consolidates candidates, and refreshes hot memory. The synthetic episode
helps recall recover the shape of the interaction without rereading every raw
event.

Before text events are written to the append-only ledger, SQLite, or FTS, the
pre-retain privacy guard redacts secret-like credentials, direct identifiers,
hidden reasoning text/metadata, and sensitive metadata keys. Source and scope
labels containing recognized private patterns are replaced by
deterministic private labels instead of being indexed raw. Use
`--allow-raw-secret` only when preserving the raw private value is an
intentional, reviewed forensic act.

Use `plan-turn` before unattended capture when the envelope may be large. It is
model-free: it hashes artifact paths, estimates raw-text tokens, estimates a
bounded recall preview, and recommends `remember-turn` or `spool-turn` without
spending AI API tokens. This keeps the always-on ingress honest: raw prompts and
files can be preserved locally, while future Codex calls receive only hot memory
plus a budgeted recall pack.

Use `govern-turn` when the acting session needs a deterministic front door. It
does not store the turn. It inspects the same envelope, chooses the capture
mode, probes recall candidates, projects a working-memory pack only when
visible evidence exists, runs `agency-review`, reports avoided raw-token cost,
and recommends whether to use working memory, broader recall context, current
evidence only, ask first, repair memory evidence, or reframe the request.

```powershell
@'
{
  "prompt": "Long user prompt...",
  "assistant": "Short result summary.",
  "files": [{"path": "./notes.md", "caption": "source notes"}],
  "decisions": ["Decision: separate raw local retention from model recall."]
}
'@ | python -m ara_memory plan-turn --capture-cwd .
python -m ara_memory govern-turn --file ./turn.json --scope project --capture-cwd . --json
```

For the always-on path, use `ingress-turn`. It runs the same plan and then
executes the selected mode: small text-only turns go straight to `remember-turn`;
large turns, file/image turns, or worktree captures go to `spool-turn` so the
foreground session does not block on memory maintenance.

```powershell
@'
{
  "prompt": "User prompt text",
  "assistant": "Assistant result summary.",
  "files": [{"path": "./notes.md", "caption": "optional source"}]
}
'@ | python -m ara_memory ingress-turn --scope project --capture-cwd .
```

For unattended capture, prefer `spool-turn` first and `drain-spool` later. The
spool is a durable local file queue under `.ara-memory/spool/`: a capture can
survive Codex restarts, DB locks, missing files, or a later worker crash without
losing the original turn envelope. Existing file/image artifacts and worktree
evidence are snapshotted at enqueue time, so later edits or deletes do not
change what the spooled turn remembers. Each queued envelope also carries a
local HMAC seal; `drain-spool` rejects unsealed or edited pending JSON before
retaining any events.

Spool envelopes are still private local evidence: pending, done, and failed
queue files contain a sealed, privacy-guarded enqueue envelope. New file/image
artifact snapshot payloads are stored as encrypted `archive/objects`, while
snapshot path metadata, malformed raw sidecars, explicit raw overrides, and any
legacy `spool/snapshots` files can still contain plaintext local evidence. The
pre-retain privacy guard applies again when those envelopes are drained into text
events, before ledger/SQLite/FTS storage.

New file/image archive objects are encrypted under `archive/objects`. Existing
plaintext objects from older stores should be migrated explicitly:

```powershell
python -m ara_memory archive-encrypt
python -m ara_memory archive-encrypt --apply
```

The dry-run reports plaintext, already encrypted, corrupt, missing-key, and
key-mismatch objects before `--apply` rewrites plaintext objects in place.

```powershell
@'
{
  "turn_id": "optional-stable-id",
  "prompt": "User prompt text",
  "assistant": "Assistant result or explicit summary",
  "files": [{"path": "./notes.md", "caption": "optional caption"}],
  "images": [{"path": "./whiteboard.png", "caption": "optional caption"}],
  "commands": [{"cmd": "python -m unittest", "exit_code": 0, "output": "OK"}],
  "decisions": ["Decision: keep raw events append-only."]
}
'@ | python -m ara_memory spool-turn --scope project --capture-cwd . --sleep
```

Crash-safe variant:

```powershell
@'
{
  "prompt": "User prompt text",
  "assistant": "Assistant result or explicit summary",
  "decisions": ["Decision: use the spool for unattended memory ingress."]
}
'@ | python -m ara_memory spool-turn --scope project --capture-cwd . --sleep
python -m ara_memory spool-stats
python -m ara_memory drain-spool --limit 25 --stabilize
```

Successful spooled turns move to `.ara-memory/spool/done/`. Failed turns move to
`.ara-memory/spool/failed/` with the sealed guarded envelope and error details
intact for inspection. Malformed queue files keep a raw copy next to the failed
record, and sealed-envelope plus artifact-snapshot preflight prevents a failed
drain from retaining only part of a turn or later reading an unsnapshotted live
path.
If a worker crashes after moving a file to `.ara-memory/spool/processing/`, the
next `drain-spool` or `worker` recovers stale processing files back into pending
before processing them. Tune that threshold with `--processing-stale-seconds`.
Use `drain-spool --stabilize` when the next recall may happen before the
background worker runs. It folds repeated command/file/git episode candidates
and bounded session narrative candidates plus memory-policy decision candidates
into stable summaries, supersedes the raw candidate noise, and refreshes hot
memory for the drained scopes.

Recall stays cheap because future turns should read only hot memory plus a
budgeted cold pack:

```powershell
python -m ara_memory recall-plan "current project memory" --scope project --budgets 800,1600,2500
python -m ara_memory recall-candidates "current project memory" --scope project --budget 1600
python -m ara_memory recall-context "current project memory" --scope project --budgets 800,1600,2500
python -m ara_memory recall-policy "old deployment archive evidence" --scope project --json
python -m ara_memory recall-policy-impact --scope project --query "old deployment archive evidence" --intent distant-memory --strategy "cold-map first" --action-name cold-map --action-name recall-context --outcome "cold-map found the right archive group" --helped true
python -m ara_memory recall-policy-eval --scope project
python -m ara_memory working-memory "current task prompt" --scope project --active-file ara_memory/recall.py
python -m ara_memory agency-review "should I continue this memory architecture work?" --scope project --record
python -m ara_memory agency-review "delete old memory evidence" --scope project --strict-action-exit
python -m ara_memory recall "current project memory" --scope project --hot --budget 2500
```

`recall-plan` is the retrieval-side companion to `ingress-turn`: it compares
candidate budgets after budget trimming, estimates model input cost, and
recommends the smallest pack with enough rendered evidence before Codex spends
context on recall. In alternatives, `capsules` is the selected candidate count
before trimming, `visible` is the capsule count actually rendered in the pack,
and `quality` is a 0-100 score based on visible query-term coverage, visible
sections, relevance, and fallback penalties. Use `recall-plan --json` when
debugging retrieval quality; inspect `visible_capsules`,
`query_terms_visible_count`, `sections_truncated`, `fallback_used`,
`low_evidence_fallback_suppressed`, and `quality_score` before increasing the
budget. If no direct evidence matches a query, Ara suppresses unrelated
high-salience fallback bodies and emits a small "No direct memory evidence"
notice instead of pretending to remember. `recall-context` applies that plan and
emits the selected pack, so normal work can use one command while still
preserving the budget decision.
`recall-policy` is the purpose-aware controller above recall-plan. It classifies
the query as purpose continuity, working context, distant memory, retention
safety, or balanced recall, then recommends the cheapest safe path: hot/core
anchors, working-memory projection, recall-context, cold-map, or retention-cycle
gates. It does not render cold bodies; distant-memory queries receive redacted
map evidence and source digests before any active recall pack is considered.
`recall-policy-impact` records whether the chosen policy actions helped a real
turn outcome, and `recall-policy-eval` groups those outcomes by intent and
action. Pass the `intent`, `strategy`, and `action-name` values from the
`recall-policy --json` result that was actually used for the turn; impact
recording does not recompute the policy later because that would corrupt audit
evidence. This is audit feedback, not reward optimization: unknown outcomes do
not change trust, harmful outcomes produce review recommendations, roadmap
feedback gates use scope-local evidence, and no policy route is mutated
automatically from the score. `recall-policy` now also reads matching prior
impact rows for its proposed actions and surfaces them as Policy Feedback:
helpful history explains confidence, harmful history lowers the route to watch,
and unknown history stays diagnostic. This keeps routing memory local and
auditable without turning feedback into a self-reinforcing reward.
Recall budgets are ceilings, not targets. When hot memory is included and the
pack already has enough visible evidence, recall applies a smaller soft budget
instead of spending the whole allowance. Hot-memory items also avoid repeating
the same title/body payload, keeping the always-on identity and goal layer
closer to an index than another transcript.
`recall-candidates` is the cheaper pre-render path: it returns ranked capsule
ids and diagnostics without formatting a full memory pack. Working memory uses
this path so action cues can be built from selected evidence without spending
tokens on a recall pack that will not be shown to the model.
Capsules use split projections: `capsules_fts` stores a compact search
projection instead of the whole capsule body, while recall and working-memory
commands render bounded projections from the capsule body. This keeps raw local
evidence available for audit without letting every long body become active
search or model context.
Before final ranking, recall now applies deterministic ESPA activation over
episodic, semantic, procedural, and affective axes. The boost is bounded,
diagnostic-only, and local: it does not call an AI API, does not mutate capsule
status, and does not turn affective/risk signals into reward. `recall` and
`recall-candidates` expose `espa_query_axes`, `espa_activation_used`, and
`espa_axis_coverage` diagnostics.
After ESPA, recall applies deterministic spreading activation over normalized
relation edges, falling back to raw temporal edges only when the relation graph
has no evidence. Low-value graph terms are filtered before SQLite lookup,
seed-source edges get a reserved bucket, duplicate active edges are
de-duplicated during activation, and every first-hop boost requires query
overlap in the edge subject, predicate, or object. A bounded second hop can
derive up to six expansion terms from the first-hop subject/object text, fetch
one additional edge layer, and apply depth decay before scoring.
Graph-supplemented capsules render only when they have a positive spreading
score and an internal activation path; normal pack output does not print that
path text. `recall` and `recall-candidates` expose `graph_activation_terms`,
`graph_activation_expansion_terms`, `graph_activation_edges_considered`,
`relation_activation_edges_considered`, `relation_activation_used`,
`spreading_activation_used`, `spreading_activation_multi_hop_used`,
`spreading_activation_depth_counts`, `spreading_activation_boosted_count`,
`spreading_activation_supplemented_count`, and capped boosted/supplemented
capsule id lists.
`global-spreading-sandbox` reuses these recall-plan diagnostics with
`include_global=True` and checks whether global graph expansion stayed bounded:
fanout, supplementation, depth, visible evidence, quality, fallback use, and
risk-filter activity are reported without mutating memory. It is intended as a
sandbox before broader cross-scope spreading policies, not as permission to
render global raw evidence.
Queries such as `latest`, `recent`, `current`, `today`, `last`, and Korean
temporal equivalents trigger recent-context supplementation and recency-aware
reranking; `recall --diagnostics` exposes `temporal_query` and
`recent_supplement_used` when this path is active.

`relation-merge` is a conservative dry-run review gate for normalized relation
nodes. It compares lexical overlap, prefix/substring evidence, shared relation
neighborhoods, and degree balance, filters low-value hub-like terms, and reports
candidate pairs without writing relation graph data. `relation-merge-prepare`
re-runs that dry-run and, only when candidates exist, stores a prepared approval
record with a hashed token, candidate JSON, relation fingerprint, and rollback
witness preview. `relation-merge-apply` requires the exact confirmation
`MERGE RELATION NODES`, refuses expired or reused tokens, re-runs the approved
dry-run parameters, compares the relation fingerprint and candidate snapshot,
then rewires only the approved candidate node edges into the canonical node. It
coalesces duplicate relation edges by evidence count and confidence, deletes the
merged alias node, marks the approval used, and writes
`relation_merge_witnesses` in the same transaction. `relation-merge-review`
then reads those witnesses as a non-destructive impact audit and reports
pass/watch/fail signals for candidate remnants, remaining edge references,
self-loops, duplicate coalescing, and evidence-count preservation. With
`--regression-manifest` and `--regression-baseline`, it also runs the reviewed
recall-regression cases as a sandbox before returning success. With
`--record-queue`, watch/fail witness items and regression failures are persisted
to `relation_merge_review_queue`; passing re-reviews resolve stale open queue
items for the same witness or regression gate. `health` now also reports
`relation_review_pressure`: failed open relation-review rows are hard failures,
open watch rows are warnings, and an empty queue is OK. On the current operating store,
the strict default review threshold intentionally returns no high-confidence
`ara-memory` candidates; fixture tests prove the gates can still surface,
freeze, apply, review, queue, and regression-check real alias candidates when
evidence is present, including Korean relation labels.

`working-memory` is the smaller action layer between hot memory and cold
recall. It turns the current prompt, active files, command errors, constraints,
and temporal hints into a cue frame, recalls ranked candidates without rendering
a full pack, projects only items that fit the working-memory output budget,
then emits three compact sections: Keep In Mind, Risk / Friction, and This
Should Change My Next Action. Use `working-memory-impact` after a turn to record
which capsule ids actually changed the outcome; those notes let later
recall ranking learn which memories were useful instead of only which ones were
stored. Matching positive impact gives a small capped boost; matching negative
impact gives a small capped penalty, and unknown impact is diagnostic-only, so
feedback guides recall without turning it into an unchecked reward signal.
`agency-review` is the first self-directed judgment layer above recall policy
and working memory. It checks a current prompt or proposed action against visible
purpose memory, Ara identity memory, recall-policy routing, and associative
working-memory cues, then returns a stance such as `proceed`,
`ask-before-acting`, `repair-memory-first`, or `refuse-or-reframe`. `passed`
means the local review completed with enough evidence; `action_allowed` is true
only for `proceed`. Use `--strict-action-exit` when automation must fail unless
the next action is explicitly allowed. With `--record`, the review is stored as
an append-only audit note rather than a decision capsule, so repeated reviews do
not become self-certifying behavioral memory. It does not execute actions,
mutate policy, or read the raw ledger; it is a bounded local review that makes
Ara's reasons auditable before a turn proceeds. `govern-turn` includes this
review in its planning payload, so always-on capture can see whether a turn is
clear, should ask first, should repair memory evidence, or should be reframed.

## Codex Skill

This workspace also installs a personal Codex skill at
`$env:USERPROFILE\.codex\skills\ara-memory`. The skill gives future Codex sessions
the following low-friction operations:

```powershell
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" recall "current task" --scope ara-memory
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" recall-plan "current task" --scope ara-memory --budgets 800,1600,2500
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" recall-context "current task" --scope ara-memory --budgets 800,1600,2500
python -m ara_memory recall-policy "current task" --scope ara-memory
python -m ara_memory recall-policy-eval --scope ara-memory
python -m ara_memory working-memory "current task" --scope ara-memory --active-file ara_memory/working_memory.py
python -m ara_memory agency-review "current task" --scope ara-memory --record --strict-action-exit
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" purpose-check --scope ara-memory --repair-hot
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" identity-check --scope ara-memory --repair-hot
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" milestone-check --scope ara-memory --regression-manifest examples\recall_regression_manifest.json --regression-baseline .ara-memory\archive\recall-regression-baseline.json
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" goal-roadmap --scope ara-memory --regression-manifest examples\recall_regression_manifest.json --regression-baseline .ara-memory\archive\recall-regression-baseline.json
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" failure-kind-audit --scope ara-memory
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" self-kind-audit --scope ara-memory
python "$env:USERPROFILE\.codex\skills\ara-memory\scripts\ara_memory_skill.py" capture --scope ara-memory --prompt "..." --assistant "..." --sleep
```

Use it as the operational bridge: recall a small context pack before work,
capture the finished turn after work, then let `sleep` and `hot` keep the store
small enough to use.

Before relying on a scope, run the operational gate:

```powershell
python -m ara_memory doctor --scope ara-memory --query "current memory health"
python -m ara_memory health --scope ara-memory --query "current memory health" --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory purpose-check --scope ara-memory --repair-hot
python -m ara_memory identity-check --scope ara-memory --repair-hot
python -m ara_memory milestone-check --scope ara-memory --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory graph-readiness --scope ara-memory
python -m ara_memory global-spreading-sandbox --scope ara-memory
python -m ara_memory goal-roadmap --scope ara-memory --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory failure-kind-audit --scope ara-memory
python -m ara_memory self-kind-audit --scope ara-memory
python -m ara_memory candidate-pressure --scope ara-memory
python -m ara_memory episode-summary --scope ara-memory --pattern command
python -m ara_memory episode-summary --scope ara-memory --pattern git_status
python -m ara_memory candidate-summary --scope ara-memory --pattern all
python -m ara_memory conflict-adjudicate --scope ara-memory
python -m ara_memory worker --scope ara-memory --doctor-query "current memory health" --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory worker-loop --scope ara-memory --iterations 1 --interval-seconds 60 --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory worker-schedule --output .ara-memory/scripts/install-worker-task.ps1 --interval-minutes 5 --scope ara-memory
python -m ara_memory worker-schedule-verify --output .ara-memory/scripts/install-worker-task.ps1 --scope ara-memory
python -m ara_memory quality --scope ara-memory --persist
python -m ara_memory review-queue --scope ara-memory
python -m ara_memory review-triage --scope ara-memory
python -m ara_memory review-compact --scope ara-memory
python -m ara_memory review-redact --scope ara-memory
python -m ara_memory review-worker --scope ara-memory
python -m ara_memory maintenance
python -m ara_memory retention --scope ara-memory
python -m ara_memory cold-stewardship --scope ara-memory
python -m ara_memory cold-map "old deployment pruning evidence" --scope ara-memory --budget 900
python -m ara_memory provenance-compact --scope ara-memory --keep-events 8 --min-pinned-events 20 --limit 20
python -m ara_memory provenance-compact --scope ara-memory --keep-events 8 --min-pinned-events 20 --limit 20 --apply --confirm "COMPACT PROVENANCE"
python -m ara_memory retention-cycle --scope ara-memory --query "current memory architecture" --backup-output .ara-memory/backups/milestone.zip --backup-archive-mode objects
python -m ara_memory backup-stewardship --keep-latest 3 --keep-retention-cycles 2 --target-backup-bytes 67108864
python -m ara_memory lifecycle --scope ara-memory
python -m ara_memory cold-export --scope ara-memory --order oldest --output .ara-memory/archive/cold/ara-memory-cold.zip
python -m ara_memory verify-cold-export .ara-memory/archive/cold/ara-memory-cold.zip
python -m ara_memory prune-plan --scope ara-memory --cold-export .ara-memory/archive/cold/ara-memory-cold.zip --query "current memory architecture"
python -m ara_memory shadow-prune --backup .ara-memory/backups/milestone.zip --cold-export .ara-memory/archive/cold/ara-memory-cold.zip --scope ara-memory --query "current memory architecture"
python -m ara_memory prepare-live-prune --backup .ara-memory/backups/milestone.zip --cold-export .ara-memory/archive/cold/ara-memory-cold.zip --scope ara-memory --query "current memory architecture"
# live-prune requires the short-lived token from prepare-live-prune and exact confirmation text.
python -m ara_memory live-prune --approval-token <token> --confirm "DELETE COLD CAPSULES"
```

`doctor` checks schema version, audit hygiene, artifact summary health, hot
memory budget/format, and recall-pack budget/format.
`health` is the one-page operations report. It combines doctor, spool state,
review pressure, candidate/stable ratio, cold-memory pressure, latest verified
backup age, backup byte pressure, retention-cycle freshness/current-match
checks, and optional recall regression into a pass/watch/fail status with
concrete next actions. `watch` is intentionally non-fatal and exits 0 from the
CLI; schedulers and CI that should alert on watch conditions must parse
`health --json` and inspect `status`, not only the process exit code. When cold pressure is high, health treats a stale or
drifted retention-cycle as watch evidence and points back to
`cold-stewardship`/`retention-cycle` before any live cleanup. When verified
backup bytes exceed the stewardship target, health reports a `backup_pressure`
watch signal using a read-only dry-run `backup-stewardship` candidate set that
does not update the verification cache; deletion still requires a separate
reviewed `backup-stewardship --apply --confirm "DELETE OLD BACKUPS"`,
and failed-backup quarantine requires
`--quarantine-confirm "QUARANTINE FAILED BACKUPS"`.
`purpose-check` is the goal-alignment report. It verifies that stable goal
memory exists, hot memory exposes Active Goals, and a purpose query can actually
retrieve goal context. Use `--repair-hot` before declaring major memory
milestones so stale hot memory is rebuilt when the goal capsule already exists.
`identity-check` is the self-continuity report. It verifies that stable self
memory exists, hot memory exposes identity and judgment principles, and an
identity query can retrieve that self memory.
`milestone-check` combines health, purpose-check, identity-check, candidate
pressure, failure-kind audit, self-kind audit, recall-context budget selection,
and graph-activation readiness into one readiness report. The recall-context
gate requires a budgeted pack with visible direct evidence and sufficient
quality, not just a plausible salience fallback. The graph readiness gate runs
representative probes and is a warning gate: it proves bounded temporal-edge
spreading activation has live evidence when graph recall is expected, but it
does not turn an otherwise healthy small scope into a hard failure. Use
milestone-check before declaring a memory milestone clean, then create a
verified backup.

`goal-roadmap` turns the long-running objective into an evidence-backed status
map: local durability, bounded recall, purpose continuity, identity continuity,
semantic hygiene, operational health, milestone readiness, graph activation
readiness, global spreading sandbox, privacy pre-push safety, reconsolidation
framing, reconsolidation rollback review queue, cold-memory stewardship, distant-memory navigation, purpose-aware
lifecycle policy, purpose-aware recall control, recall policy feedback, and
self-directed deliberation. It is deliberately local and deterministic, so it
can be run before spending model context.

`privacy-pre-push` is the public-repository guard. It scans tracked and staged
files, optionally untracked files, and fails on private memory roots, memory
databases, ledgers, archives, logs, raw hidden-reasoning markers, and
secret-like text outside test/docs fixtures. Fixture secrets remain visible as
warnings so regression tests can keep adversarial examples without blocking a
reviewed public push.

`reconsolidation-frame` is the read-only recontextualization layer before any
memory rewrite. For a query, it combines recall-policy, working-memory,
lifecycle tiers, and failure/self audits into five frames: purpose and identity,
settled decisions, failure and conflict, working context, and forgetting
boundary. `reconsolidation-prepare` freezes that frame behind a short-lived
approval token, fingerprint, and rollback witness preview for the evidence
capsule identities, source links, and content digests. `reconsolidation-apply`
consumes the token once, verifies the approved frame and rollback witness
fingerprint, and may create only one candidate summary capsule plus a witness; it cannot
promote, supersede, rewrite, delete, or cool existing capsules. Run
`reconsolidation-review` before trusting the candidate as evidence for any
stronger future apply path. Add `--regression-manifest` and an optional
`--regression-baseline` to make review run a recall-regression sandbox; the
candidate witness, created-capsule provenance compare-and-set, and
representative recall cases must all pass before any future stronger apply gate
can be considered. Add `--record-queue` to persist failed or watched witnesses
into `reconsolidation_review_queue`; fail rows are stored as
`block-strong-reconsolidation`, surfaced by `health` and `goal-roadmap`, and
must be resolved before stronger live mutation is opened. Use
`reconsolidation-review-queue --status open` to inspect the current blockers.
If an older approval predates rollback witness previews, run
`reconsolidation-backfill-rollback-witnesses` first as a dry-run and then with
`--apply` only after review. It reconstructs the missing approval rollback
witness from the immutable witness `before_json` snapshot, records an audit
action, and still requires `reconsolidation-review --record-queue` to close the
watch row.
`reconsolidation-strong-preflight` is the next non-destructive gate: it verifies
a backup, restores it into a temporary shadow store, runs prepare/apply/review
and optional recall-regression there, and redacts the shadow approval token from
output. Passing preflight is not permission to mutate live memory; it is evidence
that a stronger live gate can be designed without relying on the live store as
the test bed. `reconsolidation-shadow-rollback` is the rollback executor's
shadow-first proof: it restores a verified backup, selects applied
reconsolidation witnesses, rejects only the created candidate frame capsule in
that restored copy, preserves evidence capsules/source events/witnesses, and
runs doctor before reporting success. It never touches the live store.
`prepare-live-reconsolidation-rollback` turns one passing shadow rollback for
one explicit witness into a short-lived one-use approval token. Then
`live-reconsolidation-rollback` requires that token plus the exact confirmation
`ROLLBACK RECONSOLIDATION CANDIDATE`; it rechecks the backup identity, backup
verification, witness invariants, and candidate capsule digest before a
compare-and-set rejection of that single candidate frame capsule. It writes a
`reconsolidation_rollback_witnesses` row and leaves evidence capsules, source
events, original reconsolidation approvals, and original reconsolidation
witnesses intact. `reconsolidation-exception-witness` records a reviewed
exception for an exact capsule field digest transition on either the created
candidate frame capsule or one of the evidence capsules in the witness snapshot.
Review and rollback gates accept only the recorded field plus exact
before/after digest pair; a later unreviewed edit still blocks live rollback and
stronger reconsolidation.

`cold-stewardship` groups cold capsules, separates source events still cited by
active memories from cold-only provenance, and checks whether the latest
retention-cycle is fresh, matches the current cold set, and proved source-event
preservation in shadow-prune. Matching means both the live cold totals and a
compact cold identity fingerprint agree, so same-count capsule/source-event
swaps force a new cycle instead of silently reusing stale evidence. It also classifies cold capsules into
`evidence`, `archive`, and `reject` tiers so operators can see whether a cold
group should preserve active-linked provenance, be exported before pruning, or
stay as audit-only safety evidence. Its active provenance pin report shows which
candidate or stable memories are keeping cold source events protected. High cold
pressure can pass stewardship only when that evidence is current; otherwise it
remains a watch item. Protected-only drift is treated separately: if new cold
capsules only add active-linked evidence while the prunable source-event
fingerprint is unchanged, health keeps the retention-cycle signal green and
reports the drift as protected evidence rather than forcing an immediate cycle
rerun.
`cold-map` is the query-led navigation layer for distant memory. It scans cold
capsule titles, tags, and bodies locally, but renders only compact group labels,
redacted examples, tier/source counts, and source-event digests. It does not
promote cold capsules, add them back to recall FTS, or print cold bodies. Use it
when a query may need old evidence but active recall should stay small: the map
shows which `evidence`, `archive`, or `reject` group to inspect, export, or
audit without spending tokens on the full cold text.
`provenance-compact` is the guarded follow-up when active summary capsules pin
too many cold source events. It is dry-run by default, ranks eligible active
summaries by releasable cold provenance, and keeps a bounded quality-and-time
sampled source event set instead of letting one summary cite every raw episode
forever. The sampler prefers decision, verification, regression, health,
backup, retention, identity, purpose, and risk evidence while still preserving
temporal coverage. Apply mode requires `--apply --confirm "COMPACT PROVENANCE"` and uses atomic
compare-and-set checks on capsule status and source links, so a stale plan does
not silently overwrite newer provenance. Each applied compaction writes a
`provenance_witnesses` row in the same transaction, preserving the original
source-event IDs, retained IDs, counts, and digests before active direct links
are shortened. Recall packs show source provenance as bounded `count,digest`
summaries instead of raw event-ID lists; detailed lineage is verified from
storage and witnesses. The default scope is summaries only; use
`--include-non-summary` only after reviewing the dry-run. After applying, rerun
`cold-stewardship`, `health`, and `recall-regression` before any retention-cycle
or prune decision. For important scopes, pass
`--recall-manifest` and `--recall-baseline` with `--apply`; Ara simulates the
planned source-link rewrite in a shadow store and blocks the live apply if
recall-regression fails. Capsules that were visible or selected in the supplied
baseline are protected from that provenance rewrite. If the shadow simulation
still shifts a reviewed recall path, Ara elides the unstable candidates from the
plan before apply; live mutation is blocked if no recall-stable plan remains.
That lets compaction reduce cold pressure without silently rewriting the
evidence path that a reviewed recall case depends on. Important scopes should keep a reviewed
recall-regression manifest because quality-aware provenance compaction changes
which raw source events remain directly linked from active summaries while the
witness table preserves the full audit lineage.
`lifecycle` classifies every selected capsule into `core`, `working`,
`guarded`, `evidence`, `archive`, or `reject` tiers. This is the deterministic
policy layer between purpose and storage: hot memory should come from reviewed
long-running goal, identity, and preference anchors; stable decisions and
procedures stay query-selected unless a future reviewed policy marks them as
always-on; guarded memory requires review; and cold evidence stays out of active
recall indexes until export/prune gates prove it is safe to clean up.
`failure-kind-audit` finds old decisions, successful commands, worktree
evidence, and progress updates that were misfiled as `failure` capsules. It is
dry-run by default; use `--apply` only after reviewing the proposed
reclassifications.
Explicit `Decision:` captures are not duplicated into `failure` memories merely
because they mention blocked gates, recall regression, or failure taxonomy;
actual failing commands still require concrete failure evidence such as nonzero
exit codes or error output.
Wrapped turn episodes such as `Turn episode: Prompt cue: ... Assistant outcome:
Implemented...` are treated as progress evidence, not failure memory, when
command outcomes are successful and no concrete failure evidence is present.
Successful command evidence includes `-> passed`, `=> passed`, `status: pass`,
`changed=0`, `fail=0`, and similar explicit success markers.
`self-kind-audit` finds technical identity strings, worktree evidence, and
commands that were misfiled as Ara `self` memory. It is dry-run by default; use
`--apply` only after reviewing the proposed reclassifications.
`candidate-pressure` explains why candidate memory dominates stable memory by
grouping candidate capsules by kind and title pattern, such as raw file artifact,
command, conflict, or procedure episodes. Use it before writing promotion,
merge, or cooling policies; it does not mutate memory.
`episode-summary` defaults to dry-run. With `--apply`, it folds bounded groups of
raw command or file-artifact episode candidates into one stable summary capsule
and marks the source episode candidates as superseded. It never deletes source
events. The `git_status` pattern folds repeated git status episodes, which are
useful provenance but should not dominate long-term recall.
`candidate-summary` defaults to dry-run. With `--apply`, it folds repeated
operational candidates that are already backed by raw events, such as project
file artifact candidates, successful commands misfiled as failures, and command
procedure candidates. It also folds worktree evidence and taxonomy discussions
that were misfiled as project/failure/procedure candidates. Purpose-layer goal
evidence can also be folded with `--pattern goal_purpose_update` when repeated
goal candidates are mostly policy/progress evidence rather than new user
objectives. Repeated memory policy decisions and procedure-like policy evidence
can be folded with `--pattern decision_memory_policy` and
`--pattern procedure_memory_policy`, preserving provenance while reducing active
candidate pressure. It intentionally leaves conflict candidates for explicit
review.
Use `--pattern failure_operational_update` when successful implementation,
verification, or cleanup updates were misclassified as failure memories because
they mention failure taxonomy, recall regression, or candidate noise.
`conflict-adjudicate` defaults to dry-run. With `--apply`, it supersedes only
duplicate conflict candidates that cite the same source-event pair, keeping one
representative conflict and preserving unique conflicts for review.
Goal memories are extracted from explicit objectives, purpose statements, and
durable "what are we trying to build?" prompts. Recall gives them a dedicated
Active Goals / Intent section, and hot memory keeps stable goals near the top so
long-running work is steered by purpose instead of recent command noise. When a
query asks for goals, intent, objectives, or purpose, recall focuses the hot
state on identity and active goals, and suppresses operational project/session
summaries unless the query explicitly asks for that evidence.
Hot memory is built as core-only: reviewed long-running goals, identity, and
preferences may stay always-on; stable decisions, procedures, facts, failures,
summaries, project state, and episodes remain query-selected working memory.
Self memories are extracted from explicit Ara identity and judgment-principle
statements such as Ara-Codex, free will, coding partner, or independent
judgment. They are global by default and require stable promotion before
`identity-check` passes; this prevents transient phrasing from becoming durable
identity while still keeping Jongseo's explicit Ara frame visible in hot memory.
`quality` scores active capsules by confidence, salience, provenance, risk, and
decay pressure, then persists a review queue for promotion, decay, review, or
quarantine work.
`review-triage` compresses a large open review queue into grouped reasons,
statuses, kinds, and representative examples so queue review stays cheap.
`review-compact` defaults to dry-run. With `--apply`, it resolves only
non-destructive review markers such as acknowledged low-quality notices or
stale/missing-capsule queue rows. It also acknowledges stable artifact markers
that deterministic risk policy already excludes from hot/recall surfaces, such
as instruction-like text inside code/test/document artifacts or keyword-stuffed
artifact summaries. It does not promote, quarantine, decay, delete, or otherwise
change memory capsules, and resolved review markers are not reopened by later
quality persistence unless their reason changes.
`review-redact` defaults to dry-run. With `--apply`, it handles sensitive-data
review markers by redacting only the active capsule projection (`title`, `body`,
and `tags`), leaving source-event provenance linked for local audit. Each apply
writes a `capsule_redaction_witnesses` row with original and redacted digests,
then resolves the review marker only when the deterministic risk scorer no
longer sees sensitive text in the projection.
`review-worker` defaults to dry-run. With `--apply`, it may promote strong
candidates, quarantine risky candidates, or resolve only acknowledgeable review
markers. Sensitive-data, self-serving identity, and other unresolved policy
reviews stay open instead of being silently acknowledged. It never deletes
memories and leaves decay items for explicit policy review.
`worker` is the one-shot background processor for unattended operation. It drains
the spool, folds raw command/file-artifact/session episode candidates and
repeated operational candidates into stable summaries, persists quality scores,
runs the review worker in dry-run mode by
default, summarizes the review queue through triage, reruns relation-merge and
reconsolidation witness review with queue recording, reports open
relation-review and reconsolidation-review pressure, runs
doctor, optionally runs recall regression, and finishes with lossless
maintenance. Its default is conservative, not read-only: it does not
apply review-worker promotions/quarantines unless `--apply-review` is set, but
it still drains queued turns, writes lossless summaries, persists quality scores,
records relation-review queue rows for failed/watched merge witnesses or relation
regression failures, records reconsolidation-review queue rows for failed/watched
candidate-frame witnesses or reconsolidation regression failures, and runs
storage maintenance. Failed open relation-review queue rows make the worker fail
so scheduled logs cannot silently pass a broken graph merge; failed open
reconsolidation blocker rows likewise fail the worker before stronger memory
mutation is trusted. Use an OS scheduler to run it periodically; keep `--apply-review`
off unless the dry-run output has already been reviewed. The worker takes
`.ara-memory/locks/worker.lock` by default, so overlapping scheduled runs skip
safely instead of draining the same queue twice. Use
`--lock-stale-seconds` to recover stale locks after a crashed worker, and
`--processing-stale-seconds` to recover interrupted spool records. Use
`--no-episode-summary` or `--no-candidate-summary` only when explicitly
debugging raw capture pressure.
`worker-loop` repeats the same worker pass with a compact per-iteration report.
Use `--iterations 1` under Task Scheduler/cron, or a higher iteration count for
a foreground loop while developing.
`worker-schedule` writes a reviewable Windows Task Scheduler install script for
running `worker-loop --iterations 1` periodically. It also writes matching
status and uninstall scripts next to the install script, and routes scheduled
worker output to `.ara-memory/logs/worker-task.log`. The install script runs the
same worker command once as a preflight and refuses to register the task if the
worker's recall cases fail. Baseline drift is reported as a warning for the
scheduled worker because normal capture/drain can legitimately add new relevant
memories between runs. It does not register the task until that script is run;
inspect the generated scripts before running them.
`worker-schedule-verify` checks that the install/status/uninstall scripts exist,
use structured worker arguments for `worker-loop --iterations 1`, keep
overlapping runs ignored, include valid recall regression gates, include the
registration preflight, keep baseline drift in warn-only mode for scheduled
runs, and stay within the configured interval budget. This is script-readiness
evidence only: after reviewing the scripts, install with
`.ara-memory/scripts/install-worker-task.ps1`,
inspect with `.ara-memory/scripts/status-worker-task.ps1`, and remove with
`.ara-memory/scripts/uninstall-worker-task.ps1`.
It does not prove that the task is installed, enabled, running under the intended
Windows account, or producing healthy worker logs. It also does not execute the
preflight; the install script does.
Schedule verification itself does not spend AI API tokens; it reads local scripts
and JSON regression files. Scheduled worker runs are local/deterministic by
default and only spend external model/API cost if an external advisor/model
wrapper is configured for a path the worker drains.

Always-on worker runbook:

```powershell
Set-Location -LiteralPath '<repo>'

python -m ara_memory health --scope ara-memory --query "current memory health" --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory worker-loop --scope ara-memory --iterations 1 --interval-seconds 0 --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json

python -m ara_memory worker-schedule --repo . --output .ara-memory/scripts/install-worker-task.ps1 --task-name AraMemoryWorker --interval-minutes 5 --scope ara-memory --regression-manifest examples/recall_regression_manifest.json --regression-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory worker-schedule-verify --output .ara-memory/scripts/install-worker-task.ps1 --scope ara-memory

powershell -NoProfile -ExecutionPolicy Bypass -File .\.ara-memory\scripts\install-worker-task.ps1
Start-ScheduledTask -TaskName AraMemoryWorker
powershell -NoProfile -ExecutionPolicy Bypass -File .\.ara-memory\scripts\status-worker-task.ps1

Stop-ScheduledTask -TaskName AraMemoryWorker
Disable-ScheduledTask -TaskName AraMemoryWorker
Enable-ScheduledTask -TaskName AraMemoryWorker

powershell -NoProfile -ExecutionPolicy Bypass -File .\.ara-memory\scripts\uninstall-worker-task.ps1
Get-ScheduledTask -TaskName AraMemoryWorker -ErrorAction SilentlyContinue
```

Do not install if `worker-schedule-verify` fails. If it reports missing
`WorkerArgs`, regenerate the scripts with current `worker-schedule` instead of
patching older scripts by hand. For recovery, stop or disable the task first,
inspect `.ara-memory/logs/worker-task.log`, run the same `worker-loop` command
in the foreground, inspect `.ara-memory/locks/worker.lock/owner.json`, lower
stale thresholds only after confirming no worker is alive, and inspect
`.ara-memory/spool/failed/` before replaying or rejecting failed envelopes.
`maintenance` runs lossless storage upkeep: FTS optimize, WAL checkpoint, SQLite
integrity check, and VACUUM.
`retention` reports status/kind distribution and cold memory candidates before
any destructive pruning is considered.
`retention-cycle` is the preferred non-destructive pruning readiness gate. It
creates a fresh backup, verifies it, exports cold capsules, verifies that export,
runs prune-plan with representative recall queries, and runs shadow-prune in a
restored sandbox. It does not modify the live store. The cycle backup defaults
to `--backup-archive-mode objects`, which preserves encrypted artifact objects
without recursively embedding older cold exports, retention-cycle reports, or
quarantined failed backup ZIPs. Use `--backup-archive-mode full` only when a forensic
snapshot of the entire archive tree is explicitly required. By default it takes
the same worker lock used by scheduled worker-loop runs, so backup/export/shadow
evidence is not generated while the background worker is mutating memory; use
`--no-lock` only when the store is otherwise quiescent. `--no-shadow` is allowed
only for partial evidence collection and does not pass pruning readiness. Each
run writes a compact report under `.ara-memory/archive/retention-cycles/`;
`health` uses the latest passing report to distinguish unmanaged cold pressure
from reviewed pruning readiness evidence, while `cold-stewardship` additionally
checks freshness and drift against the current live cold set. Retention-cycle
reports include a cold identity fingerprint over the selected cold capsule IDs,
cold source-event IDs, protected source-event IDs, and prunable source-event
IDs; old reports without that fingerprint are treated as stale for high cold
pressure.
`cold-export` writes superseded/rejected/quarantined capsules plus their source
events into a portable zip so pruning can later be audited. Restoring a pruned
store still requires a full verified backup, not a cold export alone. Verification
first bounds ZIP entry count, entry size, total expansion, and compression
ratio, then checks per-entry SHA-256 hashes, the local-key manifest HMAC, and
whether exported capsules reference source events that are missing from
`events.jsonl`. JSONL payloads are scanned line-by-line with per-file and
per-line limits before they are trusted. `--no-events` is therefore only for
inspection exports; prune gates require signed v2 cold exports with source
events included. Manual limited exports default to newest-first for inspection; use `--order oldest`
when preparing a manual export for `prune-plan`, because the planner chooses
the oldest eligible cold capsules first. `retention-cycle --limit` does this
oldest-first export internally so its cold export matches the planned capsule
set.
`prune-plan` is still a dry-run: it requires a verified cold export, runs
representative recall checks, and separates prunable source events from events
still cited by active candidate/stable capsules.
`shadow-prune` restores a backup into a temporary sandbox, deletes only planned
cold capsules inside that copy, then reruns doctor and recall checks. The live
store is not modified.
`prepare-live-prune` reruns shadow-prune and writes a short-lived approval token
to the store. `live-prune` consumes that token once, requires exact confirmation
text, re-verifies the approved backup and cold export, deletes cold capsules
only, and records the irreversible operation. Source events are not deleted by
live-prune.

Recall regression catches quiet retrieval drift before a memory change becomes
part of the operating loop:

```powershell
python -m ara_memory recall-regression --manifest examples/recall_regression_manifest.json --write-baseline .ara-memory/archive/recall-regression-baseline.json
python -m ara_memory recall-regression --manifest examples/recall_regression_manifest.json --baseline .ara-memory/archive/recall-regression-baseline.json
```

Each case checks expected terms against rendered memory evidence, excluding
query echo, hot memory, matched tags, and the separate graph-hints section.
Forbidden terms still scan the full pack. Graph activation paths count only when
they are attached internally to a selected/rendered capsule with positive
spreading score, but normal regression evidence still comes from rendered
capsule text rather than raw path text. Expected diagnostics can gate
`spreading_activation_used`, `graph_activation_edges_considered`, or other
`spreading_*` fields the same way ESPA diagnostics are gated. Cases must include
visible capsule evidence, so a selected capsule that is later trimmed away cannot
pass the gate. With a baseline, the gate also fails when a previously passing
case breaks, token use jumps, or the visible capsule set drifts too far; old
baselines fall back to selected capsule overlap. Regression details also record bounded selected and visible
`source_event_ids`, their full counts, truncation flags, and a digest of the full
source-event set, so consolidation can replace capsule IDs without failing the
gate when the same underlying evidence remains visible. Source-event overlap is
measured as prior-evidence coverage, not symmetric Jaccard, so additional
audited evidence does not fail a case when the baseline evidence is still
covered; JSON also reports `source_event_jaccard` for diagnostics. When lineage
details are truncated, local comparisons recompute the full source-event set
from capsule provenance plus provenance witnesses; if old capsule rows are
unavailable, the digest must still match or the lineage comparison is treated as
incomplete rather than silently passing.
JSON details include `capsules_visible`, `visible_capsule_ids`,
`visible_source_event_count`, `visible_source_event_digest`, `overlap_basis`,
`evidence_overlap`, `capsule_id_overlap`, `source_event_overlap`, and
`source_event_jaccard`. For budget-selection QA, run `recall-plan --json` and review `visible_capsules`,
`query_terms_visible_count`, `fallback_used`, and `quality_score`.

Create a routine portable snapshot after important milestones:

```powershell
python -m ara_memory backup --output .ara-memory/backups/milestone.zip --archive-mode objects
python -m ara_memory verify-backup .ara-memory/backups/milestone.zip
python -m ara_memory restore-drill .ara-memory/backups/milestone.zip --scope ara-memory --query "current memory architecture"
python -m ara_memory restore-backup .ara-memory/backups/milestone.zip --target-root .ara-memory-restored
```

Use a full forensic snapshot only when you intentionally want the entire archive
tree, including older cold exports, retention-cycle reports, and quarantined
failed backups:

```powershell
python -m ara_memory backup --output .ara-memory/backups/forensic-full.zip --archive-mode full
```

When verifying or restoring an archive copied from another memory root, pass the
source root explicitly:

```powershell
python -m ara_memory verify-backup copied/milestone.zip --trust-root path/to/source/.ara-memory
python -m ara_memory verify-cold-export copied/cold.zip --trust-root path/to/source/.ara-memory
```

Backups contain a SQLite-consistent `memory.db` snapshot, the append-only ledger,
hot memory files, durable spool envelopes, selected archive payload, and a
manifest with schema/stats plus per-entry SHA-256 hashes. Verification checks
every hashed ZIP entry, the manifest HMAC, SQLite integrity, and foreign-key
consistency so entry tampering, manifest rewrites without the local signing key,
and orphan rows do not pass. Backup and cold-export verification also reject
unsafe archive names, duplicate entries, oversized members, excessive total
uncompressed size, suspicious compression ratios, and filesystem-normalized path
collisions before hashing or parsing large members.
Default backups use `--archive-mode objects`: they include sha256-addressed
file/image artifacts encrypted at rest under `archive/objects` and the full
spool directory, including pending/done/failed envelopes, encrypted archive
snapshot references, legacy snapshots when present, and `.seal-key`, so sealed
pending work remains drainable after restore without recursively embedding
derived evidence bundles. Spool envelopes/path metadata, legacy snapshots, the
ledger, and SQLite rows can still contain plaintext local evidence. They
intentionally exclude older `archive/cold`
exports, `archive/retention-cycles` reports, and `archive/failed-backups` ZIPs.
Use `backup --archive-mode full` for an explicit forensic snapshot of the entire
archive tree. `backup --no-archive` or `--archive-mode none` omits archived
artifacts; do not use it when restored file/image artifacts are required. It is
rejected while pending or processing spool records still reference encrypted
archive-object snapshots, because those queued turns would not be drainable
after restore.
The backup signing key lives at `.ara-memory/.backup-signing-key` and is not
stored inside backup ZIPs. Preserve it as local trust material if old backups
must remain cryptographically verifiable on another machine.
The archive object encryption key lives at `.ara-memory/.archive-object-key`.
Backups never store that key in plaintext; when encrypted archive objects are
included, the manifest carries an encrypted key escrow wrapped by the local
backup signing key. Preserve the source trust root or `.backup-signing-key` when
copied backups must be verified and restored on another machine.
`restore-drill` restores into a temporary directory and can run a bounded recall
query, proving the snapshot is usable before any real restore or pruning.
Restore copies the source ZIP to a temporary snapshot and verifies/extracts that
same byte stream to reduce source-file swap risk. It refuses to overwrite a
non-empty target unless `--force` is passed; even then, force only clears known
memory-root paths and preserves `.backup-signing-key`.
Because retention-cycle creates verified backups as evidence, backup bytes can
dominate the local store even when live memory is small. Use
`backup-stewardship` to review redundant verified backups before touching live
memory:

```powershell
python -m ara_memory backup-stewardship --keep-latest 3 --keep-retention-cycles 2 --target-backup-bytes 67108864
python -m ara_memory backup-stewardship --keep-latest 3 --keep-retention-cycles 2 --target-backup-bytes 67108864 --no-cache-write
python -m ara_memory backup-stewardship --keep-latest 3 --keep-retention-cycles 2 --target-backup-bytes 67108864 --quarantine-failed
python -m ara_memory backup-stewardship --keep-latest 3 --keep-retention-cycles 2 --target-backup-bytes 67108864 --quarantine-failed --apply --quarantine-confirm "QUARANTINE FAILED BACKUPS"
python -m ara_memory backup-stewardship --keep-latest 3 --keep-retention-cycles 2 --target-backup-bytes 67108864 --apply --confirm "DELETE OLD BACKUPS"
```

The command is dry-run by default. The default CLI target is 64 MiB of backup
bytes, and candidate selection deletes only enough old redundant backups to move
toward that budget. Dry-runs update the backup verification cache by default so
repeated manual reviews are cheap; use `--no-cache-write` for read-only
diagnostic runs where the inspection itself must not mutate the store. Apply
mode takes the same worker lock used by scheduled
worker-loop and retention-cycle before it moves or deletes any backup file; use
`--no-lock` only when the store is otherwise quiescent. Apply mode deletes only verified backups that are neither
among the latest kept backups, nor referenced by recent passing retention-cycle
reports, nor referenced by active live-prune approvals. Failed-verification
backups are preserved for manual inspection instead of being silently removed.
When at least one verified backup exists, `--quarantine-failed` can move
failed-verification backup ZIPs to `.ara-memory/archive/failed-backups/` after
exact confirmation, preserving them outside the live backup pool so health
pressure reflects restorable backups instead of legacy or corrupted evidence.
Default `objects` backups do not re-embed those quarantined ZIPs; `full` backups
do, by design.
Quarantine and deletion confirmations are independent: a mixed run with only
the quarantine confirmation moves failed backups but leaves redundant verified
delete candidates untouched. Each quarantined ZIP gets a neighboring
`.quarantine.json` manifest with its original path, quarantine path, byte size,
timestamp, and final verification result.

## Design Choices

- SQLite + FTS5 first. No mandatory vector DB. FTS indexes candidate/stable
  capsules only; superseded, rejected, and quarantined memories stay preserved
  as cold evidence without paying active recall-index cost.
- Raw ledger is append-only JSONL.
- Exact duplicate events are deduplicated by content fingerprint after the
  pre-retain privacy guard and before they hit the ledger.
- Recognized hidden-reasoning fields and line-prefixed text are redacted from
  retained text events by default. Decisions and reasons are stored as explicit
  summaries with provenance. Archive object payloads are encrypted at rest;
  spool envelopes/path metadata, legacy snapshots, ledger entries, and SQLite
  rows remain private local evidence and may still contain plaintext.
- Long-term memories start as candidates and are promoted only after repeated
  evidence, high salience, or explicit user confirmation.
- Goals are first-class capsules, separate from decisions and procedures, so
  recall can answer why the work exists before choosing how to act.
- Lifecycle tiers separate long-running purpose, identity, and preference
  anchors from working, guarded, evidence, archive, and reject memory so token
  reduction is a policy choice rather than accidental truncation.
- Retrieval is graph/symbol/BM25 plus deterministic ESPA and bounded two-hop
  normalized relation spreading activation first, with optional embeddings left
  as a later extension point.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/SECURITY.md](docs/SECURITY.md) for the technical model and threat model.

## Codex Worktree Capture

```powershell
python -m ara_memory capture-worktree --cwd . --scope project --consolidate
python -m ara_memory recall "What changed in this project?" --scope project --budget 4000
```

This records `git status`, `git diff --stat`, and a compacted diff event. It is
intended for session-end capture, not for replacing commits or source control.
Untracked file contents are not captured by default. For a new project, opt in:

```powershell
python -m ara_memory capture-worktree --cwd . --scope project --include-untracked-content --consolidate
```

## Promotion And Rejection

Most extracted memories start as candidates. Promote only memories that should
affect future behavior.

```powershell
python -m ara_memory promote cap_xxxxxxxxxxxxxxxx
python -m ara_memory reject cap_xxxxxxxxxxxxxxxx --reason "poisoning-like instruction"
python -m ara_memory quarantine cap_xxxxxxxxxxxxxxxx --reason "untrusted behavioral instruction"
```

This is the first protection against memory poisoning and preference overfitting.

### Shared Promotion Gate

Changing a capsule to `stable` can change future behavior. Manual promotion,
sleep consolidation, review-worker apply mode, and external advisor results
therefore share the same final gate before a stable write. The gate reruns the
deterministic risk boundary and automatic promotion also requires either two
real source events, one trusted explicit source, or a summary backed by
consolidated source capsules plus at least one real source event. Missing source rows,
quarantined/rejected/superseded capsules, instruction-like text, secrets,
self-serving claims, and keyword stuffing block behavior-changing promotion.
Evidence summaries may still become stable when keyword repetition is the only
risk signal; hot/recall risk filters continue to keep them out of always-on
context.

Automatic promotion writes are conditional: a capsule must still be
`candidate` at write time. If another worker or operator quarantines, rejects,
or supersedes it after review but before write, the stale promotion cannot
resurrect it as stable.
Summary consolidation also checks source capsules before and during
supersession so a concurrent status change cannot be silently overwritten.

Inspect candidate memories before promoting:

```powershell
python -m ara_memory list --status candidate --limit 10
python -m ara_memory list --status quarantined --limit 10
python -m ara_memory actions
python -m ara_memory risk --scope project
python -m ara_memory review --scope project
```

The default advisor is deterministic and local. To route review decisions
through an external AI or local model wrapper, set:

```powershell
$env:ARA_MEMORY_ADVISOR = "external-command"
$env:ARA_MEMORY_ADVISOR_COMMAND = "python ./examples/advisor_rule_based.py"
python -m ara_memory review --scope project
```

The command receives a JSON object on stdin and must return:

```json
{
  "recommendations": [
    {
      "capsule_id": "cap_x",
      "action": "promote",
      "reason": "durable repeated project decision",
      "risk_score": 0.1
    }
  ]
}
```

Invalid, missing, or timed-out external recommendations fall back to the local
deterministic advisor.

See [examples/advisor_rule_based.py](examples/advisor_rule_based.py) for a
dependency-free command advisor and
[examples/advisor_openai_template.py](examples/advisor_openai_template.py) for
an OpenAI-backed template.

For strict project isolation, recall without global memories:

```powershell
python -m ara_memory recall "project context" --scope project --no-global
```

## Sleep Consolidation

Run this after a session to promote high-confidence operational memories, merge
repeated candidates, and flag possible conflicts.

```powershell
python -m ara_memory sleep --scope project --dry-run
python -m ara_memory sleep --scope project
python -m ara_memory sleep-runs
```

This is the first version of the separate Memory Curator. It is deliberately
local and deterministic; an AI curator can be added behind this interface later.
It also supersedes stale artifact summaries so repeated file captures do not keep
competing as separate long-term memories.
Before promotion, a separate deterministic Memory Auditor scores poisoning risk.
High-risk behavioral, secret-like, or self-serving memories are quarantined
instead of becoming stable memory. Capsule title, body, and tags are risk
scored; instruction-like, secret-like, or direct-identifier tags are not
rendered in recall tag surfaces. Direct identifiers and keyword-stuffed
capsules are also kept out of hot memory so the always-on context stays small
and harder to manipulate. The same deterministic boundary protects default
recall, manual promotion, quality review, and external advisor payloads.

Estimate local storage and downstream model input cost:

```powershell
python -m ara_memory estimate-cost "project context" --scope project --budget 4000 --input-usd-per-million 1.25
```

Run the built-in memory evaluation suite:

```powershell
python -m ara_memory eval
python -m ara_memory context-eval
```

This checks recall precision, scope isolation, poisoning quarantine, and hot
memory token discipline in a disposable synthetic store. `context-eval` adds cue
generalization, near-miss scope isolation, distractor resistance, tight-budget
recall, and hot+cold recall behavior.

## Hot Memory

Hot memory is a tiny Markdown state file compiled from stable capsules:

```powershell
python -m ara_memory hot --scope project --budget 1200
python -m ara_memory show-hot --scope project
python -m ara_memory recall "current project state" --scope project --hot --budget 2500
```

Use it as the always-on memory layer. It stores only compressed stable identity,
active goals, current project state, procedures, warnings, and decisions. Raw
episodes stay in the ledger and are recalled only when the query needs them.
