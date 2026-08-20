# Decision Log

Architecture decisions that are NOT recoverable from the code or the commit
history — the "why", and specifically the alternatives that were rejected and
the reason they were rejected. Newest phase last.

Format per entry: the decision, the alternative(s) not taken, and the tripwire
that should make us revisit it.

---

## Phase: Agentic Loop, Part A — state shape (agent/state.py)

**A1. `status` Literal includes `"new"` alongside the four loop states.**
The loop's own vocabulary is running/resolved/escalated/error, but a ticket
exists before the loop ever runs — `POST /tickets` constructs a TaskState with
no diagnosis attached. Rejected: making `"running"` the initial state, which
would have meant the API reports a loop in progress when nothing is running.

**A2. `iteration` replaced `iteration_count`; the old field was removed, not kept.**
Two counters that can disagree is a defect waiting for Part B to trip over.

**A3. `trajectory` retyped `list[str]` -> `list[dict]`.**
The loop needs structured per-step records (thought / tool_call / observation /
hypothesis_after) for the eval harness to score trajectory quality, not just a
list of event names.

**A4. `called_tool_signatures` is `set[tuple[str, str]]`, not bare `set[tuple]`.**
Tool arguments are dicts and therefore unhashable, so the signature has to be
`(tool_name, json.dumps(args, sort_keys=True))`. A bare `tuple` widens to
`tuple[Any, ...]` under pydantic and would admit an unhashable element that
only fails later, at insert time.

**A5. `called_tool_signatures` is `exclude=True`.**
It is internal loop-guard bookkeeping, not API surface. Verified excluded from
`model_dump()`, `model_dump_json()`, the live `POST /tickets` response body, and
the generated OpenAPI schema.
*Tripwire:* if we ever want to debug a run through the API, this is the field to
expose — reconsider then.

**Known trap for later:** pydantic v2's `model_copy()` is SHALLOW. `plan`,
`trajectory`, `evidence_sources`, and `called_tool_signatures` are shared by
reference with any copy. If the loop ever snapshots state per iteration via
`model_copy()`, mutating the copy silently mutates the original — and
`called_tool_signatures` is exactly the field where that corruption would go
unnoticed. Either pass `deep=True` or mutate in place.

---

## Phase: Agentic Loop, Part B — orchestrator (agent/orchestrator.py)

Context: these five were explicitly delegated to me rather than decided by the
project owner, so the reasoning is recorded here in full.

**B1. The tool-invocation layer was EXTRACTED to `agent/tool_executor.py`,
shared by both the baseline and the loop.**
`TOOL_REGISTRY` / `_execute_tool_call` originally lived in
`agent/single_pass.py`, including the ticket_id-overwrite injection defense.
Rejected: having the orchestrator import from single_pass — the baseline exists
to be *measured against* the loop (docs/design.md Step 15), so making it a
dependency of the loop inverts that relationship and means a future edit to the
baseline silently changes the thing it is the control for. Also rejected:
duplicating the executor, which would have forked the injection defense into two
copies that can drift apart. Constraint accepted with the extraction:
single_pass's observable behavior must not change at all.

**B2. The critic is a separate LLM call returning strict JSON, validated by a
pydantic `Assessment` model, with exactly one re-ask on malformed output.**
This follows docs/design.md's "lightweight critic prompt" and its existing
"malformed plans are auto-repaired via a single re-ask" convention. Rejected:
forcing structure through a tool schema — `tool_choice` is already documented as
misbehaving on openai/gpt-oss-120b via Groq (see the comment block at the end of
`single_pass.run_single_pass`: both `tool_choice="none"` and omitting tools
produce a 400). Accepted cost: 2 LLM calls per iteration, so 16 in the worst
case. On a second parse failure the run does NOT crash and does NOT invent a
confidence — it leaves the belief unchanged and records `assessment_error`.

**B3. An evidence source id is the bare TOOL NAME, appended only when the critic
returns `supports: true`, deduplicated.**
So "at least 2 independent sources" means two *different* tools agreed.
Rejected: per-call source ids, which would let two `query_logs` calls
self-confirm a single hypothesis — precisely the failure mode the >=2 rule exists
to prevent. The stop-resolve condition is a strict conjunction
(`confidence >= 0.75 AND len(evidence_sources) >= 2`), including on the
no-tool-call text path: a confident-sounding final answer backed by one source
does not resolve.

**B4. design.md's third escalate condition — "all available tools tried and
confidence still < 0.75" — is deliberately NOT implemented yet.**
Only `query_logs` and `query_metrics` exist; RAG (`search_runbooks`) and memory
(`search_past_incidents`) are not built. Implementing it today would cap every
run at 2 iterations and make the loop behave indistinguishably from the
single-pass baseline it is supposed to outperform — i.e. it would corrupt the
headline eval result.
*Tripwire:* add this condition in the same change that lands the 4th tool.

**B5. The consecutive-no-new-info counter is a LOCAL variable in
`run_agent_loop`, mirrored into each trajectory entry as `"no_new_info": bool`.**
Rejected: adding a counter field to TaskState, which had just been frozen in
Part A and is the API response model. Mirroring it into the trajectory keeps the
escalation decision auditable after the fact without widening the public state.

**B6. A loop-guard hit CONSUMES an iteration.**
The guarded call is not executed and the observation becomes a "skipped" note,
but `iteration` still advances. Without this, a model stuck re-requesting one
call would never reach the iteration cap and the run would not terminate.

**B7. All tool calls in one model turn are processed as ONE iteration.**
The model may emit several tool calls per turn. The OpenAI protocol requires a
tool message for every `tool_call.id` — omitting any is a 400 — so all of them
must be answered, including guard-skipped ones. Rejected: executing only the
first and dropping the rest, which breaks the protocol. The critic then runs
once over the whole round.

**B8. The critic runs on a COPY of the message list.**
Its JSON turn is kept out of the main tool-calling conversation; only a short
hypothesis/confidence note is fed back in. Keeps the acting context clean while
still ensuring the next turn knows the current belief rather than silently
forgetting it (docs/design.md: replanning must tell the model the previous
hypothesis was likely wrong).

**B9. The critic's `supports: false` is surfaced to the model, not just used to
gate evidence.**
Found in main-session review of the first implementation: `supports` gated
`evidence_sources` correctly, but the note fed back into the conversation read
identically whether the round confirmed or refuted the hypothesis. That is
exactly the case docs/design.md's replan trigger calls out — "the next LLM call
should be told the previous hypothesis was likely wrong, not just silently
forgotten." A contradicting round now leads with an explicit warning. This was a
gap in the spec I wrote, not in the implementation of it.

**B10. The one act-call retry sleeps `RETRY_BACKOFF_SECONDS` first.**
docs/design.md specifies "retry once with backoff"; the first cut retried
immediately. The critic's re-ask does NOT sleep — a malformed-JSON re-ask is a
format correction, not a failure retry, and the likely real-world failure being
backed off here is a Groq rate limit.

### Accepted risks, Part B (reviewed, deliberately not fixed)

- **The critic call is handed the full TOOL_SCHEMAS at `tool_choice="auto"`,**
  so only the prompt stops it from emitting a real tool call instead of JSON.
  Degrades safely today (content=None -> parse fails -> one re-ask ->
  `assessment_error`, never an invented confidence), but it burns the re-ask on
  a non-JSON-error cause, and it is untested against a live model since all
  orchestrator tests stub the LLM. Cannot simply drop `tools` from the critic
  call: the documented Groq 400 (see single_pass.py) fires when the conversation
  already contains tool_call/tool messages.
  *Tripwire:* if live runs show critic tool-calls, add a critic-only schema.
- **Trajectory `tool_call.arguments` has a shape asymmetry:** an executed call
  records the args including the executor-injected `ticket_id`; a guard-skipped
  call records the model's raw args without it. Nothing reads this field for
  control flow — it is an audit record — and making them uniform would mean
  either claiming a ticket_id was used on a call that never ran, or dropping it
  from the record of calls that did.

---

## Phase: Agentic Loop, Part B — live-run corrections

First real Groq run (2026-08-18) invalidated one of the accepted risks above.
Recording the evidence, because the failure is not reproducible from the stubbed
test suite and would otherwise look like an arbitrary redesign.

**Observed:** on ticket T001 the critic tried to return its JSON *as a tool call
named `json`*, and Groq rejected the whole request:
`400 tool_use_failed: attempted to call tool 'json' which was not in
request.tools`. Both the ask and the single re-ask failed the same way. The
prediction in "Accepted risks" above — that this would degrade quietly to
`content=None` — was WRONG: it is a hard 400, not a soft parse failure.

**Consequence:** T001 finished `escalated` at the iteration cap with confidence
0.94, a correct hypothesis, and one evidence source, having called tools
successfully seven times. The critic died on the only round that ran
`query_logs`, so that tool was never credited, and the remaining rounds all ran
`query_metrics`, which dedupes to a single source. A correct diagnosis lost on a
technicality.

**B11. The critic now runs on an isolated, tools-FREE context.** Not the main
conversation plus TOOL_SCHEMAS — a purpose-built two-message list (critic system
prompt + a rendered digest of the run's observations) called with `tools=[]`.
`agent/llm.py` only sets the tools/tool_choice kwargs `if tools:`, so an empty
list omits them; that is safe here specifically because this conversation
contains no tool_call/tool messages, which is the condition that triggers the
documented Groq 400 in the baseline. Supersedes B8 (critic on a copy of the main
messages) and closes accepted-risk #1. Side benefit: the critic stops re-sending
a transcript that grows every round, which is most of the per-call latency.
The untrusted-data framing is repeated in the critic system prompt — the digest
embeds observation text, so the injection surface moves with it.

**B12. Evidence is credited across the whole run, not just the current round.**
When the critic returns `supports: true`, every distinct tool that returned
`status == "ok"` at any point this run is credited. Consistent with what the
critic is actually asked ("the observations so far this run"), and it makes a
single failed critic round non-fatal instead of permanently capping the evidence
set. B3 otherwise stands: only "ok" counts, the id is still the bare tool name,
and `supports: false` still credits nothing.

**Also measured:** 100-250s per ticket end-to-end; individual calls slow from
0.6s to 10-25s purely from transcript growth. And the loop guard cannot stop the
model burning iterations on five different `query_metrics` metric names — the
signatures genuinely differ, and they all collapse to one evidence source.
*Tripwire:* if that pattern persists after B11/B12, consider a per-tool call
budget rather than a stricter guard.

---

## Phase: RAG Layer, Part A — ingestion (rag/ingest.py)

**C1. `## Category` is hoisted into metadata; it does not become a chunk.**
Every runbook has six `##` sections, not the five the phase brief listed. The
sixth, `Category`, has a one-word body ("auth", "db"). Chunking it uniformly
with the rest would put six near-content-free vectors into the same similarity
space the retrieval contract thresholds at 0.5 — a query mentioning "auth" could
win a top-3 slot with a document containing literally the word "auth" and
nothing else, displacing a real Symptoms or Fix chunk. Instead the value rides
along as `category` metadata on all five chunks of its document. Rejected: the
uniform "every `##` is a chunk" rule (36 chunks), which is a simpler parser but
pays for that simplicity in retrieval quality at exactly the moment retrieval
matters.
*Correction to the original framing:* this was written expecting `category` to
be a Part B retrieval filter driven by the ticket. It cannot be — `category` on
a ticket in data/tickets.json labels the reasoning shape the ticket is meant to
exercise (`easy` x5, `multi_step` x5, `tool_heavy` x3, `ambiguous` x2), not the
subsystem, so there is nothing on the ticket to filter runbook category by. The metadata is still the right
call on the "don't embed a one-word document" argument alone, and it stays
available for a future filter, but Part B does not use it.
*Tripwire:* if Category bodies ever grow into prose, revisit — the argument is
about one-word documents, not about the section being unimportant.

**C2. Chunk text is prefixed `"{title} — {section}"`, not the raw section body.**
`## Root Cause` bodies are a single snake_case token
(`auth_signing_key_mismatch`) and `## Constraints` bodies are context-free
without knowing what they constrain. Embedding the bare body throws away the
only signal that situates it. Rejected: storing raw bodies and relying on
metadata for context, which metadata cannot do — metadata is not embedded.
The unprefixed body is still kept on the `Chunk` as `body` for callers that want
to render it.

**C3. The collection carries the embedding function; ingestion does not embed manually.**
`SentenceTransformerEmbeddingFunction` is attached at `create_collection` time,
so Part B's `search_runbooks` embeds its queries with the same model without
having to know which model that was. Rejected: computing vectors in `ingest.py`
and passing `embeddings=` to `add()`, which works but lets the query side drift
onto a different model — a failure that produces plausible-looking garbage
rather than an error.

**C4. Cosine space, explicitly set.**
The Session 6 retrieval contract escalates when the top score is `< 0.5`. That
threshold only means something on a bounded metric; chroma's default is L2,
which is unbounded and would make 0.5 an arbitrary number. Set via
`metadata={"hnsw:space": "cosine"}`.

**C5. Idempotency rests on delete-then-create, with `upsert` as the actual backstop.**
`build_index` deletes the collection and rebuilds it, so `python -m rag.ingest`
is a rebuild rather than an append. Chunk ids are deterministic
(`{doc_id}::{section}`) as a second line of defense — but that only holds with
`upsert`. Measured on chromadb 1.5.9: `add()` with an already-present id neither
raises nor updates, it silently no-ops, so a skipped delete would have meant a
runbook edit silently failing to reach the index with no error and no count
change. Rejected: `add()` plus the deterministic ids, which reads like
defense-in-depth and is not.

**C6. First-ever build catches `chromadb.errors.NotFoundError`, not `ValueError`.**
On chromadb 1.5.9 `delete_collection` on a missing collection raises
`NotFoundError`, which is not a `ValueError` — catching the latter meant the
very first build on a clean machine crashed. Caught narrowly so that any other
chroma failure still surfaces.
*Tripwire:* this is version-coupled; recheck on a chromadb major bump.

**Known latent fragility:** the section splitter is a line-anchored `^## ` regex
with no fenced-code awareness. A runbook that ever shows example markdown inside
a ``` fence would have that fence's `## ` lines split as real section headers.
None of the six runbooks contain fenced code today. Duplicate and missing
sections, missing H1, and a missing Category all raise instead of degrading.

---

## Phase: RAG Layer, Part B — retrieval + tool (rag/retrieve.py)

**D1. `score = 1.0 - distance`, and the conversion is the reason C4 mattered.**
Chroma returns *distances*, not similarities. Under cosine space that distance
is `1 - cosine_similarity`, so the Session 6 contract's "top score < 0.5" only
means what it says after converting back. Had the collection been left on the
default L2 metric (see C4), this subtraction would have produced a number that
looks like a similarity and is not one — the worst kind of wrong, since nothing
would have crashed.

**D2. `search_runbooks` returns the repo's `{status, data, summary}` shape, not
the bare `{doc_id, section, text, score}` list the contract names.**
The contract describes the per-chunk record; the envelope has to match what
`agent/tool_executor.py` and `agent/orchestrator.py` already consume —
the orchestrator credits evidence on `status == "ok"` and drives its
no-new-info counter off the same field. The contract's four keys are the shape
of each element of `data["chunks"]`. Rejected: returning the raw list and
special-casing this tool in the executor.

**D3. `no_confident_match` needed no orchestrator change, which was luck worth
recording.** PROGRESS previously described the no-new-info escalation as keying
on `status == "empty"`. It does not — it keys on `any_new_ok`, i.e. whether any
tool returned `"ok"`. So a `no_confident_match` already credits zero evidence
and increments the no-new-info counter, exactly as the contract wants, with no
edit to the loop. *Tripwire:* if that counter is ever rewritten to enumerate
specific statuses, `no_confident_match` must be in the list.

**D4. A weak match still returns its chunks, alongside the refusing status.**
`no_confident_match` carries the rejected chunks and `top_score` in `data`, so
calibration and a human can see how close it got. The status, not the absence of
data, is what tells the agent not to act. Rejected: returning empty data on a
weak match, which would have made eval/calibrate_retrieval.py impossible to
write without a second private code path.

**D5. `ticket_id` injection is now gated by `TICKET_SCOPED_TOOLS`, and
non-scoped tools have it *stripped* rather than merely not injected.**
`search_runbooks` takes no `ticket_id`, so the old unconditional
`args["ticket_id"] = ticket_id` would have `TypeError`d on every call. Stripping
rather than passing through preserves the original defense: the model has no
business naming a ticket for a tool that isn't ticket-scoped. Note the failure
mode is fail-closed — a tool registered but forgotten in `TICKET_SCOPED_TOOLS`
loses its injection (a visible TypeError) rather than accepting a model-supplied
id. An invariant test pins the two sets against drift.

**D6. Query-time embedding function must be passed explicitly to
`get_collection` — but not for the reason first recorded here.**
`get_collection` does not restore the embedding function used at
`create_collection` time; it defaults to chroma's own
`DefaultEmbeddingFunction`. The original entry claimed omitting it would embed
queries with a different model and silently produce meaningless scores. That is
**wrong as stated**, and the corrected measurements on chromadb 1.5.9 are:

- Omitting the argument scores *identically* (0.6922 on the pool-exhaustion
  probe) — because chroma's default is ALSO all-MiniLM-L6-v2, just via its
  bundled onnx runtime instead of sentence-transformers. It works by
  coincidence, not by design, and the coincidence ends the moment
  `EMBEDDING_MODEL` in rag/ingest.py changes.
- Passing a genuinely *different* model is accepted SILENTLY. Reopening the
  same collection with `paraphrase-MiniLM-L3-v2` returned score 0.18 and the
  wrong document where the correct model returns 0.69 — no exception, no
  conflict guard, no warning.

So the explicit pass is still right, but it is protecting against model drift
between ingest and query rather than against a wrong default today. The real
hazard is the second bullet: chroma will not stop you from querying an index
with the wrong model. C3's "attach it to the collection" framing remains only
half true — the attachment does not survive a reopen, so both sides must name
the model.
*Tripwire:* recheck on a chromadb major bump, and re-measure if
`EMBEDDING_MODEL` ever changes — that is the moment the omit-the-arg path stops
being harmless.

**D7. The retrieval cache repairs itself instead of being invalidated by ingest.**
`_get_collection` caches the client+collection because re-instantiating the
sentence-transformers embedding function costs seconds and the loop calls
`search_runbooks` repeatedly. But `build_index` deletes and recreates the
collection, giving it a new internal UUID and leaving any cached handle dead —
verified to raise an unhandled `NotFoundError` on the next query. Fixed by
evicting and retrying once on `NotFoundError` at query time. Rejected: having
`build_index` call `_reset_cache()`, which is circular (`retrieve` imports
`ingest`) and cannot help when a *different* process rebuilds the index under a
long-lived server.

**D8. `SCORE_THRESHOLD` stays at 0.5 despite calibration not validating it.**
`eval/calibrate_retrieval.py` over all 15 tickets: Recall@1 = Recall@3 = 14/15,
zero tickets fall below 0.5, and every candidate threshold from 0.30 to 0.55
admits the identical 15/14 split at no cost. But the correct and incorrect
top-1 score distributions OVERLAP — the single wrong retrieval (T015) scored
0.5739, inside the correct range of 0.5511-0.7992 — so no threshold separates
them and 0.5 is an assumption this data fails to contradict rather than one it
confirms. Rejected: tuning the constant to fit 15 tickets over 6 runbooks, which
would overfit, and would tune against the very data Part C is evaluated on.
*Tripwire:* revisit with a materially larger ticket set, or if any real ticket
starts landing near the cut — the margin is thin (min correct 0.5511, and a
merely-rephrased cache symptom measured 0.4933).

**The failure mode the threshold cannot catch:** T015 retrieves the wrong
runbook *confidently*. Its top score sits in the healthy range, so no cut
rejects it. Guarding against a confident-wrong retrieval is Part C's problem —
the agent has to disconfirm rank 1 against logs/metrics rather than trust it,
which is exactly why the design requires >=2 independent evidence sources.

---

## Phase: Memory Layer (vectorstore.py, memory/ingest.py, memory/store.py)

**E1. Chroma plumbing extracted to a top-level `vectorstore.py` before writing
`memory/`, rather than copying `rag/`.** The shared surface is the cached
client/collection, the explicit embedding function on reopen, the
evict-and-retry-once query, the cosine->similarity conversion, and the
delete+upsert rebuild. Two of those exist only because D6 and D7 were found the
hard way; duplicating them would have meant a future fix landing in one layer
and silently missing the other. Rejected: a standalone parallel `memory/` layer,
which is lower-risk in the moment and worse in six weeks.
Verified no-behavior-change by the T015 watched case still reporting UNCHANGED at
0.5739/rank 4 — a refactor of retrieval plumbing that moves no scores is a claim
worth having evidence for.

**E2. Runbook-specific logic deliberately did NOT move into `vectorstore.py`.**
Markdown parsing, the fence scanner, `Chunk`, the Category rule, collection
names, thresholds, the `{status, data, summary}` envelopes and every summary
string stay in their own layer. `vectorstore.py` imports from neither `rag` nor
`memory` — the dependency runs one way only.

**E3. Only `symptom_summary` is embedded; root cause and resolution are metadata.**
The tool is looked up by observed symptoms, so the vector must represent
symptoms. Embedding the resolution text would let a query match on fix wording
rather than on what was observed, which is the wrong retrieval axis for a tool
whose input is a description of a live incident.

**E4. The two doubled root causes are deliberate, and they are the confusable pair.**
`data/past_incidents.json` carries two `db_connection_pool_exhaustion` and two
`network_ingress_queue_exhaustion` incidents. That is the exact pair the runbook
layer already confuses (T015 retrieves the DB runbook when the gold is the
network one), so the memory layer's discrimination test has something real to
separate rather than six trivially-distinct classes. Measured: a DB-flavored
query scores 0.5574 db vs 0.4099 network; a network-flavored query scores 0.7423
network vs 0.3369 db. It separates.

**E5. Memory's gate is 0.40, NOT the 0.5 mirrored from runbooks, and the two
constants stay independent.** Measured over the 13 gold-bearing tickets:
Recall@3 = 13/13, Recall@1 = 11/13, correct top-1 scores spanning 0.3244-0.6170
with a median of 0.5719 — against the runbook layer's 0.5511-0.7992 / 0.7171.
Memory scores are structurally lower because a ticket-to-incident match is short
symptom-paraphrase against short symptom-paraphrase, where a runbook match is a
query against a long prose section. At 0.5 the gate rejected 4 of the 13 tickets
whose gold incident was present in top-3 (T001 0.4824, T002 0.3244, T003 0.4944,
T007 0.3816) — where the same 0.5 rejects 0/15 on runbooks. The sweep at 0.40
admits 11 with 10 correct and wrongly rejects 1 correct top-1, against 3 at 0.50.
Rejected: keeping 0.5 for cross-layer consistency, which would ship a gate
measured to suppress correct answers on ~30% of gold tickets; and sharing one
constant in `vectorstore.py`, which the scale difference makes incoherent.
*The specific justification for a looser gate on memory and not on runbooks:*
the contract already makes a memory hit a HINT that must be independently
verified via query_logs/query_metrics before its root cause is adopted. The gate
is not the safeguard for this tool — verification is. Suppressing a correct prior
incident costs more than surfacing a weak one the agent is required to check
anyway. That argument does NOT extend to search_runbooks, where a retrieved fix
can be acted on more directly.

**E6. The threshold cannot be tuned to fix both failure modes, and both are pinned.**
CORRECT and WRONG score ranges overlap on memory too — a wrong top-1 scored
0.5981, above the correct median of 0.5719 — so no cut separates them here
either. The two watched cases in eval/calibrate_retrieval.py now record opposite
failures: T015 is a WRONG retrieval scoring inside the correct range (no
threshold can reject it), T002 is a CORRECT retrieval scoring far below any
plausible gate (no threshold can admit it without admitting noise). Anyone
reaching for the threshold to fix either one should read both first.

**E7. `tests/test_ticket_dataset.py`'s runbook join was unfalsifiable and is fixed.**
It parsed root causes from the runbook files and then `.update()`d the parsed set
with the same six strings as literals, so `gold_root_cause in runbook_roots`
passed regardless of what the files contained; there was also a dead no-op loop.
Both removed, plus an `assert len(runbook_roots) == 6` so a parse failure
surfaces instead of vacuously passing. Verified falsifiable by renaming a root
cause in an isolated copy of the tree: the ticket join and both new incident
joins now fail, where previously the ticket one passed.
*Tripwire:* any assertion of the form `x in <set built from the data x came
from>` deserves this same suspicion.

---

## Phase: Loop Integration (agent/orchestrator.py)

Context: this phase contains the FIRST live Groq runs of the agent loop. Every
prior phase's loop tests were stubbed, so several defects below had been sitting
in committed, "passing" code since Session 5.

**F1. The loop could never resolve, and stubbed tests could not have caught it.**
`state.hypothesis` starts None; `_build_critic_digest` therefore renders
"Current hypothesis: none yet"; `CRITIC_INSTRUCTION` asked the critic to assess
"the current hypothesis". Shown none, the live model correctly refused and
returned `supports=false, confidence=0.0, hypothesis="No hypothesis defined yet"`
— which Step 5 then wrote into `state.hypothesis`, so every later round assessed
the literal sentence "No hypothesis defined yet". `supports` never became true,
`_credit_evidence` never ran, and every run escalated at the iteration cap
regardless of evidence. Measured on T001: 8 successful tool calls,
`evidence_sources=[]`, confidence 0.0.
*The lesson worth keeping:* 172 tests passed throughout, because all of them
stubbed the critic's response. The tests pinned our model of the critic, not the
critic. A live smoke run is not optional for an LLM-in-the-loop system.

**F2. Fixed by instructing inference, not by plumbing the act model's text.**
The first attempt passed `message.content` as `answer_text`. It looked verified —
substituting a hand-written act text flipped the same digest from
supports=false/0.0 to supports=true/0.95 — but the verification tested the
mechanism without checking its precondition. Measured after: gpt-oss-120b returns
`content=None` whenever it emits tool calls, putting its reasoning in a separate
field, so the argument is always "". The real fix is `CRITIC_INSTRUCTION`
telling the critic to INFER the best-supported hypothesis when none is stated.
Rejected: reading the provider-specific `message.reasoning`, which would couple
the loop to a non-standard field and silently reintroduce the deadlock on a model
repin. The `answer_text` pass-through was kept — correct for providers that do
populate content, explicitly documented as not load-bearing.

**F3. Inference had to be constrained in the same breath.**
Telling the critic to infer rather than refuse produced a false resolve on the
first try: T015 (`ambiguous`, `gold_root_cause: null`, `expected_behavior:
escalate`) resolved at confidence 0.78 on a hypothesis naming no mechanism
("occasional upstream service latency or cache miss bursts"), after its ERROR
logs returned `empty`. `CRITIC_INSTRUCTION` now also states that an empty or
no-match observation is absence of evidence and must LOWER confidence, and that
a hypothesis naming no concrete checkable mechanism gets low confidence even when
uncontradicted. The two rules are orthogonal — one governs the `hypothesis`
field, the other `confidence` — so they do not fight. T015 subsequently
escalated at 0.35.

**F4. `OBSERVATIONAL_TOOLS`: retrieval alone cannot establish what is happening now.**
`_credit_evidence` credits by bare tool name, so `search_runbooks` +
`search_past_incidents` both returning "ok" satisfied MIN_EVIDENCE_SOURCES=2 with
zero observation of the incident — contradicting design.md's rule that a memory
hit must be verified via logs/metrics. `_can_resolve` now additionally requires
at least one credited source from {query_logs, query_metrics}. T015 is the reason
this is not theoretical: a confidently-wrong runbook match plus a memory hit that
agrees with it is two "independent" sources agreeing on the same error.

**F5. `_can_resolve` also requires a credited `search_runbooks`.**
Live, once the deadlock was fixed, the agent resolved T001 and T009 in 2
iterations on logs+metrics alone and never consulted a runbook at all — the RAG
layer was reachable and unused, because the stop condition was satisfied before
the agent had any reason to reach for it. A prompt line alone was rejected: the
prior two prompt edits showed the model reliably optimizes for the stop
condition, and a prompt competing with a satisfied stop condition loses.
Because `_credit_evidence` only credits `status == "ok"`, and `search_runbooks`
returns `no_confident_match` below its gate, a ticket with no confident runbook
match now CANNOT resolve. That is the code-level expression of "escalate, do not
fabricate a fix from a bad match".

**F6. What the runbook requirement does NOT guarantee — stated plainly.**
It guarantees *a* runbook was consulted and matched confidently. It does not
guarantee the RIGHT runbook. T015 escalated live because the agent happened to
query "slow upstream call exceeded latency budget", scoring 0.43, below the gate.
eval/calibrate_retrieval.py measures T015's own ticket text retrieving RB-DB-001
at 0.5739 — ABOVE the gate. Had the model phrased its query closer to the ticket
text, the wrong runbook would have been credited and the run could have resolved
incorrectly. The escalation is emergent from query phrasing, not enforced.
*Tripwire:* this is the same confidently-wrong-retrieval failure watched case
T015 exists to record. Do not read "T015 escalated" as evidence the design
prevents it.

**F7. FIXED — citations are structured and verified, not requested.**
(Original finding, kept for the record, followed by what was done.)
The hypothesis text is written by the CRITIC, not the act model. The doc_id
instruction was added to `SYSTEM_PROMPT`, which only the act model sees, so it
lands on the wrong model and no citation appears in the output. `TaskState` also
has no final-answer or citation field, and design.md's verifier pass that checks
citation presence does not exist. Fixable cheaply by requiring the citation in
`CRITIC_INSTRUCTION` (the critic does see the runbook observations in its
digest), but deliberately not done as a fourth tuning round on n=1 evidence.

**Method note.** Every fix in this phase was validated on single live runs of
1-3 tickets. Each was correct for the defect it targeted and revealed the next
one, which is a good sign — but n=1 cannot separate a real behavioral change
from model variance at temperature 0 with a changing prompt. None of F1-F6
should be considered measured until the eval harness runs all 15 tickets.

**F8. Citations: enforced in code, because asking has lost three times this phase.**
`Assessment` and `TaskState` both gained `citations: list[str]`. `_can_resolve`
now additionally requires at least one citation AND that EVERY citation is a
doc_id actually returned this run by a `search_runbooks` observation with status
"ok" — a fabricated doc_id blocks the resolve rather than being ignored. This is
design.md's "verifier checks citation presence" pass, which had never been built.
Rejected: asking for the doc_id in prose inside the hypothesis, which cannot be
verified in code and is exactly the kind of request the model has repeatedly
optimized away from.

**F9. The citation feature would have silently done nothing.**
`_format_observation` JSON-dumps an observation's `data` and truncates at 500
chars. A `search_runbooks` observation's `data.chunks[*].text` holds whole
runbook sections, so the `doc_id` fields were being cut off before the critic
ever saw them — the critic could not have cited correctly even when willing, and
would have returned empty citations or guessed. Fixed by rendering a compact
`doc_ids=[...]` list ahead of the truncated blob. Rejected: raising the 500-char
cap, which exists to bound a digest that already grows every round.
*The general lesson:* a feature that depends on the model SEEING something needs
a check that the something survives the pipeline. Verified live: the digest now
carries `doc_ids=["RB-DB-001"]`, and the critic returns `citations=["RB-DB-001"]`
with reasoning that names the runbook.

**F10. RESOLVED — see G1 below. Original finding kept for the record.**
The Groq free tier is 200k tokens/day and one loop run costs ~20k; this phase's
live testing exhausted the daily quota, after which the loop correctly reported
`status="error"` on every ticket (graceful, no crash, no invented data — the
error path works). The citation mechanism was therefore verified at the component
level only: the digest surfaces doc_ids, the critic emits structured citations,
`_parse_assessment` accepts them, and `_can_resolve`'s fabrication guard is
covered by offline tests. What has NOT been observed is a full live run resolving
with a populated `state.citations`. One earlier critic call during the rate-limit
window returned an unparseable reply; `finish_reason` was not captured, so it is
recorded as an unexplained one-off to watch rather than diagnosed.
*Tripwire:* re-run T001/T009/T015 live once quota resets, before trusting F8.

---

## Phase: Provider portability (agent/llm.py)

**G1. F10 closed: the citation path is verified end-to-end, live.**
T001 resolved at 0.95 with `citations=['RB-DB-001']`; T009 resolved at 0.95 with
`citations=['RB-DISK-001']`; T015 escalated at 0.60 with `citations=[]`. T015 is
the load-bearing case: it gathered three evidence sources including two
observational, but `search_runbooks` returned `no_confident_match`, so no runbook
was credited, so no citation existed, so `_can_resolve` refused and it escalated
at the cap. The gate is enforcement, not persuasion.

**G2. The provider is now configurable; the MODEL stays pinned.**
`agent/llm.py` moved from the `groq` SDK to the `openai` SDK against a
configurable `LLM_BASE_URL`, with `LLM_API_KEY` (falling back to `GROQ_API_KEY`)
and `LLM_MODEL`. Defaults preserve the existing Groq behavior, so an unchanged
`.env` keeps working. Driven by the Groq free tier's 200k/day limit being
exhausted with no paid tier available.
*The distinction that made this cheap:* `gpt-oss-120b` is open-weights, so
switching HOST costs a base_url while switching MODEL would have invalidated
every live finding in F1-F9. Rejected: adding Gemini, which would have meant a
real multi-provider abstraction plus re-validating all the critic tuning against
different tool-calling and JSON-mode semantics.

**G3. Same weights do not guarantee same behavior — so it was acceptance-tested.**
Serving stacks differ in chat templates and function-calling implementations, and
F1-F3 are the record of how fragile that surface is. Before trusting the switch,
three checks were run against the new endpoint: tool calls returned for the four
schemas with correct args and no `ticket_id` (executor-injection defense intact);
`message.content` is None on tool-calling turns (same quirk as Groq, so the F2
reasoning holds); and the critic returns parseable bare JSON including
`citations`. All three passed before any loop was run.
*Tripwire:* re-run those three checks on any future provider or model change.
They cost ~2k tokens and would have caught F1 and F2 immediately.

**G4. Latency benchmarks were the wrong metric for this workload.**
A ~1.17s time-to-first-token figure nearly drove the provider choice. Measured
end-to-end, the new host ran the same three tickets in 19s/8s/56s against the old
host's 33s/74s/316s — 3-5x FASTER despite the worse TTFT number. This is a batch
eval with no human waiting; per-call TTFT is noise next to throughput, iteration
count, and the transcript growth already documented in B11.
*If wall-clock ever matters:* the 15 tickets are independent, so the eval harness
should fan out concurrently. That beats any per-call latency tuning.

**G5. Operational: the spend ceiling moved from hard to soft.**
The Groq free tier STOPPED work at 200k tokens/day, which is a crude but
effective budget guard. Per-token billing does not stop. At ~20k tokens per
ticket run, a 15-ticket sweep is ~300k, and a day of loop debugging consumed
about that much. Set a provider-side spend limit before running sweeps.
*Also:* the key used to enable this was pasted into a chat transcript and must be
rotated. `.env` is gitignored and no key belongs in a tracked file.

---

## Phase: Write Gate (agent/approval.py, tools/ticket_tools.py, tools/ticket_store.py, orchestrator terminal step)

**H1. The constraint matcher uses subject-token + unit matching, scoped per
semicolon-separated clause.** A fix number is checked against a bound only if
the units match AND at least one of that clause's subject tokens (words drawn
from the clause's own wording, minus comparator phrasing and a stopword list)
appears in the fix text. Rejected: flat unit-only matching, the first
implementation, which produced concrete false violations — RB-DEPLOY-001's
`timeoutSeconds <= 5s` and `initialDelaySeconds >= 15s` both normalize to unit
`"seconds"`, so a correct 3-second timeout fix tripped the unrelated
initialDelaySeconds bound; RB-DB-001 and RB-MEMORY-001 collide the same way on
`"%"`. Also rejected: full NL reasoning over the Constraints section (an LLM
call per proposed fix) — out of scope for this phase, since a human approves
every action regardless, so this function only needs to be a cheap tripwire,
not the safeguard itself.

**H2. `update_ticket` is ticket-scoped: it takes no `ticket_id` from the model.**
`UPDATE_TICKET_SCHEMA` omits the parameter and `TICKET_SCOPED_TOOLS` includes
`update_ticket`, so the executor injects `ticket_id` from `TaskState` and
overwrites anything the model supplies — identical treatment to `query_logs`/
`query_metrics`. Rejected: letting the model supply `ticket_id`, which would let
a prompt-injected instruction ("resolve ticket T999 instead") queue a write
against the wrong incident.

**H3. The mutation function (`apply_write`) lives in a module the model-facing
tool never imports, and the runtime empty-store assertion is the real
enforcement — not the static check.** `tools/ticket_tools.py` does not import
`tools/ticket_store.py` at all, so there is no code path from the tool to a
mutation to accidentally wire up. Rejected: keeping both in one module behind an
internal "don't call this from the tool" comment convention, which is exactly
the kind of rule a future edit silently violates. An AST-based test does check
that `ticket_tools.py` contains no import of `ticket_store`, but it is
deliberately documented as a WEAK secondary tripwire — it cannot catch
`importlib.import_module` called with a dynamically built name. The honest
primary guard is a separate test asserting `RESOLVED_TICKETS` is still empty
immediately after `update_ticket` returns; that one cannot be fooled by import
mechanics because it observes the actual mutation surface.

**H4. `POST /approvals/{action_id}/approve` and `/reject` take no request
body.** Rejected: letting a human edit the fix text at approval time, which is
a materially different feature (approve-with-edits) than approve/reject of what
the agent proposed, and was not asked for in this phase's spec.

**H5. The loop composes the write with a dedicated extra LLM call in
`_queue_write_action`, rather than widening the critic's `Assessment` contract
with a `proposed_fix` field.** The `Assessment` model / `CRITIC_INSTRUCTION`
pairing took real live debugging to stabilize (F1-F3: the hypothesis-deadlock,
the `content=None` inference fix, the confidence-constraint rules), and none of
that stability was worth reopening for a field this phase doesn't need on every
critic round — the fix only needs to be composed once, at the very end of a
resolved run.

**H6. Twice-failed verification demotes the run from `resolved` to
`escalated`, with `MAX_WRITE_ATTEMPTS = 2`.** Rejected: resolving anyway without
queueing a write, which would silently drop docs/design.md's requirement that a
`verification_failed` result forces a replan rather than being discarded — the
run would report "resolved" with nothing actually queued for a human to act on.
Also rejected: unlimited retries, which has no natural stopping point and could
loop the compose call indefinitely against a runbook constraint the model
cannot satisfy no matter how it rephrases the fix.

**H7. The loop never reads a ticket's `expected_behavior` field, anywhere.**
The literal reading of the phase request — "when `expected_behavior` is
`resolve_with_approval`, call `update_ticket`" — was rejected even though it is
a direct match for the spec's wording, because `expected_behavior` is gold eval
data. Reading it to decide whether to write would leak the answer into the run
itself, making the eval unable to tell a correct resolve from a leaked one. The
write decision instead comes purely from `_can_resolve`'s own belief-state
check (confidence, evidence sources, citations) — the same mechanism that
already decides resolve vs. escalate everywhere else in the loop. Both
specified test cases (a resolving ticket, an escalating one) produce the same
observable behavior either way, so nothing is lost by not reading the field.
