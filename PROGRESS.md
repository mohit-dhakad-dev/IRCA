# Progress Log

## Phase 0 — Setup
- [x] Folder structure, venv, git init
- [x] CLAUDE.md, design.md in place

## Phase 1 — MVP Scaffold
- [x] Minimal FastAPI app with POST /tickets endpoint
- [x] Pydantic TaskState model in agent/state.py
- [x] requirements.txt for the minimal runtime dependencies
- [x] .env.example with GROQ_API_KEY placeholder
- [x] Basic endpoint verification via local curl request

## Phase 2 — Data Layer
- [x] Created 6 synthetic runbooks in data/runbooks for db, memory, network, deploy, auth, and disk
- [x] Verified each runbook follows the required format and includes realistic symptoms, diagnosis, root cause, fix, and constraints
- [x] Created initial synthetic ticket batch in data/tickets.json with 15 records and required schema fields
- [x] Distributed ticket set across required categories: easy, multi_step, tool_heavy, ambiguous
- [x] Added validation test to confirm ticket category counts, required fields, and real runbook references
- [x] Verified join integrity between ticket gold_root_cause/gold_runbook_id and actual runbook files
- [ ] Expand to 50-80 tickets (later, once agent loop works — don't over-invest in data before the loop is proven)

## Phase 3 — Fake Tool Layer
- [x] ROOT_CAUSE_SIGNATURES data generator
- [x] query_logs, query_metrics with correct/wrong-metric differentiation
- [x] ambiguous tickets return weak signals
- [x] pytest coverage on 3+ tickets
- [ ] Metric summary polish: "still rising" contradicts a 100.0% cap on disk; T004 headlines
      the last point (15.0) over a post-collapse level nearer 5%, and window-wide avg straddles
      the step so it describes no moment that actually occurred

## Phase: Single-Pass Tool Calling (MVP)
- [x] LLM wrapper (Groq, tool-calling)
- [x] Tool schemas for query_logs/query_metrics
- [x] Single round-trip: tool call -> observation -> final answer
- [x] Manually verified on 3 tickets, tools actually get called
- [x] Model repinned to openai/gpt-oss-120b — llama-3.3-70b-versatile is retired on
      Groq and 404s (model_not_found); CLAUDE.md rule updated to match
- [x] ticket_id excluded from both tool schemas and injected by the executor from
      TaskState, so the model can never choose which incident is queried
- [ ] tool_choice="none" is unusable on gpt-oss-120b: the model still emits a tool
      call and Groq 400s (tool_use_failed); passing no tools behaves identically.
      A FINAL_INSTRUCTION user message forces the text answer instead — revisit if
      the model is ever repinned

## Phase: Agentic Loop (Part A — state, Part B — orchestrator)
- [x] TaskState extended for the loop: hypothesis, confidence (0-1), evidence_sources,
      iteration (replaced iteration_count), max_iterations=8, called_tool_signatures,
      trajectory retyped to list[dict], status Literal incl. "new"
- [x] Tool-invocation layer extracted to agent/tool_executor.py, shared by the
      single-pass baseline and the loop; baseline behavior verified unchanged
- [x] agent/orchestrator.py: run_agent_loop() — stop checks at top of each iteration,
      act -> tool round -> critic -> belief update -> replan
- [x] Loop guard on (tool_name, args-minus-ticket_id, sort_keys) — repeated call is
      skipped with a "choose differently" note and still consumes an iteration
- [x] Critic pass: separate JSON-only call, pydantic-validated, one re-ask; on second
      failure the belief is left unchanged and assessment_error recorded, never invented
- [x] Evidence source id is the bare tool name, so two calls to the same tool cannot
      self-confirm a hypothesis; resolve is a strict conjunction (>=0.75 AND >=2 sources)
- [x] Contradicting rounds (critic supports=false) are surfaced to the model rather than
      silently folded into a neutral note — docs/design.md replan trigger
- [x] Act-call retry backs off RETRY_BACKOFF_SECONDS; critic's malformed-JSON re-ask does not
- [x] POST /tickets/{ticket_id}/resolve wired to the loop, 404 on unknown ticket
- [x] 11 offline orchestrator tests (loop guard, iteration cap, 2-step resolve, no-new-info
      escalation, no self-confirm, LLM error, critic failure, unknown ticket, contradiction
      surfaced, unchanged-belief note, backoff) — 118 passed, 3 deselected, no network
- [x] Decisions + rejected alternatives recorded in docs/decisions.md (A1-A5, B1-B10)
- [ ] design.md's third escalate condition ("all tools tried, confidence still < 0.75") is
      deliberately deferred — with only 2 tools it would cap every run at 2 iterations and
      make the loop indistinguishable from the baseline. Add it with the 4th tool.
- [ ] Critic is handed full TOOL_SCHEMAS at tool_choice="auto" and could emit a tool call
      instead of JSON; degrades to assessment_error, but untested against a live model
- [ ] Not yet run against the real Groq API — all loop tests are stubbed

## Phase: RAG Layer (Part A — ingestion)
- [x] rag/ingest.py: markdown runbooks -> per-section chunks -> persisted Chroma
      collection "runbooks" at ./chroma_db (gitignored)
- [x] Chunking is by `## ` header, not fixed token windows, per docs/design.md
      Session 6 retrieval contract
- [x] `## Category` hoisted to `category` metadata rather than chunked — a one-word
      body is a near-content-free vector that would compete for top-3 slots against
      real Symptoms/Fix content (docs/decisions.md C1). Expected count is therefore
      6 docs x 5 sections = 30, not 36
- [x] Chunk document text is prefixed "{title} — {section}" so single-token
      Root Cause bodies and context-free Constraints bodies still embed usefully (C2)
- [x] Embedding function (all-MiniLM-L6-v2) attached to the collection, so Part B's
      search_runbooks embeds queries with the same model automatically (C3)
- [x] Cosine space set explicitly — the contract's "top score < 0.5 -> escalate"
      threshold is meaningless on chroma's default unbounded L2 (C4)
- [x] `python -m rag.ingest` is an idempotent clear+rebuild; verified two consecutive
      runs both report 30. delete_collection + upsert (NOT add: on chromadb 1.5.9
      add() silently no-ops on a duplicate id, so it was not the backstop the
      deterministic ids implied) (C5)
- [x] First-ever build catches chromadb.errors.NotFoundError, not ValueError — the
      latter crashed on a clean machine (C6)
- [x] Malformed runbooks fail loudly: missing H1, missing Category, missing chunk
      section, and duplicate section all raise ValueError rather than ingesting partial
- [x] 6 offline tests (30-chunk count + shape, real build + double-build idempotency
      into tmp_path, full 30-record metadata round-trip, 4 parametrized malformed cases)
      — 132 passed, 3 deselected
- [x] chromadb + sentence-transformers added to requirements.txt
- [x] Decisions + rejected alternatives recorded in docs/decisions.md (C1-C6)
- [x] Section splitter is now fenced-code-aware (done in Part B): a `## ` or `# ` line
      inside a ``` or ~~~ fence stays body content. Closing fences must be bare per
      CommonMark, so a ```python line inside a block does not close it.
- [ ] Part B not started: search_runbooks tool, top-3 + score, and the
      "top score < 0.5 -> no_confident_match" branch from the Session 6 contract
- [ ] Retrieval quality itself is unmeasured — the tests pin ingestion shape, not
      whether the right chunk comes back for a given ticket. Needs the eval harness.

## Phase: RAG Layer (Part B — retrieval + tool)
- [x] rag/retrieve.py: search_runbooks(query, top_k=3) -> {status, data, summary},
      chunks carry {doc_id, section, text, score} per the Session 6 contract
- [x] score = 1.0 - chroma cosine distance, so the contract's 0.5 is a real bounded
      similarity and not an arbitrary number on an unbounded metric (docs/decisions.md D1)
- [x] status="no_confident_match" below 0.5, and it still returns the rejected chunks +
      top_score so calibration and humans can see how close it got (D4)
- [x] No orchestrator change needed: the loop's no-new-info counter keys on any_new_ok
      (status == "ok"), NOT on status == "empty" as this file previously claimed — so
      no_confident_match already credits zero evidence and counts as no-new-info (D3)
- [x] Query-time embedding function passed explicitly to get_collection — the EF from
      create_collection is NOT restored on reopen. Correction to an earlier claim here:
      omitting it is currently harmless (chroma's default is also all-MiniLM-L6-v2), but
      passing a WRONG model is accepted silently with no guard — measured 0.18 and the
      wrong doc vs 0.69 correct. Naming the model on both sides pins them together (D6)
- [x] Client/collection cached module-level (re-instantiating the ST model costs seconds
      and the loop calls this repeatedly); cache repairs itself on NotFoundError by
      evicting and retrying once, so a rebuild between calls no longer raises (D7)
- [x] agent/tool_schemas.py: SEARCH_RUNBOOKS_SCHEMA, one `query` param, no ticket_id
- [x] agent/tool_executor.py: search_runbooks registered; ticket_id injection gated by
      TICKET_SCOPED_TOOLS, and non-scoped tools have model-supplied ticket_id STRIPPED
      rather than passed through, preserving the original injection defense (D5)
- [x] eval/calibrate_retrieval.py: runs all 15 tickets, reports rank-of-gold, score
      histogram, correct-vs-wrong score separation, and a threshold sweep
- [x] Calibration result: Recall@1 = Recall@3 = 14/15, zero tickets below 0.5, every
      threshold 0.30-0.55 admits the identical 15/14 split at no cost
- [x] SCORE_THRESHOLD left at 0.5 — NOT validated by the data (correct and wrong score
      ranges overlap; the one wrong retrieval scored 0.5739, inside the correct range
      of 0.5511-0.7992), but not contradicted either, and 15 tickets over 6 runbooks is
      too small to tune a cutoff on (D8)
- [x] 150 passed, 3 deselected; the stale-cache regression test verified to genuinely
      fail against the unfixed code
- [ ] T015 (ambiguous) retrieves RB-DB-001 over its gold RB-NETWORK-001 with gold at
      rank 4, at a top score inside the healthy range — a CONFIDENT WRONG retrieval,
      which no threshold can catch. This is the case Part C's >=2-independent-sources
      rule has to defend against; the agent must disconfirm rank 1, not trust it.
- [ ] Threshold margin is thin: min correct score 0.5511, and a merely-rephrased cache
      symptom measured 0.4933. Revisit with a larger ticket set.
- [ ] The loop's system prompt in agent/orchestrator.py still describes a
      query_logs -> query_metrics investigation order and does not mention
      search_runbooks — deliberately deferred to Part C, where the effect on real runs
      can be observed. The tool is registered and callable but the loop won't reach
      for it until the prompt names it.
- [ ] docs/design.md's third escalate condition ("all tools tried, confidence < 0.75")
      is still deferred — now that a 3rd tool exists, reconsider it in Part C.
- [ ] Retrieval tests use hand-picked symptom wording that scores comfortably above
      threshold. Ticket-text-driven assertions belong with the eval harness.

## Phase: Memory Layer (vectorstore.py, memory/, past_incidents)
- [x] vectorstore.py: shared Chroma plumbing extracted from rag/ BEFORE writing memory/ —
      cached client/collection, explicit embedding function on reopen, evict-and-retry-once
      query, cosine->similarity conversion, delete+upsert rebuild. The D6/D7 fixes now exist
      once instead of being copy-pasted into a second layer (E1)
- [x] Refactor verified behavior-preserving: T015 watched case still UNCHANGED at 0.5739 /
      rank 4, rag ingest still 30 chunks, all rag public names still importable (E1)
- [x] vectorstore.py imports from neither rag nor memory — dependency runs one way (E2)
- [x] data/past_incidents.json: 8 incidents over the 6 runbook root causes, two doubled
      (db_connection_pool_exhaustion, network_ingress_queue_exhaustion) because that is the
      pair retrieval already confuses on T015, so discrimination tests have real work (E4)
- [x] memory/ingest.py: validates schema loudly (missing/extra key, empty string, duplicate
      incident_id all raise); embeds symptom_summary ONLY, root cause + resolution as
      metadata, since the lookup axis is observed symptoms not fix wording (E3);
      `python -m memory.ingest` idempotent, 8 both runs
- [x] memory/store.py: search_past_incidents returns the contract's five keys including
      similarity_score (not `score`); summary text states plainly that a hit is a HINT
      requiring independent verification via logs/metrics before its root cause is adopted
- [x] search_past_incidents registered in tool_executor, NOT ticket-scoped (goes through the
      strip path); drift-invariant test updated deliberately to include it
- [x] Memory gate is 0.40, NOT rag's 0.5, and the constants are deliberately independent (E5)
- [x] Calibration: memory Recall@3 = 13/13, Recall@1 = 11/13. Correct top-1 scores span
      0.3244-0.6170 (median 0.5719) vs runbooks' 0.5511-0.7992 (median 0.7171) — memory
      scores on a structurally lower scale (short paraphrase vs long prose section)
- [x] Mirroring 0.5 onto memory rejected 4/13 tickets whose correct incident WAS retrieved
      (T001, T002, T003, T007). At 0.40 that drops to 2/13
- [x] eval/calibrate_retrieval.py covers both layers with --layer runbooks|memory|both, a
      side-by-side comparison, and no hardcoded threshold literals in its output
- [x] Second watched case T002 pins the OPPOSITE failure from T015: a CORRECT retrieval at
      0.3244 that no practical gate admits, vs a WRONG retrieval at 0.5739 that no gate
      rejects. Together they show the threshold cannot be tuned to fix both
- [x] tests/test_ticket_dataset.py join was unfalsifiable (parsed root causes then unioned in
      the same six as literals) — fixed, plus a dead no-op loop removed. Verified falsifiable
      by renaming a root cause in an isolated copy: all three join tests now fail (E7)
- [x] tests/test_incident_dataset.py: real join, coverage, and anti-rigging guards (no
      symptom_summary contains its own root-cause token or reuses a ticket_text verbatim)
- [x] 171 passed, 3 deselected; reviewed, one test-strength finding fixed
- [ ] 0.40 does NOT separate correct from wrong on memory: a wrong top-1 scored 0.5981,
      above the correct median 0.5719. It is a coverage/noise tradeoff. The real safeguard
      for this tool is the contract's independent-verification rule, not the gate (E5/E6)
- [ ] Both thresholds still rest on 15 tickets / 8 incidents. Re-run calibration when the
      ticket set expands to 50-80 (Session 2's deferred step)
- [x] Done in the Loop Integration phase below
- [ ] design.md's third escalate condition ("all tools tried, confidence < 0.75") is still
      deferred; with 4 tools now registered, reconsider it in Part C

## Phase: Loop Integration (four tools in the loop) — FIRST LIVE RUNS
- [x] SYSTEM_PROMPT rewritten for all four tools, with an EVIDENCE RULE (retrieval
      describes OTHER incidents; confirm against this incident's own logs/metrics),
      no_confident_match handling, and the untrusted-data warning extended to cover
      retrieved runbook chunks and past-incident records (a new injection surface)
- [x] No wiring needed: TOOL_SCHEMAS already carried all four and the orchestrator
      already passed it through. Only the prompt narrative was stale
- [x] FIRST live Groq runs of the loop. Found a hard deadlock that 172 stubbed tests
      could not catch: the critic was asked to assess a hypothesis, was shown none,
      correctly refused, and its refusal ("No hypothesis defined yet") was written back
      into state.hypothesis, poisoning every later round. Every run escalated at the cap
      regardless of evidence — T001: 8 successful tool calls, evidence_sources=[] (F1)
- [x] Fixed via CRITIC_INSTRUCTION inference rule, NOT the act-text pass-through: measured,
      gpt-oss-120b returns content=None whenever it emits tool calls (F2)
- [x] Inference constrained in the same change: empty/no-match observations are absence of
      evidence and must lower confidence; mechanism-free hypotheses get low confidence.
      Without this, T015 false-resolved at 0.78 on a hypothesis naming no mechanism (F3)
- [x] OBSERVATIONAL_TOOLS = {query_logs, query_metrics}; _can_resolve requires >=1 credited
      observational source, so retrieval alone can no longer satisfy the >=2-source bar (F4)
- [x] _can_resolve also requires a credited search_runbooks — live, the agent had been
      resolving in 2 iterations on logs+metrics alone and never consulting a runbook.
      Since only status=="ok" is credited, a no_confident_match cannot resolve (F5)
- [x] Live after the fixes: T001 resolved 0.95/3 iters (RB-DB-001 @ 0.70), T009 resolved
      0.95/3 iters (RB-DISK-001), T015 escalated 0.35/8 iters on no_confident_match
- [x] 185 passed, 3 deselected. Reviewed: ship, two documentation findings, both applied
- [ ] T015's escalation is NOT enforced by the design — it depended on the agent phrasing a
      query that scored 0.43. Calibration measures T015's own ticket text retrieving the
      WRONG runbook at 0.5739, above the gate. A closer-phrased query would have credited
      the wrong runbook and could have resolved incorrectly (F6)
- [x] Citations FIXED and enforced: Assessment + TaskState carry citations: list[str];
      _can_resolve requires >=1 citation and that EVERY citation is a doc_id actually
      returned this run by an "ok" search_runbooks observation, so a fabricated doc_id
      blocks the resolve. This is design.md's missing citation-verification pass (F8)
- [x] Caught that the feature would have silently done nothing: _format_observation
      truncates data at 500 chars and runbook chunk text ate the doc_ids before the critic
      saw them. Now renders a compact doc_ids=[...] ahead of the truncated blob (F9)
- [x] Verified at component level: digest carries doc_ids=["RB-DB-001"], critic returns
      citations=["RB-DB-001"], parse succeeds, fabrication guard covered offline
- [x] VERIFIED end-to-end live (F10 closed, see G1): T001 resolved 0.95 citations
      ['RB-DB-001']; T009 resolved 0.95 citations ['RB-DISK-001']; T015 escalated 0.60
      citations [] because search_runbooks returned no_confident_match, so nothing was
      credited and _can_resolve refused — enforcement, not persuasion
- [ ] One critic call during the rate-limit window returned an unparseable reply and
      finish_reason was not captured — unexplained one-off, watch for it (F10)
- [ ] BUDGET CONSTRAINT for the eval phase: ~20k tokens per ticket run means a 15-ticket
      sweep is ~300k, which does NOT fit in one day on the free tier. Plan for a paid tier,
      a multi-day split, or a reduced subset
- [ ] Everything in this phase was tuned on single live runs of 1-3 tickets. n=1 cannot
      separate a real behavioral change from model variance. Nothing here is measured until
      the eval harness runs all 15 tickets — that is the next phase and it now has real
      questions to answer, not just a scorecard to fill in
- [ ] Live runs cost 30-320s per ticket; a 15-ticket eval sweep will not be quick

## Phase: Provider portability
- [x] agent/llm.py moved to the openai SDK against a configurable LLM_BASE_URL, with
      LLM_API_KEY (falling back to GROQ_API_KEY) and LLM_MODEL. Groq defaults preserved,
      so an unchanged .env keeps working (G2)
- [x] The MODEL stays pinned to openai/gpt-oss-120b. Because it is open-weights, switching
      HOST cost a base_url; switching MODEL would have invalidated every live finding in
      F1-F9 (G2)
- [x] Acceptance-tested the new endpoint BEFORE trusting it: tool calls returned for all
      four schemas with correct args and no ticket_id, content is None on tool-calling
      turns, critic returns parseable JSON with citations. ~2k tokens; would have caught
      F1 and F2 immediately (G3)
- [x] CLAUDE.md provider rule updated; .env.example documents the three vars (placeholders
      only, no real keys); openai added to requirements.txt
- [x] 201 passed, 3 deselected
- [x] Latency worry was unfounded: new host ran the same tickets 3-5x faster end-to-end
      despite a worse TTFT benchmark. TTFT is noise for a batch eval (G4)
- [ ] ROTATE the current API key — it was pasted into a chat transcript (G5)
- [x] Cost measured, not estimated: resolving ticket $0.0017 (10.1k in / 1.3k out, 6 calls),
      escalating ticket $0.0060 (40k in / 4k out, 16 calls). A 15-ticket sweep is ~$0.06,
      so a $2 budget is ~35 sweeps. Escalating tickets cost 3.6x resolving ones because
      they burn the iteration cap on a growing transcript, so eval cost scales with the
      FAILURE rate rather than ticket count. Cached pricing equals input pricing, so prompt
      caching buys nothing here (G5)
- [ ] Budget is no longer the constraint — n=1 is. Spend it on REPEATED sweeps (5 runs
      = ~$0.30) to get variance on resolve/escalate rates, RAG-skip frequency, and
      false-resolve rate, rather than one careful run that still cannot distinguish a real
      behavioral change from model variance
- [ ] Eval harness should fan out over the 15 independent tickets concurrently — that beats
      any per-call latency tuning for wall-clock (G4)

## Phase: Write Gate (Parts A/B/C — approval-gated writes)
- [x] Part A: agent/approval.py — PendingAction model; verify_against_constraints parses the
      cited runbook's ## Constraints via rag.ingest.parse_runbook and checks numeric bounds
      against the proposed fix; in-memory pending-action store
- [x] Constraint matcher extracts bounds per SEMICOLON-separated clause with subject tokens
      drawn from that clause's own wording, and only checks a fix number against a bound when
      the UNIT matches AND a subject token from that clause appears in the fix text. Flat
      unit-only matching (the first implementation) produced false violations on real runbooks:
      RB-DEPLOY-001's timeoutSeconds<=5s and initialDelaySeconds>=15s both normalize to unit
      "seconds", so a correct 3-second timeout fix was rejected by the unrelated
      initialDelaySeconds bound; RB-DB-001 and RB-MEMORY-001 had the same collision on "%"
- [x] The first version's test for that exact case used a synthetic single-bullet runbook that
      dodged the collision; tests now run against the real data/runbooks files. 4 of the 6 new
      assertions were confirmed to fail against the pre-fix code
- [x] Part B: tools/ticket_tools.py update_ticket — verify_against_constraints runs BEFORE
      queueing; a failing verification returns status="verification_failed" with nothing
      queued, a passing one returns status="awaiting_approval" with a new PendingAction
- [x] tools/ticket_store.py — apply_write is the ONLY mutation path, and the status=="approved"
      guard lives INSIDE the function, not at the call site
- [x] Four /approvals endpoints (GET list, GET one, POST approve, POST reject) with 404 on an
      unknown action_id and 409 if the action is not currently "pending"
- [x] update_ticket registered in TOOL_REGISTRY and added to TICKET_SCOPED_TOOLS, so the
      executor overwrites any model-supplied ticket_id exactly as it does for query_logs/
      query_metrics; UPDATE_TICKET_SCHEMA exposes no ticket_id parameter
- [x] Approve-handler TOCTOU fix: status was set to "approved" BEFORE calling apply_write, so a
      raising apply_write left the action permanently stuck in "approved" (the endpoint only
      acts on "pending", apply_write only writes on "approved") with no path to retry. Fixed by
      reverting status to "pending" on failure and returning 500
- [x] The static invariant guard (an AST-based test that tools/ticket_tools.py contains no
      import of tools.ticket_store) is deliberately labelled a WEAK secondary tripwire — it
      cannot catch e.g. importlib called with a dynamically built module name. The real
      enforcement is the runtime assertion that RESOLVED_TICKETS is still empty immediately
      after update_ticket returns. This replaced an earlier source-substring scan that had
      distorted both modules' docstrings into circumlocutions just to avoid tripping itself
- [x] Part C: agent/orchestrator.py _queue_write_action is the loop's terminal action for a
      resolved run — one dedicated "compose the fix" LLM call on a COPY of the messages list,
      then update_ticket called DIRECTLY (not via execute_tool_call), because ticket_id comes
      from TaskState and routing through the model's tool-choice path would reopen the
      injection hole TICKET_SCOPED_TOOLS exists to close
- [x] One retry on verification_failed, feeding the rejection reason back into the next compose
      prompt (MAX_WRITE_ATTEMPTS=2); if both attempts fail verification the run is demoted from
      resolved to escalated rather than silently dropping the write
- [x] New TaskState.pending_action_id, set only on a successful queue
- [x] The loop does NOT read the ticket's expected_behavior field anywhere — that field is gold
      eval data, and reading it would leak the answer into the run. The write decision comes
      purely from _can_resolve's own belief-state check. An escalated run cannot reach
      _queue_write_action's call site structurally (it lives inside the single `if
      _can_resolve(state)` branch), not via a conditional guard that could be bypassed
- [x] The compose call retries once on a transient LLM error — it was the only LLM call site in
      orchestrator.py without one, and a single blip would have discarded a fully-evidenced
      resolved run. This retry is nested INSIDE one compose attempt so it does not consume any
      of the MAX_WRITE_ATTEMPTS verification budget
- [x] 11 pre-existing tests in tests/test_orchestrator.py each needed one extra scripted compose
      response, since a resolved run now makes one more LLM call than before; the scripted fix
      text was chosen to pass the real verifier so those tests still assert a resolve rather
      than silently becoming escalation tests
- [x] test_digest_carries_prior_observation_summary's count assertion was first merged 3->4 and
      then split into separate critic-call-count (3) and compose-call-count (1) assertions,
      because the merged count could no longer distinguish the write gate firing from an extra
      critic round — restoring a regression it existed to catch
- [x] Final suite: 231 passed, 3 deselected, fully offline
- [ ] NOT run against a live model. Everything in this phase is stubbed. Given F1/F2 (a deadlock
      that survived 172 stubbed tests and appeared on the first real run), the compose prompt in
      particular is unvalidated — needs a live run on a resolving ticket (T001) and an
      escalating one (T015)
- [ ] The constraint matcher can misattribute which bullet it cites: "Set maxmemory to 95%"
      against RB-MEMORY-001 correctly FAILS but quotes the used_memory_rss<75% bullet rather
      than the maxmemory<=80% one. Verdict right, quoted bullet suboptimal
- [ ] Latent: _queue_write_action retries a non-verification_failed update_ticket status (e.g.
      "error") using constraints-rejection wording that does not fit. Unreachable today since
      all four update_ticket args are guaranteed non-empty by that point
- [ ] Both stores (pending actions, RESOLVED_TICKETS) are in-memory and do not survive a
      restart. Fine for MVP, not for the eval harness if it ever runs across processes
- [ ] The approval flow has no UI — approve/reject is curl against the endpoints
- [ ] Constraint verification is a shallow regex heuristic, not reasoning: it can miss a real
      violation on unusual phrasing and can flag a safe fix. The human approver is the actual
      safeguard, not this function

## Phase: Dataset Expansion (Part A) + threshold recalibration
- [x] data/tickets.json 15 -> 63. T001-T015 byte-identical (verified: git diff shows zero
      removed lines, pure append). New: easy +10, multi_step +15, tool_heavy +7, rag_heavy +8,
      ambiguous +8 => easy 15 / multi_step 20 / tool_heavy 10 / rag_heavy 8 / ambiguous 10
- [x] New category value `rag_heavy` — symptom text deliberately worded to have LOW lexical
      overlap with its gold runbook, so retrieval must match on meaning rather than strings
- [x] Join integrity re-derived from the filesystem (parse each runbook's `## Root Cause`, build
      the real {root_cause: filename} map) and checked INDEPENDENTLY of the implementer's own
      script: 0 errors over all 63 — contiguous ids, no duplicate ticket_text, no crossed
      root_cause/runbook pairings, ambiguous all null + escalate. Falsifiability proven by
      crossing one pairing on an isolated copy and watching it fail
- [x] RUNBOOK THRESHOLD RECALIBRATED on 63 tickets (the step deferred since the memory phase).
      CORRECT top-1 n=56 spans 0.3385-0.7992; WRONG top-1 n=7 spans 0.2048-0.5739. Ranges still
      OVERLAP — the 15-ticket finding survived a 4x larger set
- [x] SCORE_THRESHOLD KEPT at 0.5, now on measured tradeoff rather than assumption: at 0.5,
      1 wrong admitted / 4 correct rejected / precision 0.981; raising one step to 0.55 LOWERS
      precision to 0.980 while rejecting 4 more correct. Lowering it to rescue rag_heavy would
      have traded T015-class false-resolves for coverage — rejected deliberately
- [x] 4 rag_heavy tickets (T051-T054) reworded from the fragile 0.42-0.49 band into 0.55-0.68,
      all rank-1. The fragile band is now EMPTY: every rag_heavy ticket is unambiguously one
      side of the gate. Rewrites kept the low-lexical-overlap constraint (forbidden-vocabulary
      list per ticket, verified empirically by re-measuring, not by inspection)
- [x] 3 rag_heavy tickets (T049 0.3734, T050 0.3385, T055 0.2048) kept BELOW the gate as
      intentional retrieval-failure cases, expected_behavior="escalate" with their real
      gold_root_cause retained — honestly labelled coverage, not a gap. All >=0.13 clear of 0.5
- [x] tests/test_ticket_dataset.py counts updated; non-ambiguous branch relaxed MINIMALLY so
      "escalate" is permitted only for category rag_heavy. Verified non-vacuous
- [x] T018 added to WATCHED_CASES beside T015 — a SECOND confirmed instance of the same
      DB/network confusion (both retrieve RB-DB-001, both with gold RB-NETWORK-001 at rank 4).
      The printer's rationale was hardcoded and true only of T015, so it moved to a per-case
      `why` field: T015 (0.5739) is inside the correct band so no threshold rejects it; T018
      (0.4994) IS rejected by the gate but gold sits outside top-3 so no threshold surfaces it
      either. T018 is an *easy* ticket — the confusion is not confined to hard cases
- [x] tests/test_rag_heavy_escalation.py: 3 tests pinning that T049/T050/T055 escalate
      specifically because _can_resolve requires a CREDITED search_runbooks — not merely that
      final status == "escalated". Includes the flip case (add search_runbooks -> resolves) so
      the requirement is proven load-bearing, and a check that no_confident_match is never
      credited. Verified test (b) is not vacuous: same input with status="ok" DOES credit
- [x] eval/calibrate_retrieval.py: removed `assert len(tickets) == 15`, which made the script
      unrunnable after any dataset growth; VERDICT block now derives every figure from the run
      (ticket/runbook/incident/root-cause counts, live-gate wrong-admitted and correct-rejected,
      precision, marginal cost of one step up)
- [x] Replaced the qualitative "precision knee / flat curve" label with the computed marginal
      tradeoff. The old precision_trend() compared only sweep endpoints, and memory's 0.65 row
      admits ONE match at precision 1.0 — so a flat curve read as "rises" and got labelled a
      knee. Degenerate rows now excluded via MIN_ADMITTED_FOR_TREND=5
- [x] Reviewer caught two stale-number findings, both mine, both the exact failure this phase
      existed to kill: docs/design.md quoted the 0.55 sweep row as if it were 0.5 (the rag_heavy
      rewrites had moved 4 tickets above the gate after I wrote it), and print_side_by_side
      still hardcoded "at 0.5, memory rejected 4 of 13 ... against 0/15 on runbooks" from the
      15-ticket era. Both fixed; the cross-mirror sentence is now computed and reads 17 of 43
      vs 4 of 56 — a stronger version of the same argument
- [x] MEMORY THRESHOLD RECALIBRATED on 53 gold-bearing tickets. CORRECT n=43 spans
      0.2270-0.6501; WRONG n=10 spans 0.2794-0.6323 — the wrong distribution now covers almost
      the whole correct one. 0.40 KEPT: precision is flat 0.82-0.86 across the entire usable
      range, so unlike runbooks the gate barely discriminates at all; 0.40 sits at the
      recall-favouring end, which is right for a layer whose output is contractually a HINT
- [x] docs/design.md updated: T015+T018 as a reproducible corpus property with the
      >=2-independent-sources justification, the recalibrated threshold paragraph, and a note
      that escalation for the rag_heavy cases depends on the runbook-credit requirement
      specifically — memory alone would credit T050 at 0.45
- [x] .claude/agents/implementer.md: hard rule added forbidding edits to TRACKED files for any
      demo/experiment (copy to a scratch path outside the repo first, confirm it exists, edit
      only the copy). Prompted by the implementer editing the live agent/orchestrator.py to
      prove a test falsifiable — it restored cleanly, confirmed via git diff against HEAD, but
      it was safe by luck. Second occurrence of this failure mode in the project
- [x] .claude/agents/implementer.md: corrected a stale line briefing the implementer on the
      RETIRED llama-3.3-70b-versatile / GROQ_API_KEY setup, contradicting CLAUDE.md
- [x] 236 passed, 3 deselected, fully offline. Reviewed; both findings fixed
- [ ] Both thresholds remain unvalidated as precise cutoffs — neither separates correct from
      wrong. The gate is a coverage/precision knob; the real safeguards are the
      >=2-independent-sources rule and the independent-verification requirement
- [ ] The `why` text in WATCHED_CASES still embeds score literals (0.5739, 0.4994, the
      0.3385-0.7992 range). Recorded expectations by design for the first two, but the range
      could drift silently — it is prose, not a compared field
- [ ] Memory-layer Recall@1 fell to 43/53 (0.811) on the larger set, from 11/13. Worth a look
      when past_incidents grows beyond 8 records
- [ ] Eval harness (Part A) NOT started — deliberately left for a separate session

## Phase: Eval Harness — Part A (runner)
- [x] eval/run_benchmark.py: sequential run_agent_loop over all 63 tickets, per-ticket raw
      output to eval/results/raw/{ticket_id}.json, written immediately after each ticket
      (never batched) and atomically (tmp + os.replace) so a kill mid-run cannot corrupt a
      completed ticket
- [x] `--live` is REQUIRED (mirrors pytest.ini `-m "not live"`); without it the script exits 2
      before touching a ticket. `--subset N` / `--tickets T001,T009` / `--out DIR` for cheap
      iteration
- [x] Token/LLM-call capture via a shim local to eval/ that patches
      agent.orchestrator.call_llm_with_tools (the orchestrator imports the symbol directly, so
      patching agent.llm would NOT take effect) and reads .usage off each ChatCompletion.
      Production code untouched. Chosen over instrumenting agent/llm.py so this phase stays in
      eval/ and is reversible — revisit if efficiency metrics need to survive outside the harness
- [x] state is a verbatim state.model_dump(); ticket fields are denormalized into each file so
      the scorer needn't re-join tickets.json. Part A computes NO metrics
- [ ] Constraint for Part B (eval/report.py): it MUST explicitly check run.runner_error on every
      file and surface crashed tickets as a distinct "crashed" count — never silently excluded
      from aggregates, never averaged away. A crashed ticket has state=null + populated
      runner_error, which is what distinguishes it from an absent file (not yet run) and from a
      legitimately escalated/errored run (state populated, runner_error null)
- [x] Sweep archiving (305e5d8): each sweep's raw output is MOVED to
      eval/results/runs/{UTC timestamp}/ before the next run, so raw/ always holds exactly one
      sweep. Two data-loss paths closed in review first — timestamp collision overwriting an
      existing archive, and a partial move falling through into the sweep over a half-cleared dir

## Phase 9a — Part A: In Progress
- [x] Runner (eval/run_benchmark.py) built, byte-identical checkpoint at 305e5d8
- [x] First full 63-ticket sweep complete (raw only, report.py not yet built)
- [x] Escalation accuracy: 12/13 correct
- [ ] **KNOWN ISSUE: 31/50 resolve-expected tickets escalated instead — under-resolution,
      root cause not yet isolated (confidence threshold / evidence-source count /
      observational-tool requirement — one of these three is binding)**
- [ ] Diagnostic breakdown of the 31 (next step)
- [ ] Part B (metrics.py + report.py) — deferred until diagnostic complete

### Correction to the KNOWN ISSUE above
The stop-success gate (_can_resolve, agent/orchestrator.py) has SIX conditions, not three, so
the candidate list above is incomplete. In addition to confidence >= 0.75, >= 2 evidence
sources, and >= 1 observational tool, a resolve also requires: "search_runbooks" itself
credited as an evidence source; >= 1 citation; and every citation being a doc_id actually
observed this run (a fabricated doc_id blocks the resolve exactly like no citation).

The runbook-credit condition is the prime suspect. _credit_evidence only credits a tool whose
observation status was "ok", and search_runbooks returns "no_confident_match" rather than "ok"
when it has no confident hit — so any ticket whose runbook retrieval misses the confidence
threshold can NEVER resolve, by design, regardless of how strong the log/metric evidence is.
That connects this issue directly to the unresolved retrieval-threshold calibration noted
above under the RAG/memory phase. The diagnostic must check all six conditions independently.

### First full sweep — raw results (2026-08-22)
- 63/63 tickets, 0 crashed, 0 corrupt files. Archived under eval/results/runs/
- Outcome: 20 resolved / 43 escalated
- Against expected_behavior: escalate 12/13 correct (1 wrongly resolved);
  resolve_with_approval 19/50 correct (31 wrongly escalated)
- Cost/efficiency: 1,243 LLM calls, 1.82M tokens in, 191k out (2.01M total), ~$0.39,
  50.4 min wall clock (mean 48.0s, p50 47.1s, p95 69.2s, max 94.6s per ticket)
- Estimate calibration: the projection from a single T001 run (~$0.25, ~20 min) was low by
  ~1.5x on cost and ~2.5x on time. T001 is a fast ticket, not a representative one — project
  future sweep cost from the mean, not from a sample of one
- The sweep was killed at T029 by a session teardown and resumed with
  --live --no-archive --tickets T030..T063. All 29 pre-kill files were valid and parseable,
  which is the atomic-write guarantee doing its job

### Under-resolution diagnosis (eval/diagnose_underresolution.py)
Of the 31 resolve-expected tickets that escalated:
- 15 passed ALL SIX _can_resolve conditions and were then demoted by the WRITE GATE
- 13 failed only has_citations; 2 failed confidence+citations; 1 failed confidence+evidence+observational
- runbook_credited and citations_grounded: ZERO failures each. The runbook-credit condition was
  the prime suspect and was wrong — worth recording, since the diagnostic is what falsified it
- Confidence is barely implicated: median 0.90 among failures, p25 0.85, well clear of the 0.75 bar.
  Do NOT tune CONFIDENCE_THRESHOLD on this evidence
Of the 15 write-gate demotions: 13 were "model returned no fix text", 2 were genuine constraint
violations (T002 95% vs a 75% bound, T035 15s vs a 5s bound) — the gate working correctly.

### Compose-step bug and fix
Root cause, established by live experiment rather than inspection: WRITE_COMPOSE_INSTRUCTION orders
the model to use the cited runbook's Fix/Constraints wording and units, but on the failing contexts
that text was never retrieved into the conversation. The model reaches for search_runbooks instead
of writing prose; because the compose call passes tools=[], agent/llm.py omits the tools parameter
entirely, so the provider returns finish_reason="tool_calls" with an EMPTY tool_calls list and empty
content. The fix text is destroyed in transit and nothing is recoverable.

This is deterministic per context, not flaky: T010 and T020 failed 9/9 on their captured contexts,
T011 succeeded 9/9. Ticket-level variation comes from which context a run happens to reach.

Two candidate fixes were tested and REJECTED — record them so they are not retried:
- tool_choice="none" with TOOL_SCHEMAS passed: provider ignores it for gpt-oss-120b, 10/10 still
  empty, indistinguishable from control
- flattening the tool-call history into prose turns: scored 0/8 "empty" but the content was GARBAGE
  containing raw harmony control tokens (<|message|><|start|>assistant). This would have been worse
  than the bug — non-empty debris passes the emptiness check and lands in the human approval queue
  as a proposed fix. The lesson: non-emptiness was the wrong success metric; sanity is the metric

Accepted fix: inject the cited runbook's Fix+Constraints text into the compose prompt, loaded via
the SAME rag.ingest.parse_runbook path that agent/approval.py verify_against_constraints uses, so
the model is shown exactly the text it will be judged against. Measured 18/18 sane vs 0/18 control
on the known-bad contexts. Plus _is_usable_fix_text, which rejects empty or control-token text and
triggers one in-attempt re-ask that does NOT consume a MAX_WRITE_ATTEMPTS slot.

Live validation: T020 and T010 flipped from deterministic false escalation to queued approvals.
T004 now escalates on a GENUINE constraint violation (15s vs a 5s bound) — correct behaviour, and
note the model was shown those constraints and still exceeded them, which is why the independent
verifier must stay independent.
- [ ] FOLLOW-UP (not this change): T011 now escalates on "Proposed fix value 500 (no unit)" — the
      constraint verifier rejecting a bare number for lacking a unit. A separate verifier-strictness
      question and a plausible remaining contributor to under-resolution. Look at it after Part B
- [ ] FOLLOW-UP: the sweep hit HTTP 429s repeatedly during probing. call_llm_with_tools retries a
      transient error exactly once with a 1s backoff, which is thin for a 63-ticket sequential run.
      Unrelated to the compose bug, but a second possible source of lost runs
- [ ] FOLLOW-UP: retry amplification. agent/llm.py now retries transient errors internally (up to
      6 attempts), but agent/orchestrator.py still has its own single retry-on-dict at the critic,
      loop, and compose sites, and _queue_write_action stacks a second one. Worst case for one
      compose step is 3 x 6 = 18 API attempts / ~450s under a sustained provider outage. Not a
      correctness risk — a stall, not a wrong answer — but the orchestrator-level retries are now
      largely redundant and should probably be dropped. Deliberately NOT changed immediately before
      the clean re-sweep, to avoid touching the agent loop right before the run it must validate
