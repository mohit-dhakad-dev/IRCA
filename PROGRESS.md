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
