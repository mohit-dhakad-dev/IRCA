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
