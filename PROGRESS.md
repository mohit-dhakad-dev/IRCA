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
