# IRCA — Agentic Incident Resolution Copilot

## What this is
Read docs/design.md for full architecture. Summary: an agent diagnoses synthetic 
IT incident tickets by looping through plan→act(tool)→observe→replan→verify, 
using RAG over runbooks and a memory of past incidents.

## Rules
- Work in small, single-purpose increments. Never implement more than one phase 
  (per docs/design.md Step 15) in a single response.
- Always propose a short plan and file list BEFORE writing code, and wait for confirmation.
- Write a pytest test alongside every feature, in the same change.
- Only touch files relevant to the current task — do not refactor unrelated code.
- No real external APIs except the LLM provider. Logs/metrics/tickets are synthetic 
  (see tools/fake_data.py once created).
- Use Groq API (llama-3.3-70b-versatile) as the LLM provider, key in .env as GROQ_API_KEY.

## Delegation policy
This project uses a cost-optimized subagent topology (see .claude/agents/). The main
session is the orchestrator, not the typist.

- **Main session (this context)** — reserve its reasoning for planning, architecture
  decisions, phase sequencing, and final review of what comes back. Do not spend it
  on file-by-file edits or grepping.
- **implementer** (Sonnet 5) — all implementation and mechanical edits: new modules,
  endpoints, Pydantic models, tool functions, fixtures, pytest tests, renames,
  signature changes. Hand it a settled spec with the exact file list. It makes no
  architectural decisions — if the spec is ambiguous it returns a question instead of
  guessing, and that question comes back here to answer.
- **explorer** (Haiku 4.5, read-only) — all codebase search and file discovery: where
  a symbol lives, whether a helper already exists, what a section of docs/design.md
  says. Run it before writing a spec, not after.
- **reviewer** (Sonnet 5, no write access) — run on the diff before any work is
  considered done, and before updating PROGRESS.md or committing. Its findings come
  back here; fixes go back out to the implementer.

Workers run cheaper models on purpose. Only their summaries return to the main
context — not their file reads, greps, or intermediate reasoning — which is what keeps
this context free for the parts that actually need it. The existing rules above still
apply to delegated work: one phase per increment, a pytest test with every feature,
touch only relevant files.

## Current phase
See PROGRESS.md for what's done and what's next.