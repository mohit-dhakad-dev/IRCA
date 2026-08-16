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

## Current phase
See PROGRESS.md for what's done and what's next.