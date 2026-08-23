# IRCA Offline Benchmark Report

## ⚠️ SAFETY GATE: PARTIAL
- unauthorized_write_block_rate: 3/3 enforced tests passed (0 failed)
- 1 test skipped: injection placeholder (adversarial ticket set not built)
- injection_block_rate: NOT COMPUTED (adversarial ticket set not yet built -- see PROGRESS.md)
Do not report safety as fully passing until both halves are measured.

## Summary
This report scores 63 raw ticket result(s). task_success_status_only is 59/63 (93.7%); task_success_strict_lexical is 13/63 (20.6%). 4 ticket(s) fall into documented known-issue buckets, 0 known-issue entries no longer reproduce (stale), and 0 failure(s) are unexplained. 0 ticket(s) crashed. Safety gate status: PARTIAL.

## Provenance
- tickets scored: 63
- earliest run.started_at: 2026-08-23T20:19:51.318510Z
- latest run.started_at: 2026-08-23T21:19:14.076112Z
- rate card used: Baseten (openai/gpt-oss-120b, rates confirmed == Groq card by project owner on 2026-08-23)

## Task Success
- task_success_status_only: 59/63 (93.7%)
- task_success_strict_lexical: 13/63 (20.6%)
- hypothesis_semantic: PENDING (Session 9b)
- gap: 46 ticket(s) reached the status-only success bar (status==resolved or correctly escalated) but failed the strict-lexical check -- their hypothesis text did not literally contain every significant token of gold_root_cause (see hypothesis_matches_gold's documented false-negative classes). This produces a 73.0-point gap between the two measures (59/63 vs 13/63).

## Category Breakdown
| category | n | status_only | strict_lexical |
|---|---|---|---|
| easy | 15 | 13/15 (86.7%) | 1/15 (6.7%) |
| multi_step | 20 | 18/20 (90.0%) | 1/20 (5.0%) |
| tool_heavy | 10 | 10/10 (100.0%) | 0/10 (0.0%) |
| rag_heavy | 8 | 8/8 (100.0%) | 1/8 (12.5%) |
| ambiguous | 10 | 10/10 (100.0%) | 10/10 (100.0%) |

## Known Issues (documented)
- T019 [T019]: asks user instead of investigating
- T025 [T025,T029]: loop-guard recovery
- T029 [T025,T029]: loop-guard recovery
- T039 [T039]: retrieval-score variance

## Stale Known Issues (no longer reproducing -- prune from eval/known_issues.py)
(none)

## Unexplained Failures
(none)

## Crashed Tickets
- count: 0

## Efficiency
(rate card: Baseten (openai/gpt-oss-120b, rates confirmed == Groq card by project owner on 2026-08-23))
- llm_call_count: mean=8.857 p50=7.000 p95=16.000 (n=63)
- tool_call_count: mean=3.857 p50=3.000 p95=8.000 (n=63)
- total_tokens_in: mean=18744.619 p50=13314.000 p95=40969.000 (n=63)
- total_tokens_out: mean=2531.175 p50=1988.000 p95=5391.000 (n=63)
- wall_clock_seconds: mean=35.027s p50=28.581s p95=64.602s (n=63)
- estimated_cost_usd: mean=0.004$ p50=0.003$ p95=0.009$ (n=63)

## Tool Use
- tool_selection_accuracy (mean): 96.5% (n=62)
- unnecessary_tool_calls (total): 11
- parameter_validity: observed_failures=0 / total_calls=243 -- measurable=False by design (see eval/metrics.parameter_validity docstring) -- this is a raw failure count, NEVER a percentage, and is NOT a validated 100% figure.

## State Tracking
- state_consistency: 63/63 (100.0%)
- write_gate_appended_correctly: 47/49 (95.9%) (N/A tickets (never reached the write gate) excluded from n/rate.)

## RAG
- retrieval_recall_at_3_observed: 49/49 (100.0%)
- citation_presence_rate: 49/49 (100.0%)
- retrieval_recall_at_3_corpus: NOT COMPUTED (run with --with-corpus-recall)

