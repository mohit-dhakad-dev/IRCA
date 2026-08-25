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
- hypothesis_semantic (headline): 41/46 (89.1% over n_judged=46) -- rubric v3, repeats=3
  - failure decomposition: semantic_only=1 evidence_only=3 both=1
  - judge stability across repeats: unanimous=39 split=7
- gap: 46 ticket(s) reached the status-only success bar (status==resolved or correctly escalated) but failed the strict-lexical check -- their hypothesis text did not literally contain every significant token of gold_root_cause (see hypothesis_matches_gold's documented false-negative classes). This produces a 73.0-point gap between the two measures (59/63 vs 13/63). Of the 46, 41 were judged semantically correct under rubric v3.

## Judge Validation
The validation sample deliberately oversamples judge disagreement -- these figures describe calibration on hard cases and must NOT be used to correct the headline rate above.
- n_compared: 25 (n_skipped: {'human_unlabeled': 0, 'judge_failed': 0, 'not_in_both': 21})
- contingency (human x judge): both_true=18 both_false=3 human_true_judge_false=2 human_false_judge_true=2
- raw_agreement: 84.0%
- kappa: 0.500
- pabak: 0.680
- prevalence: 0.800
- kappa=0.500, PABAK=0.680, prevalence=0.800. PABAK equals 2*p_o - 1 and does not depend on the raters' marginal label distributions; kappa is chance-corrected against those marginals and can diverge from PABAK when prevalence is far from 0.5.
- source: eval/results/judge_agreement_report.json

### Rubric history
Rubric v1 was unspecified and produced kappa 0.066 (chance-level) because the human rater and judge were answering different questions (semantic match vs evidence grounding). Rubric v3 states a shared criterion and reaches kappa 0.500 / 84.0% raw agreement (vs v1's raw agreement 64.0%). This is the diagnostic that justified specifying the rubric, not a discarded failure. Preserved evidence: eval/results/hypothesis_semantic.v1.json, eval/results/judge_agreement_report.v1.json (v1 semantic_correct_rate was 84.8% -- do not cite this number as current).

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

## RAGAS Metrics — Diagnostic (NOT agent-quality claims)
All four RAGAS metrics measured in this project (context_precision, context_recall, faithfulness, answer_relevancy) proved structurally unfit for this dataset. None of the numbers below should be read as a measure of agent quality -- they are diagnostic evidence of *why* each metric failed.

### (a) context_precision / context_recall — degenerate constant
- context_precision: 1.0000 on 14/14 readings (n=3 subset3, n=3 subset3_mw16, n=8 subset8) -- constant across all readings
- context_recall: 1.0000 on 8/8 readings (subset8 only) -- constant
- subset3 and subset3_mw16 returned NaN for context_recall (metric-key bug -- they scored llm_context_recall, which came back None/NaN) so they are NOT evidence of degeneracy; only subset8's 8 readings are.
- working explanation: state.citations records only doc_id, not section, so `contexts` is always all 5 chunks of the cited runbook regardless of retrieval quality, plausibly saturating both metrics at a trivial ceiling. Not re-run at full scale because the degeneracy is already established.

### (b) faithfulness — confounded by answer mood
- overall faithfulness: 0.698 over 46 tickets
- proposed_fix answers are written either as imperative plans or past-tense reports; the 9 past-tense ones averaged 0.498 vs 0.747 for the 37 imperative ones, and both 0.000 scores (T020, T031) were past-tense.
- a controlled experiment rewrote those 9 into imperative mood, preserving every fact, number and threshold (verified before scoring), holding contexts identical: mean faithfulness rose from 0.498 to 0.850 (+0.353). Normalised, those 9 score above the 37 already-imperative tickets at 0.747 -- so they were never worse-grounded.

| ticket | before (gap46) | after (mood-normalized) |
|---|---|---|
| T010 | 0.857 | 1.000 |
| T018 | 0.400 | 0.800 |
| T020 | 0.000 | 0.833 |
| T023 | 0.600 | 0.800 |
| T030 | 0.889 | 1.000 |
| T031 | 0.000 | 0.818 |
| T034 | 0.400 | 0.778 |
| T036 | n/a | 1.000 |
| T043 | 0.333 | 0.625 |
| T045 | 1.000 | 1.000 |

- conclusion: a metric that moves 0.35 on a content-preserving paraphrase is not measuring grounding stably enough to report as a system-quality figure.

### (c) answer_relevancy — question/answer genre mismatch
- overall answer_relevancy: 0.472 over 46 tickets
- does not discriminate against ground truth: mean 0.473 on judge-correct tickets (n=41) vs 0.466 on judge-incorrect (n=5), a 0.007 difference.
- the mood experiment moved answer_relevancy only -0.019, so mood is not the cause.
- working explanation: the metric reverse-generates questions from the answer and compares them to the "question", but our question is ticket_text (a symptom narrative, not a question) and our answer is a remediation plan -- different genres of text, so similarity is low regardless of quality.

**Summary:** four of four metrics failed by three distinct mechanisms, which is itself a finding about applying a QA-shaped eval framework to incident remediation.

