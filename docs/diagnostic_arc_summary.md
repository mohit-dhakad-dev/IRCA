# Diagnostic arc: 49% → 94% on the 63-ticket benchmark

A narrative record of one debugging arc, written the day it happened against live trace
data. It is deliberately honest about the wrong turns, because the wrong turns are where
the method actually shows.

## Where it started

The first full sweep scored 31/63 (49%). Escalation behaviour was near-perfect — 12 of 13
tickets that should escalate did — but **31 of 50 tickets that should have resolved
escalated instead**. The agent was systematically refusing to finish work it had already
done correctly.

The obvious move was to tune `CONFIDENCE_THRESHOLD`. That would have been wrong, and the
data said so: median confidence among the failures was 0.90 against a 0.75 bar. The
constraint was somewhere else entirely.

## Wrong hypothesis, falsified by its own diagnostic

Reading `_can_resolve`, the six-condition resolve gate, the prime suspect was the
runbook-credit requirement: `search_runbooks` returns `no_confident_match` rather than
`ok` when retrieval is weak, and an uncredited runbook can never satisfy the gate. It was
a clean story and it was wrong. `eval/diagnose_underresolution.py` — which evaluates all
six conditions independently rather than short-circuiting, and cross-checks every verdict
against the real `_can_resolve` — reported **zero** failures on that condition. It also
cleared `citations_grounded` and largely exonerated confidence.

What it found instead partitioned the 31 exactly: 15 passed all six gates and were demoted
afterwards by the write gate, 13 failed only on having no citation, 2 were genuine
constraint violations, 1 had genuinely weak evidence. Two distinct bugs, not one.

The lesson worth keeping: the diagnostic was built to confirm a hypothesis and earned its
cost by destroying it.

## Bug one — output destroyed in transit

Thirteen runs reached the write gate and got back empty fix text. The recorded
`compose_text` was `''` — length zero — so this was not a parsing problem. Replaying real
compose contexts against the live API showed `finish_reason="tool_calls"` with an **empty**
`tool_calls` list: the model was emitting a tool call instead of prose, and because the
compose call passed `tools=[]` the provider had no schema to parse those tokens against, so
the output was discarded entirely. Nothing was recoverable.

It was deterministic per context, not flaky — T010 and T020 failed 9/9 on their captured
contexts while T011 succeeded 9/9. My initial "run-to-run variance" read was wrong, and it
mattered: a retry-based fix would have rescued none of them.

Two candidate fixes died on the way to the right one. `tool_choice="none"` was ignored by
the provider — 10/10 still empty, indistinguishable from control. Flattening the tool-call
history looked like a triumph at 0/8 empty, until I printed the actual text and found raw
harmony control tokens (`<|message|><|start|>assistant`). **That fix would have been worse
than the bug**: non-empty garbage passes an emptiness check and lands in a human approval
queue with a citation attached. Non-emptiness was the wrong success metric.

The real cause was in the prompt all along: `WRITE_COMPOSE_INSTRUCTION` ordered the model to
use the cited runbook's Fix and Constraints wording, on contexts where that text had never
been retrieved. Told to obey a document it could not see, the model reached for the tool.
Injecting the runbook — through the same loader the verifier uses, so the model sees exactly
what it will be judged against — gave 18/18 sane responses against 0/18 control.

## Bug two — citations made, then thrown away

The second bucket had sufficient evidence and no citation. The critic was not at fault:
replaying T009/T024/T038's own final digests returned the correct doc_id **9/9**. The loop
was destroying it. `state.citations = assessment.citations` ran after every round, and
`Assessment.citations` defaults to `[]`, so a critic reply that merely *omitted* the field
was indistinguishable from one retracting everything. A citation from round 2 survived only
if every later round restated it.

Fixing it by accumulating introduced a new risk that review caught before it shipped: a
citation from a hypothesis later abandoned could still ground the write. The resolution —
reaffirmed doc_ids move to the end, and the write grounds in the most recent — was chosen
over comparing hypothesis strings, because an LLM rephrasing the same hypothesis would have
falsely wiped citations and reintroduced the original bug.

## The systemic finding nobody was looking for

Correcting one ticket's gold label exposed something bigger. Three `rag_heavy` tickets were
built as "intentional retrieval-failure" cases whose ticket text was scrubbed of runbook
vocabulary. All three confidently retrieved their **own gold runbook** — because the *log
fixtures* still carried the vocabulary, and the agent queries with what it observes, not
with the ticket text.

The tempting fix was to scrub the fixtures at source. Measuring first killed that idea:
**53 of 53** non-ambiguous tickets carry gold-runbook vocabulary in their logs, and 0 of 10
`ambiguous` ones do. The leakage is not a flaw, it is the design every resolvable ticket
depends on — real logs contain the strings runbooks document. Scrubbing would have broken
the entire benchmark. The narrow error was three tickets assuming a retrieval failure
incompatible with a universal design.

## Bug three — and my own scorecard on it

The constraint verifier was rejecting fixes that complied with the runbook exactly. Round
one fixed unit-stripping and parameter association against hand-picked cases, and I verified
it against cases I had predicted. It cleared 3 of 6 targets — **and regressed three tickets
that had previously passed**, because a percentage bound now false-rejected. Hand-picked
verification could not have caught that, which is precisely why it was worthless here.

Round two started with `eval/verify_constraint_parsing.py`, dumping every one of the 18
Constraints bullets across all 6 runbooks with no assertions. The mechanism was visible
almost immediately: an identifier inside a leading conditional (``If `maxmemory` is set,``)
was being dropped, leaving generic tokens that overlapped a *different* bullet's bound.
Matching on parameter identity fixed it, and the corpus dump became a permanent regression
test rather than a one-off.

Honest tally of my own errors across the arc: one confidently wrong root-cause hypothesis;
one fix that looked successful while being actively harmful; two tickets reported as fixed
from a running log without checking the results; T004 characterised three different ways
before landing correctly; a parser fix that caused three regressions through inadequate
verification; every dollar figure computed with the wrong provider's rate card; and one
confounder flagged that did not exist. Each was caught — by a diagnostic, by review, or by
checking — before it reached a conclusion that mattered.

## Where it ended

| | before | after |
|---|---|---|
| task success | 31/63 (49%) | 59/63 (94%) |
| resolve-expected | 19/50 | 49/53 |
| escalate-expected | 12/13 | 10/10 |
| LLM calls | 1,243 | 558 |

Escalation accuracy never regressed through any round — the gains never came out of the
direction that matters most for a system that queues actions for human approval. All four
remaining failures map to documented follow-ups: loop-guard recovery (T025, T029), the agent
asking a human instead of investigating (T019), and retrieval-score variance (T039). Zero
unexplained failures remain.

## What generalises

Build the diagnostic before the fix, and let it falsify you. Choose the success metric
before you measure — "non-empty" nearly shipped a harmful fix that "sane" caught instantly.
Verify a fix against inputs that *should* fail, in isolation from the tree, because a test
that cannot fail proves nothing. Measure the corpus before fixing the fixtures — 53/53 was a
two-minute query that prevented breaking a benchmark. And when a fix looks like a win,
check what it made newly possible: accumulation solved a real bug and quietly opened a path
to citing an abandoned hypothesis.
