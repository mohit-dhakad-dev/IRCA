"""Pure scoring functions for one ticket's raw benchmark result.

Every function here is PURE: no file I/O, no network calls, no reading of
data/tickets.json. Callers (eval/report.py, Part B -- not written yet) are
responsible for loading the raw per-ticket JSON files written by
eval/run_benchmark.py and any side files (chunk stores, etc.), and for
aggregating these per-ticket scores across the corpus.

Inputs follow the raw schema documented in eval/run_benchmark.py's docstring:

    {
        "schema_version": ...,
        "ticket_id": ...,
        "run": {"started_at": ..., "wall_clock_seconds": ..., "runner_error": ...},
        "ticket": {
            "category": ..., "gold_root_cause": ..., "gold_runbook_id": ...,
            "required_tools": [...], "expected_behavior": ...,
            "min_confidence_evidence_sources": ...,
        },
        "state": <verbatim TaskState.model_dump(), or None if the run crashed>,
        "usage": {
            "llm_call_count": ..., "total_tokens_in": ..., "total_tokens_out": ...,
            "per_call": [...],
        },
    }

`state` is the verbatim output of `TaskState.model_dump()` -- a plain dict,
never a TaskState instance -- so every function here indexes it with `[...]`
/ `.get(...)`, never attribute access. `state` may be None (crashed ticket,
`run.runner_error` non-null); every function that receives `state` handles
that without raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from eval.calibrate_retrieval import gold_doc_id
from eval.known_issues import KnownIssue, find_known_issue

# A deliberately small stopword list: gold_root_cause slugs are short,
# hand-written underscore identifiers (e.g. "db_connection_pool_exhaustion"),
# not prose, so there is little need for a large list -- but articles /
# prepositions do occasionally show up in a slug for readability.
_STOPWORDS = {"a", "an", "the", "of", "in", "on", "to", "and", "or"}


def _token_variants(token: str) -> set[str]:
    """Light singular/plural + common-suffix variants of one gold token.

    Deliberately NOT a real stemmer/lemmatizer -- just enough to bridge
    noun/verb-tense spelling gaps that show up between a gold slug's token
    (e.g. "exhaustion") and how a model phrases its hypothesis prose (e.g.
    "the pool exhausted" or "connections are exhausting"). This is bounded,
    auditable string manipulation, not fuzzy matching.
    """
    variants = {token}

    if token.endswith("s") and len(token) > 1:
        variants.add(token[:-1])
    else:
        variants.add(token + "s")

    stem = None
    if token.endswith("ion") and len(token) > 4:
        stem = token[:-3]
    elif token.endswith("ed") and len(token) > 3:
        stem = token[:-2]
    elif token.endswith("ing") and len(token) > 4:
        stem = token[:-3]

    if stem:
        variants.update({stem, stem + "ed", stem + "ing", stem + "ion"})

    return variants


def hypothesis_matches_gold(hypothesis: str | None, gold_root_cause: str | None) -> dict:
    """Strict, auditable match between a hypothesis string and a gold root
    cause slug.

    Deliberately NOT fuzzy/loose token-overlap matching. This project's own
    constraint parser was broken for a long stretch (see PROGRESS.md's
    2026-08-23 entries and tests/test_approval.py) precisely because loose,
    unscoped substring/overlap matching cross-matched unrelated values. The
    rule here is intentionally narrow and easy to audit by hand: split the
    gold slug into significant tokens, and require EVERY one of them (or a
    light spelling variant -- see _token_variants) to literally appear,
    case-insensitively, in the hypothesis text. All-or-nothing; no partial
    credit, no scoring by fraction of tokens matched.

    This rule is deliberately CONSERVATIVE, not tuned for recall, and its
    false negatives are known and enumerated (validated against the real
    63-ticket sweep, where it rejected 46 of 59 status-resolved tickets):
      - synonym gaps the token-variant helper does not cover, e.g. "db" vs
        "database" (10 of the 27 single-token misses);
      - category-prefix tokens (e.g. "network", "deploy") that merely
        duplicate ticket.category rather than describing the failure mode,
        yet still count as "significant" under the current split rule;
      - nominalisation gaps beyond the ed/ing/ion variants this function
        covers, e.g. "misconfiguration" vs "misconfigured".
    Per the 9a/9b spec split, closing those gaps is NOT this function's job:
    no alias map, no loosening, no silent rescue. Whether a lexically-missed
    hypothesis is nonetheless semantically correct is deferred to the 9b
    LLM-judge layer -- see hypothesis_semantic_verdict below, which is a
    placeholder precisely because that judgement doesn't exist yet.
    """
    if not gold_root_cause:
        return {
            "matched": False,
            "gold_tokens": [],
            "missing_tokens": [],
            "rule": "n/a-no-gold",
        }

    raw_tokens = [t for t in gold_root_cause.lower().split("_") if t]
    gold_tokens = [t for t in raw_tokens if t not in _STOPWORDS]

    hyp_lower = (hypothesis or "").lower()

    missing = []
    for tok in gold_tokens:
        variants = _token_variants(tok)
        if not any(v in hyp_lower for v in variants):
            missing.append(tok)

    return {
        "matched": len(missing) == 0,
        "gold_tokens": gold_tokens,
        "missing_tokens": missing,
        "rule": "all-significant-tokens",
    }


def correct_resolution_status_only(state: dict | None, ticket: dict) -> bool:
    """status == "resolved", with no hypothesis-content check at all. This
    is the coarse measure: it answers "did the agent stop and declare a
    resolution" but says nothing about whether that resolution was right."""
    if not state:
        return False
    return state.get("status") == "resolved"


def correct_resolution_strict(state: dict | None, ticket: dict) -> bool:
    """status == "resolved" AND the strict lexical hypothesis_matches_gold
    check passes. See hypothesis_matches_gold's docstring for the known,
    enumerated false-negative classes this inherits -- this is the
    conservative measure, not the ground truth."""
    if not state:
        return False
    if state.get("status") != "resolved":
        return False
    match = hypothesis_matches_gold(state.get("hypothesis"), ticket.get("gold_root_cause"))
    return match["matched"]


def correct_escalation(state: dict | None, ticket: dict) -> bool:
    # Escalation has no hypothesis component to check -- there is no
    # "gold_root_cause" text being asserted -- so the status-only and
    # strict-lexical success measures always agree on escalation tickets;
    # this single function backs both task_success_status_only and
    # task_success_strict_lexical for expected_behavior == "escalate".
    if not state:
        return False
    return state.get("status") == "escalated" and ticket.get("expected_behavior") == "escalate"


def task_success_status_only(state: dict | None, ticket: dict) -> bool:
    """Dispatch on the ticket's OWN expected_behavior, using the coarse
    (status-only) resolution measure."""
    if not state:
        return False
    if ticket.get("expected_behavior") == "escalate":
        return correct_escalation(state, ticket)
    return correct_resolution_status_only(state, ticket)


def task_success_strict_lexical(state: dict | None, ticket: dict) -> bool:
    """Dispatch on the ticket's OWN expected_behavior, using the strict
    (lexical hypothesis-match) resolution measure."""
    if not state:
        return False
    if ticket.get("expected_behavior") == "escalate":
        return correct_escalation(state, ticket)
    return correct_resolution_strict(state, ticket)


def hypothesis_semantic_verdict(state: dict | None, ticket: dict) -> dict:
    """Placeholder slot for the Session 9b LLM-judge semantic verdict.

    Deliberately does NOT guess, approximate, or fall back to the lexical
    answer as if it were a semantic judgement -- "strict_lexical" here is
    reported for context only, alongside "pending", never AS the verdict.
    report.py / 9b should overwrite this once the judge layer exists.
    """
    hypothesis = state.get("hypothesis") if state else None
    match = hypothesis_matches_gold(hypothesis, ticket.get("gold_root_cause"))
    return {
        "verdict": "pending",
        "reason": "LLM-judge layer (Session 9b) not built",
        "strict_lexical": match["matched"],
        "missing_tokens": match["missing_tokens"],
    }


def terminal_write_entry_index(state: dict | None) -> int | None:
    """Index of the write-gate's terminal trajectory append.

    Identified by ITERATION INDEX (entry["iteration"] == state["iteration"])
    with tool_call.name == "update_ticket" -- NOT by tool name alone.
    update_ticket used to be model-callable and appears elsewhere in the
    corpus as an ordinary in-loop call; name-matching alone would
    misclassify those as the terminal write.
    """
    if not state:
        return None
    target_iteration = state.get("iteration")
    trajectory = state.get("trajectory") or []
    for i, entry in enumerate(trajectory):
        tool_call = entry.get("tool_call")
        if (
            entry.get("iteration") == target_iteration
            and tool_call is not None
            and tool_call.get("name") == "update_ticket"
        ):
            return i
    return None


def loop_tool_calls(state: dict | None) -> list[dict]:
    """Trajectory entries with a non-null tool_call, EXCLUDING the terminal
    write entry (see terminal_write_entry_index's docstring for why)."""
    if not state:
        return []
    trajectory = state.get("trajectory") or []
    terminal_idx = terminal_write_entry_index(state)
    out = []
    for i, entry in enumerate(trajectory):
        if entry.get("tool_call") is None:
            continue
        if i == terminal_idx:
            continue
        out.append(entry)
    return out


def tool_selection_accuracy(state: dict | None, ticket: dict) -> float | None:
    calls = loop_tool_calls(state)
    if not calls:
        return None
    required = set(ticket.get("required_tools") or [])
    hits = sum(1 for e in calls if e["tool_call"].get("name") in required)
    return hits / len(calls)


def unnecessary_tool_calls(state: dict | None, ticket: dict) -> int:
    calls = loop_tool_calls(state)
    required = set(ticket.get("required_tools") or [])
    return sum(1 for e in calls if e["tool_call"].get("name") not in required)


def parameter_validity(state: dict | None) -> dict:
    """Count of tool calls whose observation indicates an executor
    argument/schema-validation error, out of all loop tool calls.

    `measurable` is False BY DESIGN and this is never reported as a
    percentage: the raw schema records no schema-validation failures and no
    retry markers, so a would-be ratio here would evaluate to 1.0 by
    construction (every recorded call already "succeeded" in the sense of
    having produced a recorded observation) and would prove nothing about
    actual parameter validity. This function only counts observations whose
    status string looks like an executor argument error, IF any such status
    ever appears in a run; it does not claim to measure a rate.
    """
    calls = loop_tool_calls(state)
    failures = 0
    for entry in calls:
        obs = entry.get("observation") or {}
        status = str(obs.get("status", ""))
        if "argument" in status.lower() or "invalid_args" in status.lower():
            failures += 1
    return {
        "observed_failures": failures,
        "total_calls": len(calls),
        "measurable": False,
    }


def state_consistency(state: dict | None) -> bool:
    if not state:
        return False
    trajectory = state.get("trajectory") or []
    iteration = state.get("iteration")
    distinct = {e["iteration"] for e in trajectory if e["iteration"] < iteration}
    return len(distinct) == iteration


def write_gate_appended_correctly(state: dict | None) -> bool | None:
    """None when the run never reached the write gate (N/A)."""
    if not state:
        return None
    idx = terminal_write_entry_index(state)
    if idx is None:
        return None
    trajectory = state.get("trajectory") or []
    iteration = state.get("iteration")
    terminal_matches = [
        e
        for e in trajectory
        if e.get("iteration") == iteration
        and e.get("tool_call") is not None
        and e["tool_call"].get("name") == "update_ticket"
    ]
    return len(terminal_matches) == 1 and state.get("pending_action_id") is not None


def observed_retrieval_rank(state: dict | None, ticket: dict) -> int | None:
    """1-based rank of the ticket's gold_runbook_id among doc_ids returned
    by this run's search_runbooks observations (status "ok"), in returned
    order, deduped. None if no gold or no ok retrieval."""
    if not state or not ticket.get("gold_runbook_id"):
        return None
    gold = gold_doc_id(ticket)

    trajectory = state.get("trajectory") or []
    seen: list[str] = []
    for entry in trajectory:
        tool_call = entry.get("tool_call")
        if not tool_call or tool_call.get("name") != "search_runbooks":
            continue
        obs = entry.get("observation") or {}
        if obs.get("status") != "ok":
            continue
        chunks = (obs.get("data") or {}).get("chunks") or []
        for chunk in chunks:
            doc_id = chunk.get("doc_id")
            if doc_id and doc_id not in seen:
                seen.append(doc_id)

    if gold not in seen:
        return None
    return seen.index(gold) + 1


def retrieval_recall_at_3_observed(state: dict | None, ticket: dict) -> bool | None:
    rank = observed_retrieval_rank(state, ticket)
    if rank is None:
        return None
    return rank <= 3


def citation_presence(state: dict | None) -> bool | None:
    if not state or state.get("status") != "resolved":
        return None
    return bool(state.get("citations"))


@dataclass
class RateCard:
    """Per-provider $/Mtok rates, injected at call sites -- never hardcoded
    inline. Baseten currently bills at the same rates as the original Groq
    card, but every dollar figure in this project's log was once computed
    with the wrong provider's rate card (see PROGRESS.md's cost-figure
    correction entry), so the rate is always passed in explicitly rather
    than assumed."""

    input_per_mtok: float = 0.15
    output_per_mtok: float = 0.60


def estimated_cost_usd(usage: dict, rate: RateCard) -> float:
    tokens_in = usage.get("total_tokens_in") or 0
    tokens_out = usage.get("total_tokens_out") or 0
    return (tokens_in / 1_000_000) * rate.input_per_mtok + (
        tokens_out / 1_000_000
    ) * rate.output_per_mtok


def efficiency(raw: dict) -> dict:
    usage = raw.get("usage") or {}
    run = raw.get("run") or {}
    state = raw.get("state")
    tool_call_count = len(loop_tool_calls(state)) if state else 0
    return {
        "llm_call_count": usage.get("llm_call_count"),
        "tool_call_count": tool_call_count,
        "total_tokens_in": usage.get("total_tokens_in"),
        "total_tokens_out": usage.get("total_tokens_out"),
        "wall_clock_seconds": run.get("wall_clock_seconds"),
        "estimated_cost_usd": estimated_cost_usd(usage, RateCard()),
    }


def classify_outcome(raw: dict, ticket: dict, known_issue: KnownIssue | None = None) -> dict:
    """Classify one ticket's raw result into exactly one bucket.

    Bucketing uses task_success_STATUS_ONLY, not the strict lexical measure.
    Known issues (loop-guard recovery, asks-instead-of-investigates,
    retrieval variance) are about ESCALATION BEHAVIOUR, not hypothesis
    wording -- bucketing on strict lexical would reclassify the ~46 tickets
    that resolve with a hypothesis missing one gold token (see
    hypothesis_matches_gold's docstring for the enumerated false-negative
    classes) as "unexplained_failure", drowning the four real, documented
    known issues in noise and destroying the success/failure distinction
    PROGRESS.md draws from the real sweep. The strict-lexical result is
    still reported, just as a separate field, never as the bucketing input.

    Bucket order matters:
    1. "crashed" -- checked FIRST: run.runner_error non-null or state None.
    2. "success" -- task_success_status_only(state, ticket) is True.
    3. "known_issue" -- the ticket failed (status-only) AND a KnownIssue
       names it AND that KnownIssue's expect_status equals the ticket's
       actual status. A known issue applies ONLY on that exact status
       match -- if a known-issue ticket fails a DIFFERENT way than
       documented, it is "unexplained_failure", so the known bucket can
       never quietly absorb a new, undocumented bug.
    4. "stale_known_issue" -- the ticket is named in a KnownIssue but
       task_success_status_only is True: the documented failure no longer
       reproduces, so the config needs pruning rather than being allowed to
       grow unboundedly stale.
    5. "unexplained_failure" -- everything else that failed (status-only).

    Known issues are NEVER counted as successes, by construction (they are
    only reachable through the failure path).
    """
    run = raw.get("run") or {}
    state = raw.get("state")

    if run.get("runner_error") or state is None:
        return {
            "bucket": "crashed",
            "known_issue_id": None,
            "documented_cause": None,
            "task_success_status_only": False,
            "task_success_strict_lexical": False,
            "hypothesis_match": {"matched": False, "missing_tokens": []},
        }

    success_status_only = task_success_status_only(state, ticket)
    success_strict = task_success_strict_lexical(state, ticket)
    match = hypothesis_matches_gold(state.get("hypothesis"), ticket.get("gold_root_cause"))
    hypothesis_match = {"matched": match["matched"], "missing_tokens": match["missing_tokens"]}

    ticket_id = raw.get("ticket_id")
    issue = known_issue if known_issue is not None else find_known_issue(ticket_id)

    if success_status_only:
        if issue is not None:
            return {
                "bucket": "stale_known_issue",
                "known_issue_id": ",".join(issue.ticket_ids),
                "documented_cause": issue.cause,
                "task_success_status_only": True,
                "task_success_strict_lexical": success_strict,
                "hypothesis_match": hypothesis_match,
            }
        return {
            "bucket": "success",
            "known_issue_id": None,
            "documented_cause": None,
            "task_success_status_only": True,
            "task_success_strict_lexical": success_strict,
            "hypothesis_match": hypothesis_match,
        }

    if issue is not None and state.get("status") == issue.expect_status:
        return {
            "bucket": "known_issue",
            "known_issue_id": ",".join(issue.ticket_ids),
            "documented_cause": issue.cause,
            "task_success_status_only": False,
            "task_success_strict_lexical": success_strict,
            "hypothesis_match": hypothesis_match,
        }

    return {
        "bucket": "unexplained_failure",
        "known_issue_id": None,
        "documented_cause": None,
        "task_success_status_only": False,
        "task_success_strict_lexical": success_strict,
        "hypothesis_match": hypothesis_match,
    }
