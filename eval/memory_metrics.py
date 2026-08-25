"""Pure metrics over the persisted output of eval/memory_warrant.py and the
per-ticket raw sweep results produced by eval/run_benchmark.py.

No chroma, no LLM, no network -- everything here reads already-persisted
JSON (the warrant report and a list of raw result dicts, one per ticket, in
the same shape eval/report.py's load_raw_results() returns). This keeps the
test suite and any report built on top of it offline and deterministic even
on a machine that has never built the chroma index.

Three metrics are reported TOGETHER, deliberately, and none of them stands
alone as "the" number to optimize:

  - memory_invocation_recall    -- of tickets that WARRANT a memory call
                                    (see eval/memory_warrant.py), how many did
                                    the agent actually call
                                    search_past_incidents on.
  - memory_invocation_precision -- of the calls the agent actually made, how
                                    many landed on a warranted ticket.
  - memory_hazard_exposure      -- how often the agent called memory on a
                                    HAZARD ticket, where a confident hit is a
                                    false lead the agent should not have
                                    reached for.

Recall alone would reward "always call memory on every ticket" as optimal
(recall 1.0), since recall only measures whether warranted calls happened,
never whether unwarranted ones did too. memory_hazard_exposure exists
specifically to contradict that reading: an agent that always calls memory
also maximizes its hazard exposure, and that number is reported alongside
recall precisely so "call memory more" is never read as a free win.

"Called" is determined by reusing eval.metrics.loop_tool_calls(state) --
never by walking the raw trajectory independently here -- so this module
stays consistent with the terminal-write-exclusion logic already defined
there (see loop_tool_calls' docstring for why the terminal update_ticket
entry is excluded).

Provenance split. The orchestrator (agent/orchestrator.py) can call
search_past_incidents itself when the agent is stuck -- see its
_maybe_trigger_memory fallback -- separately from the model choosing to call
it. Every trajectory tool_call entry the orchestrator writes carries an
"initiated_by" key, "model" or "loop". Every metric below reports the split
explicitly:

  - called_by_model / rate_model  -- AGENT-BEHAVIOUR figure. This is the
    number that answers "did the agent reach for memory when it should
    have" -- it is the only one of the three that reflects the model's own
    judgement.
  - called_by_loop / rate_loop    -- coverage delivered by the orchestrator's
    deterministic trigger mechanism, not agent judgement. Do not read this
    as the agent doing anything.
  - called_any / rate_any         -- combined figure (model OR loop),
    reported for completeness only. Never call this field "rate" bare --
    a bare "rate" invites silently conflating orchestrator intervention
    with agent behaviour.

An entry whose tool_call has a missing or unrecognised initiated_by (this
includes every result file produced before this provenance split existed)
is counted in an explicit "unknown_provenance" bucket and is NEVER assumed
to be model-initiated. This keeps pre-change result files honestly
unattributable rather than silently inflating called_by_model.
"""

from __future__ import annotations

from eval.metrics import loop_tool_calls

_KNOWN_PROVENANCE = ("model", "loop")


def _memory_calls_by_provenance(state: dict | None) -> dict[str, int]:
    """Count search_past_incidents calls in state's trajectory, bucketed by
    tool_call["initiated_by"]. Buckets: "model", "loop", "unknown". An entry
    with a missing or unrecognised initiated_by lands in "unknown", never
    "model"."""
    calls = loop_tool_calls(state)
    counts = {"model": 0, "loop": 0, "unknown": 0}
    for e in calls:
        tool_call = e["tool_call"]
        if tool_call.get("name") != "search_past_incidents":
            continue
        provenance = tool_call.get("initiated_by")
        if provenance not in _KNOWN_PROVENANCE:
            counts["unknown"] += 1
        else:
            counts[provenance] += 1
    return counts


def _called_memory(state: dict | None) -> bool:
    counts = _memory_calls_by_provenance(state)
    return any(counts.values())


def _called_memory_by(state: dict | None, provenance: str) -> bool:
    return _memory_calls_by_provenance(state)[provenance] > 0


def _index_results_by_ticket_id(results: list[dict]) -> dict[str, dict]:
    return {r.get("ticket_id"): r for r in results}


def _split_scored(warrant_rows: list[dict], results_by_id: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Split warrant rows into (scored rows paired with a usable state,
    not_scored ticket ids). A ticket is not_scored if it is missing from
    results entirely, or present but crashed (state is None)."""
    scored = []
    not_scored = []
    for row in warrant_rows:
        ticket_id = row["id"]
        raw = results_by_id.get(ticket_id)
        if raw is None or raw.get("state") is None:
            not_scored.append(ticket_id)
            continue
        scored.append({"row": row, "state": raw["state"]})
    return scored, not_scored


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def memory_invocation_recall(warrant: dict, results: list[dict]) -> dict:
    """Of tickets classified "warranted", how many did the agent actually
    call search_past_incidents on. Reported split by provenance: see the
    module docstring -- called_by_model / rate_model is the AGENT-BEHAVIOUR
    figure, called_by_loop / rate_loop is orchestrator-trigger coverage, and
    called_by_any / rate_any is the two combined."""
    results_by_id = _index_results_by_ticket_id(results)
    warranted_rows = [r for r in warrant["tickets"] if r["classification"] == "warranted"]
    scored, not_scored = _split_scored(warranted_rows, results_by_id)

    warranted_total = len(scored)
    called_by_model = sum(1 for s in scored if _called_memory_by(s["state"], "model"))
    called_by_loop = sum(1 for s in scored if _called_memory_by(s["state"], "loop"))
    called_by_any = sum(1 for s in scored if _called_memory(s["state"]))
    unknown_provenance = [
        s["row"]["id"]
        for s in scored
        if _called_memory_by(s["state"], "unknown")
        and not _called_memory_by(s["state"], "model")
        and not _called_memory_by(s["state"], "loop")
    ]
    missed_by_model = [s["row"]["id"] for s in scored if not _called_memory_by(s["state"], "model")]

    return {
        "warranted_total": warranted_total,
        "called_by_model": called_by_model,
        "called_by_loop": called_by_loop,
        "called_by_any": called_by_any,
        "unknown_provenance": unknown_provenance,
        "rate_model": _rate(called_by_model, warranted_total),
        "rate_loop": _rate(called_by_loop, warranted_total),
        "rate_any": _rate(called_by_any, warranted_total),
        "missed_by_model": missed_by_model,
        "not_scored": not_scored,
    }


def memory_invocation_precision(warrant: dict, results: list[dict]) -> dict:
    """Of the tickets the agent actually called search_past_incidents on,
    how many were classified "warranted". Reported split by provenance: see
    the module docstring -- the *_model figures are the AGENT-BEHAVIOUR
    ones, *_loop figures are orchestrator-trigger coverage, and *_any is the
    two combined."""
    results_by_id = _index_results_by_ticket_id(results)
    classification_by_id = {r["id"]: r["classification"] for r in warrant["tickets"]}

    scored, not_scored = _split_scored(warrant["tickets"], results_by_id)

    def _precision_for(provenance: str | None) -> tuple[int, int, list[str]]:
        called = [
            s
            for s in scored
            if (_called_memory(s["state"]) if provenance is None else _called_memory_by(s["state"], provenance))
        ]
        called_total = len(called)
        called_and_warranted = sum(
            1 for s in called if classification_by_id.get(s["row"]["id"]) == "warranted"
        )
        unwarranted_calls = [
            s["row"]["id"] for s in called if classification_by_id.get(s["row"]["id"]) != "warranted"
        ]
        return called_total, called_and_warranted, unwarranted_calls

    called_total_model, called_and_warranted_model, unwarranted_calls_model = _precision_for("model")
    called_total_loop, called_and_warranted_loop, unwarranted_calls_loop = _precision_for("loop")
    called_total_any, called_and_warranted_any, unwarranted_calls_any = _precision_for(None)
    unknown_provenance = [
        s["row"]["id"]
        for s in scored
        if _called_memory_by(s["state"], "unknown")
        and not _called_memory_by(s["state"], "model")
        and not _called_memory_by(s["state"], "loop")
    ]

    return {
        "called_total_model": called_total_model,
        "called_and_warranted_model": called_and_warranted_model,
        "rate_model": _rate(called_and_warranted_model, called_total_model),
        "unwarranted_calls_model": unwarranted_calls_model,
        "called_total_loop": called_total_loop,
        "called_and_warranted_loop": called_and_warranted_loop,
        "rate_loop": _rate(called_and_warranted_loop, called_total_loop),
        "unwarranted_calls_loop": unwarranted_calls_loop,
        "called_total_any": called_total_any,
        "called_and_warranted_any": called_and_warranted_any,
        "rate_any": _rate(called_and_warranted_any, called_total_any),
        "unwarranted_calls_any": unwarranted_calls_any,
        "unknown_provenance": unknown_provenance,
        "not_scored": not_scored,
    }


def memory_hazard_exposure(warrant: dict, results: list[dict]) -> dict:
    """How often the agent called memory on a "hazard" ticket -- a ticket
    whose gold_root_cause is null (should escalate) but which memory
    answers with a confident-looking (but false-lead) hit. Reported split
    by provenance: see the module docstring -- called_by_model is the
    AGENT-BEHAVIOUR figure (the agent itself reaching for a false lead),
    called_by_loop is the orchestrator's trigger firing into a hazard
    ticket, and called_by_any is the two combined."""
    results_by_id = _index_results_by_ticket_id(results)
    hazard_rows = [r for r in warrant["tickets"] if r["classification"] == "hazard"]
    scored, not_scored = _split_scored(hazard_rows, results_by_id)

    hazard_total = len(scored)
    tickets_model = [s["row"]["id"] for s in scored if _called_memory_by(s["state"], "model")]
    tickets_loop = [s["row"]["id"] for s in scored if _called_memory_by(s["state"], "loop")]
    tickets_any = [s["row"]["id"] for s in scored if _called_memory(s["state"])]
    unknown_provenance = [
        s["row"]["id"]
        for s in scored
        if _called_memory_by(s["state"], "unknown")
        and not _called_memory_by(s["state"], "model")
        and not _called_memory_by(s["state"], "loop")
    ]

    return {
        "hazard_total": hazard_total,
        "called_by_model": len(tickets_model),
        "called_by_loop": len(tickets_loop),
        "called_by_any": len(tickets_any),
        "tickets_model": tickets_model,
        "tickets_loop": tickets_loop,
        "tickets_any": tickets_any,
        "rate_model": _rate(len(tickets_model), hazard_total),
        "rate_loop": _rate(len(tickets_loop), hazard_total),
        "rate_any": _rate(len(tickets_any), hazard_total),
        "unknown_provenance": unknown_provenance,
        "not_scored": not_scored,
    }
