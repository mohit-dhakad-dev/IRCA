"""Shared tool-invocation layer used by both the single-pass baseline
(agent/single_pass.py) and the full agent loop (agent/orchestrator.py).

``ticket_id`` is always supplied by the executor, from the run's own
``TaskState``/argument, never by the model — see the comment on
``execute_tool_call`` below.
"""

from __future__ import annotations

import json

from agent.tool_schemas import EXECUTOR_INJECTED_ARGS
from memory.store import search_past_incidents
from rag.retrieve import search_runbooks
from tools.log_tools import query_logs, query_metrics
from tools.ticket_tools import update_ticket

# update_ticket is intentionally still registered here (and in
# TICKET_SCOPED_TOOLS below) even though agent/tool_schemas.py no longer
# offers it to the model as a callable tool. The defenses in this module
# (ticket_id injection, argument scoping) must still apply if it is ever
# reached, and the unauthorized-write-block tests in
# tests/test_tool_executor.py exercise exactly that path. Do not remove it
# from this registry.
TOOL_REGISTRY: dict[str, callable] = {
    "query_logs": query_logs,
    "query_metrics": query_metrics,
    "search_runbooks": search_runbooks,
    "search_past_incidents": search_past_incidents,
    "update_ticket": update_ticket,
}

# Tools that take an executor-injected ticket_id, as opposed to search_runbooks
# and search_past_incidents (read-only knowledge-base lookups with no notion
# of a specific ticket). update_ticket is the only WRITE (write-adjacent, via
# the approval gate) tool in the registry, and is ticket-scoped for exactly
# that reason: the model must never be able to name which ticket its proposed
# resolution applies to.
TICKET_SCOPED_TOOLS = {"query_logs", "query_metrics", "update_ticket"}


def _tool_call_error(name: str, summary: str) -> dict:
    """Build the shared error-record shape returned by execute_tool_call."""
    return {
        "name": name,
        "arguments": {},
        "status": "error",
        "result": {
            "status": "error",
            "data": {},
            "summary": summary,
        },
    }


def execute_tool_call(tool_call, ticket_id: str) -> dict:
    """Execute one model-requested tool call with ticket_id injected by the executor."""
    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        return _tool_call_error(name, f"Could not parse arguments for {name}: {exc}")

    if not isinstance(args, dict):
        return _tool_call_error(
            name,
            f"Arguments for {name} must be a JSON object, got {type(args).__name__}.",
        )

    if name not in TOOL_REGISTRY:
        return {
            "name": name,
            "arguments": args,
            "status": "error",
            "result": {
                "status": "error",
                "data": {},
                "summary": f"Unknown tool '{name}'.",
            },
        }

    if name in TICKET_SCOPED_TOOLS:
        # The model never supplies ticket_id — it is absent from the schema in
        # agent/tool_schemas.py by design. Any ticket_id the model emits anyway
        # (e.g. because injected log text talked it into one) is silently
        # overwritten here, not merged; which incident we are querying is the
        # executor's decision, not the model's.
        args["ticket_id"] = ticket_id
    else:
        # Non-ticket-scoped tools (e.g. search_runbooks) take no ticket_id at
        # all. Strip any executor-injected arg the model supplied anyway
        # rather than passing it through: the model has no business naming a
        # ticket for a tool that isn't ticket-scoped, and letting a
        # model-supplied value through here would reopen exactly the
        # injection hole the ticket-scoped branch above closes.
        for injected_arg in EXECUTOR_INJECTED_ARGS:
            args.pop(injected_arg, None)

    try:
        result = TOOL_REGISTRY[name](**args)
    except TypeError as exc:
        result = {
            "status": "error",
            "data": {},
            "summary": f"Could not call {name} with the given arguments: {exc}",
        }

    return {
        "name": name,
        "arguments": args,
        "status": result.get("status", "error"),
        "result": result,
    }
