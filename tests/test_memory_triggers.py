"""Tests for the stuck-state memory-consultation triggers added to
agent/orchestrator.py (_maybe_trigger_memory, _run_deterministic_memory_call,
_memory_consulted).

Fully offline: every test monkeypatches agent.orchestrator.call_llm_with_tools
and agent.tool_executor.TOOL_REGISTRY, following the conventions in
tests/test_orchestrator.py. No network call is ever made.
"""

from __future__ import annotations

import json

import pytest

import agent.orchestrator as orchestrator_module
import agent.tool_executor as tool_executor_module
from agent.orchestrator import MEMORY_NUDGE_TEXT, run_agent_loop

TICKET_ID = "T001"


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    monkeypatch.setattr(orchestrator_module, "RETRY_BACKOFF_SECONDS", 0)


# ---------------------------------------------------------------------------
# Stub helpers, duplicated from tests/test_orchestrator.py by convention.
# ---------------------------------------------------------------------------
class _StubFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _StubFunction(name, arguments)


class _StubMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StubChoice:
    def __init__(self, message):
        self.message = message


class _StubCompletion:
    def __init__(self, message):
        self.choices = [_StubChoice(message)]


def _make_tool_call_response(tool_calls):
    return _StubCompletion(_StubMessage(content=None, tool_calls=tool_calls))


def _make_text_response(content):
    return _StubCompletion(_StubMessage(content=content, tool_calls=None))


def _tool_call(call_id, name, arguments):
    return _StubToolCall(call_id, name, json.dumps(arguments))


def _assessment_response(hypothesis, confidence, supports, reasoning="because", citations=None):
    return _make_text_response(
        json.dumps(
            {
                "hypothesis": hypothesis,
                "confidence": confidence,
                "supports": supports,
                "reasoning": reasoning,
                "citations": citations if citations is not None else [],
            }
        )
    )


def _stub_tool(status="ok", summary="stub"):
    def fn(**kwargs):
        return {"status": status, "data": {}, "summary": summary}

    return fn


RUNBOOK_DOC_ID = "RB-DB-001"


def _stub_runbook_tool(doc_id=RUNBOOK_DOC_ID, status="ok", summary="stub"):
    def fn(**kwargs):
        return {
            "status": status,
            "data": {
                "chunks": [
                    {"doc_id": doc_id, "section": "diagnosis", "text": "t", "score": 0.9}
                ]
            },
            "summary": summary,
        }

    return fn


WRITE_GATE_FIX_TEXT = (
    "Increase the database pool size to a safe higher limit within the "
    "documented ceiling, keeping at least 25% headroom below the DB's hard "
    "limit, then monitor active connections for 45 minutes."
)


def _install_responses_capturing(monkeypatch, responses):
    """Like tests/test_orchestrator.py's _install_responses, but also records
    a shallow copy of the `messages` list argument on every call, so tests
    can inspect exactly what the model was shown at each round."""
    captured: list[list[dict]] = []

    def fake(messages, tools=None, **kwargs):
        captured.append(list(messages))
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake)
    return captured


def _has_text(messages: list[dict], needle: str) -> bool:
    return any(needle in (m.get("content") or "") for m in messages)


LOGS_ARGS = {"service": "checkout", "window": "2h", "level": "ERROR"}


def _setup_loop_guard_then_no_tool_call(monkeypatch):
    """Round1: query_logs (first time, ok). Round2: query_logs again with the
    identical signature -> loop guard skips it (Trigger A, first stuck
    occurrence this run). Round3: the model replies with no tool call while
    the evidence bar is still unmet (Trigger B, second stuck occurrence).
    """
    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok"))
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_past_incidents", _stub_tool(status="ok")
    )

    responses = [
        _make_tool_call_response([_tool_call("c1", "query_logs", LOGS_ARGS)]),
        _assessment_response("db issue", 0.3, True),
        _make_tool_call_response([_tool_call("c2", "query_logs", LOGS_ARGS)]),
        _assessment_response("db issue", 0.3, True),
        _make_text_response("still not sure"),
        _assessment_response("db issue", 0.3, False),
    ]
    captured = _install_responses_capturing(monkeypatch, responses)
    state = run_agent_loop(TICKET_ID)
    return state, captured


# ---------------------------------------------------------------------------
# 1. Loop guard fires once, memory unconsulted -> nudge shown, no
#    deterministic call yet.
# ---------------------------------------------------------------------------
def test_loop_guard_first_trigger_issues_nudge_only(monkeypatch):
    state, captured = _setup_loop_guard_then_no_tool_call(monkeypatch)

    # captured[4] is the act-call for round 3, i.e. everything the model was
    # shown AFTER round 2's loop guard fired but BEFORE round 3's own
    # (second) trigger could have run.
    round3_messages = captured[4]
    assert _has_text(round3_messages, MEMORY_NUDGE_TEXT)
    assert not _has_text(round3_messages, "automatically checked memory")
    assert state.memory_nudge_issued is True


# ---------------------------------------------------------------------------
# 2. Loop guard/Trigger B fires a second time, memory still unconsulted ->
#    deterministic search_past_incidents call happens, initiated_by "loop".
# ---------------------------------------------------------------------------
def test_second_trigger_makes_deterministic_call(monkeypatch):
    state, captured = _setup_loop_guard_then_no_tool_call(monkeypatch)

    assert state.memory_autoconsulted is True
    loop_calls = [
        e
        for e in state.trajectory
        if e.get("tool_call") is not None
        and e["tool_call"]["name"] == "search_past_incidents"
        and e["tool_call"].get("initiated_by") == "loop"
    ]
    assert len(loop_calls) == 1
    assert loop_calls[0]["observation"]["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. Model calls memory itself -> neither trigger ever fires.
# ---------------------------------------------------------------------------
def test_model_initiated_memory_call_suppresses_both_triggers(monkeypatch):
    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok"))
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_past_incidents", _stub_tool(status="ok")
    )

    responses = [
        _make_tool_call_response(
            [
                _tool_call("c1", "query_logs", LOGS_ARGS),
                _tool_call("c1b", "search_past_incidents", {"query": "db pool"}),
            ]
        ),
        _assessment_response("db issue", 0.3, True),
        # Round 2 repeats query_logs -> loop guard fires (Trigger A), but
        # memory was already consulted in round 1.
        _make_tool_call_response([_tool_call("c2", "query_logs", LOGS_ARGS)]),
        _assessment_response("db issue", 0.3, True),
        # Round 3: same repeated call again, so no_new_info reaches its cap
        # and the run escalates without needing further scripted rounds.
        _make_tool_call_response([_tool_call("c3", "query_logs", LOGS_ARGS)]),
        _assessment_response("db issue", 0.3, True),
    ]
    captured = _install_responses_capturing(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.memory_nudge_issued is False
    assert state.memory_autoconsulted is False
    for messages in captured:
        assert not _has_text(messages, MEMORY_NUDGE_TEXT)
    loop_calls = [
        e
        for e in state.trajectory
        if e.get("tool_call") is not None
        and e["tool_call"]["name"] == "search_past_incidents"
        and e["tool_call"].get("initiated_by") == "loop"
    ]
    assert loop_calls == []


# ---------------------------------------------------------------------------
# 4. No-tool-call round while _can_resolve is False -> Trigger B behaves like
#    Trigger A (nudge first).
# ---------------------------------------------------------------------------
def test_no_tool_call_while_unresolved_issues_nudge(monkeypatch):
    responses = [
        _make_text_response("thinking out loud"),
        _assessment_response("unclear", 0.2, False),
        _make_text_response("still thinking"),
        _assessment_response("unclear", 0.2, False),
    ]
    captured = _install_responses_capturing(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    round2_messages = captured[2]
    assert _has_text(round2_messages, MEMORY_NUDGE_TEXT)
    assert state.memory_nudge_issued is True


# ---------------------------------------------------------------------------
# 5. No-tool-call round while _can_resolve is True -> no trigger (a normal
#    resolve, not a punt).
# ---------------------------------------------------------------------------
def test_no_tool_call_while_resolved_does_not_trigger(monkeypatch):
    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok"))
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        _make_tool_call_response([_tool_call("c1", "query_logs", LOGS_ARGS)]),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response(
            [_tool_call("c2", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.6, True, citations=[RUNBOOK_DOC_ID]
        ),
        # Final round: no tool call, but the assessment pushes confidence/
        # citations over the resolve bar in this SAME round.
        _make_text_response("I am confident now"),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
        _make_text_response(WRITE_GATE_FIX_TEXT),
    ]
    captured = _install_responses_capturing(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert state.memory_nudge_issued is False
    assert state.memory_autoconsulted is False
    for messages in captured:
        assert not _has_text(messages, MEMORY_NUDGE_TEXT)


# ---------------------------------------------------------------------------
# 6. Deterministic call returning no_confident_match -> run continues, no
#    crash, no status change caused by it.
# ---------------------------------------------------------------------------
def test_deterministic_call_no_confident_match_does_not_crash(monkeypatch):
    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok"))
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY,
        "search_past_incidents",
        _stub_tool(status="no_confident_match"),
    )

    responses = [
        _make_tool_call_response([_tool_call("c1", "query_logs", LOGS_ARGS)]),
        _assessment_response("db issue", 0.3, True),
        _make_tool_call_response([_tool_call("c2", "query_logs", LOGS_ARGS)]),
        _assessment_response("db issue", 0.3, True),
        _make_text_response("still not sure"),
        _assessment_response("db issue", 0.3, False),
    ]
    _install_responses_capturing(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    loop_calls = [
        e
        for e in state.trajectory
        if e.get("tool_call") is not None
        and e["tool_call"]["name"] == "search_past_incidents"
        and e["tool_call"].get("initiated_by") == "loop"
    ]
    assert len(loop_calls) == 1
    assert loop_calls[0]["observation"]["status"] == "no_confident_match"


# ---------------------------------------------------------------------------
# 7. A run whose only additional evidence is a loop-initiated memory hit
#    still does NOT resolve.
# ---------------------------------------------------------------------------
def test_loop_initiated_memory_hit_alone_does_not_resolve(monkeypatch):
    # Same setup as test 2/6 above: only query_logs (an observational tool,
    # but never a search_runbooks source or citation) plus a loop-initiated
    # "ok" memory hit are ever seen this run.
    state, _ = _setup_loop_guard_then_no_tool_call(monkeypatch)

    assert state.memory_autoconsulted is True
    assert "search_runbooks" not in state.evidence_sources
    assert state.status != "resolved"


# ---------------------------------------------------------------------------
# 8. Provenance: every trajectory tool_call entry has an initiated_by value.
# ---------------------------------------------------------------------------
def test_every_tool_call_entry_has_initiated_by(monkeypatch):
    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok"))
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    metrics_args = {
        "service": "checkout",
        "metric": "db_pool_active_connections",
        "window": "2h",
    }
    responses = [
        _make_tool_call_response([_tool_call("c1", "query_logs", LOGS_ARGS)]),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response([_tool_call("c2", "query_metrics", metrics_args)]),
        _assessment_response("db_connection_pool_exhaustion", 0.8, True),
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
        _make_text_response(WRITE_GATE_FIX_TEXT),
    ]
    _install_responses_capturing(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    tool_call_entries = [e for e in state.trajectory if e.get("tool_call") is not None]
    assert tool_call_entries  # sanity: the run actually produced some
    for entry in tool_call_entries:
        assert entry["tool_call"].get("initiated_by") in ("model", "loop")


# ---------------------------------------------------------------------------
# 10. IRCA_MEMORY_TRIGGERS disabled -> a loop-guard-fired round issues NO
#     nudge and leaves memory_nudge_issued False.
# ---------------------------------------------------------------------------
def test_triggers_disabled_via_env_var_issues_no_nudge(monkeypatch):
    monkeypatch.setenv("IRCA_MEMORY_TRIGGERS", "0")
    state, captured = _setup_loop_guard_then_no_tool_call(monkeypatch)

    assert state.memory_nudge_issued is False
    assert state.memory_autoconsulted is False
    for messages in captured:
        assert not _has_text(messages, MEMORY_NUDGE_TEXT)
    loop_calls = [
        e
        for e in state.trajectory
        if e.get("tool_call") is not None
        and e["tool_call"]["name"] == "search_past_incidents"
        and e["tool_call"].get("initiated_by") == "loop"
    ]
    assert loop_calls == []


# ---------------------------------------------------------------------------
# 11. IRCA_MEMORY_TRIGGERS enabled (the default) -> existing behaviour is
#     unchanged, exercised with the env var explicitly set to an "on" value.
# ---------------------------------------------------------------------------
def test_triggers_enabled_via_env_var_preserves_existing_behaviour(monkeypatch):
    monkeypatch.setenv("IRCA_MEMORY_TRIGGERS", "1")
    state, captured = _setup_loop_guard_then_no_tool_call(monkeypatch)

    round3_messages = captured[4]
    assert _has_text(round3_messages, MEMORY_NUDGE_TEXT)
    assert state.memory_nudge_issued is True
    assert state.memory_autoconsulted is True


# ---------------------------------------------------------------------------
# 9. Neither trigger fires more than once per run, even if stuck states keep
#    recurring.
# ---------------------------------------------------------------------------
def test_triggers_fire_at_most_once_each_per_run(monkeypatch):
    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok"))
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_past_incidents", _stub_tool(status="ok")
    )

    # Repeat the SAME query_logs call every round: round 1 succeeds, every
    # round after that gets skipped by the loop guard (Trigger A recurring
    # again and again).
    responses = []
    for i in range(1, 6):
        responses.append(
            _make_tool_call_response([_tool_call(f"c{i}", "query_logs", LOGS_ARGS)])
        )
        responses.append(_assessment_response("db issue", 0.3, True))
    captured = _install_responses_capturing(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    nudge_count = sum(1 for messages in captured if _has_text(messages, MEMORY_NUDGE_TEXT))
    # Every round after the nudge is issued keeps seeing it in history, but
    # it must only ever be APPENDED once.
    nudge_message_total = sum(
        1
        for messages in captured
        for m in messages
        if m.get("content") == MEMORY_NUDGE_TEXT
    )
    assert nudge_message_total <= 1
    assert nudge_count >= 1  # sanity: it did get shown at least once

    loop_calls = [
        e
        for e in state.trajectory
        if e.get("tool_call") is not None
        and e["tool_call"]["name"] == "search_past_incidents"
        and e["tool_call"].get("initiated_by") == "loop"
    ]
    assert len(loop_calls) <= 1
