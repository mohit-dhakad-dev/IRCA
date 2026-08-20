"""Tests for the write gate integration into agent/orchestrator.py
(_queue_write_action and its single call site in run_agent_loop).

Fully offline and deterministic: every test monkeypatches
agent.orchestrator.call_llm_with_tools, so no network call is ever made.
"""

from __future__ import annotations

import json

import pytest

import agent.approval as approval_module
import agent.orchestrator as orchestrator_module
import agent.tool_executor as tool_executor_module
import tools.ticket_store as ticket_store_module
from agent.orchestrator import run_agent_loop
from agent.state import TaskState

TICKET_ID = "T001"
RUNBOOK_DOC_ID = "RB-DISK-001"


@pytest.fixture(autouse=True)
def _clear_stores_and_backoff(monkeypatch):
    approval_module.clear_store()
    ticket_store_module.clear_store()
    monkeypatch.setattr(orchestrator_module, "RETRY_BACKOFF_SECONDS", 0)
    yield
    approval_module.clear_store()
    ticket_store_module.clear_store()


# ---------------------------------------------------------------------------
# Stub helpers mimicking the groq ChatCompletion response shape (same
# convention as tests/test_orchestrator.py).
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


def _install_responses(monkeypatch, responses):
    monkeypatch.setattr(
        orchestrator_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )


def _install_resolving_setup(monkeypatch):
    """Install tool stubs + a 2-round response script that reaches
    "resolved" via query_logs + search_runbooks (RB-DISK-001)."""
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )
    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("disk_log_rotation_gap", 0.6, True),
        _make_tool_call_response(
            [_tool_call("c2", "search_runbooks", {"query": "disk full log backlog"})]
        ),
        _assessment_response(
            "disk_log_rotation_gap", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    return responses


PASSING_FIX = "Rotate logs immediately and cap the log file size at 50 MB."
FAILING_FIX = "Rotate logs immediately and cap the log file size at 500 MB."


# ---------------------------------------------------------------------------
# 1. A resolved run queues exactly one PendingAction; state.pending_action_id
#    matches it; the ticket store itself is never written to.
# ---------------------------------------------------------------------------
def test_resolved_run_queues_pending_action(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    # Compose call after resolve.
    responses.append(_make_text_response(PASSING_FIX))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    pending = approval_module.list_pending_actions()
    assert len(pending) == 1
    action = pending[0]
    assert action.ticket_id == state.ticket_id
    assert action.proposed_root_cause == state.hypothesis
    assert action.citation_doc_id == state.citations[0]
    assert state.pending_action_id == action.action_id

    # Core invariant: the loop queues, it never writes.
    assert ticket_store_module.RESOLVED_TICKETS == {}


# ---------------------------------------------------------------------------
# 2. Escalation at the iteration cap never attempts a write.
# ---------------------------------------------------------------------------
def test_iteration_cap_escalation_does_not_attempt_write(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )

    responses = []
    for i in range(1, 9):
        args = {"service": "checkout", "window": f"{i}h", "level": "ERROR"}
        responses.append(_make_tool_call_response([_tool_call(f"c{i}", "query_logs", args)]))
        responses.append(_assessment_response("weak hypothesis", 0.3, True))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert approval_module.list_pending_actions() == []
    assert not any(
        e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
        for e in state.trajectory
    )


# ---------------------------------------------------------------------------
# 3. Escalation via the no-new-info counter never attempts a write.
# ---------------------------------------------------------------------------
def test_no_new_info_escalation_does_not_attempt_write(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("unclear", 0.1, False),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _assessment_response("unclear", 0.1, False),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert approval_module.list_pending_actions() == []
    assert not any(
        e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
        for e in state.trajectory
    )


# ---------------------------------------------------------------------------
# 4. search_runbooks returning no_confident_match throughout escalates
#    without attempting a write.
# ---------------------------------------------------------------------------
def test_no_confident_match_escalates_without_write(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY,
        "search_runbooks",
        _stub_tool(status="no_confident_match"),
    )

    responses = []
    for i in range(1, 9):
        if i % 2 == 1:
            tc = _tool_call(f"c{i}", "query_logs", {"service": "checkout", "window": f"{i}h", "level": "ERROR"})
        else:
            tc = _tool_call(f"c{i}", "search_runbooks", {"query": f"q{i}"})
        responses.append(_make_tool_call_response([tc]))
        responses.append(_assessment_response("disk_log_rotation_gap", 0.9, True))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert approval_module.list_pending_actions() == []
    assert not any(
        e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
        for e in state.trajectory
    )


# ---------------------------------------------------------------------------
# 5. Verification fails on attempt 1 (real gate, real runbook), then a
#    revised fix passes on attempt 2.
# ---------------------------------------------------------------------------
def test_verification_fails_then_passes_on_retry(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response(FAILING_FIX))
    responses.append(_make_text_response(PASSING_FIX))

    captured = []

    def fake_call(messages, tools):
        captured.append([m["content"] for m in messages])
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    pending = approval_module.list_pending_actions()
    assert len(pending) == 1

    update_ticket_entries = [
        e for e in state.trajectory
        if e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
    ]
    assert len(update_ticket_entries) == 2
    assert update_ticket_entries[0]["observation"]["status"] == "verification_failed"
    assert update_ticket_entries[1]["observation"]["status"] == "awaiting_approval"

    # The second compose prompt must have carried the first rejection reason.
    rejection_reason = update_ticket_entries[0]["observation"]["data"]["reason"]
    second_compose_messages = captured[-1]
    assert any(rejection_reason in (c or "") for c in second_compose_messages)


# ---------------------------------------------------------------------------
# 6. Verification fails on both attempts -> no PendingAction, escalated.
# ---------------------------------------------------------------------------
def test_verification_fails_on_both_attempts_escalates(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response(FAILING_FIX))
    responses.append(_make_text_response(FAILING_FIX))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert approval_module.list_pending_actions() == []

    update_ticket_entries = [
        e for e in state.trajectory
        if e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
    ]
    assert len(update_ticket_entries) == 2
    last_reason = update_ticket_entries[-1]["observation"]["data"]["reason"]
    final_entry = state.trajectory[-1]
    assert last_reason in final_entry["observation"]["error"]


# ---------------------------------------------------------------------------
# 7. The compose LLM call returning the error-dict sentinel TWICE in a row
#    (the retry inside a single compose attempt also fails) escalates
#    cleanly, with no exception and no PendingAction.
# ---------------------------------------------------------------------------
def test_compose_llm_error_sentinel_escalates_cleanly(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append({"error": "boom"})
    responses.append({"error": "boom again"})
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert approval_module.list_pending_actions() == []


# ---------------------------------------------------------------------------
# 8. A single transient compose error, retried once, succeeding -> the run
#    still resolves and exactly one PendingAction is created. This retry is
#    nested inside a single compose attempt and must not consume any of the
#    MAX_WRITE_ATTEMPTS verification retries.
# ---------------------------------------------------------------------------
def test_compose_transient_error_retries_and_resolves(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append({"error": "transient blip"})
    responses.append(_make_text_response(PASSING_FIX))

    sleep_calls = []
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda s: sleep_calls.append(s))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    pending = approval_module.list_pending_actions()
    assert len(pending) == 1
    assert sleep_calls == [orchestrator_module.RETRY_BACKOFF_SECONDS]

    update_ticket_entries = [
        e for e in state.trajectory
        if e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
    ]
    # The transient error + retry happened inside ONE compose attempt, so
    # only a single update_ticket call was ever made -- the attempt budget
    # was untouched by the transient failure.
    assert len(update_ticket_entries) == 1
