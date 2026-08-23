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


# ---------------------------------------------------------------------------
# 9. _runbook_fix_and_constraints: loads both sections for a real doc_id,
#    returns None for a missing doc_id, and never raises either way.
# ---------------------------------------------------------------------------
def test_runbook_fix_and_constraints_real_doc_id():
    text = orchestrator_module._runbook_fix_and_constraints("RB-AUTH-001")
    assert text is not None
    assert "## Fix" in text
    assert "## Constraints" in text
    assert "signing key" in text.lower()


def test_runbook_fix_and_constraints_missing_doc_id():
    text = orchestrator_module._runbook_fix_and_constraints("RB-NOPE-999")
    assert text is None


# ---------------------------------------------------------------------------
# 10. The compose prompt actually contains the cited runbook's Fix and
#     Constraints text plus the delimiters.
# ---------------------------------------------------------------------------
def test_compose_prompt_contains_runbook_fix_and_constraints(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response(PASSING_FIX))

    captured = []

    def fake_call(messages, tools):
        captured.append([m["content"] for m in messages])
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    compose_messages = captured[-1]
    joined = "\n".join(c or "" for c in compose_messages)
    assert f"--- BEGIN RUNBOOK {RUNBOOK_DOC_ID} ---" in joined
    assert "--- END RUNBOOK ---" in joined
    assert "## Fix" in joined
    assert "## Constraints" in joined


# ---------------------------------------------------------------------------
# 11. When the loader returns None (unknown doc_id), the compose prompt is
#     unchanged (no injected block) and the existing escalate-on-failure
#     path still holds.
# ---------------------------------------------------------------------------
def test_missing_runbook_leaves_compose_prompt_unchanged_and_still_escalates(monkeypatch):
    state = TaskState(
        ticket_id=TICKET_ID,
        description="desc",
        hypothesis="some cause",
        citations=["RB-NOPE-999"],
    )

    captured = []

    def fake_call(messages, tools):
        captured.append([m["content"] for m in messages])
        return {"error": "boom"}

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    orchestrator_module._queue_write_action(state, [])

    assert state.status == "escalated"
    compose_messages = captured[-1]
    joined = "\n".join(c or "" for c in compose_messages)
    assert "BEGIN RUNBOOK" not in joined


# ---------------------------------------------------------------------------
# 12. An unusable (empty) response triggers ONE in-attempt re-ask; a usable
#     response on that re-ask queues a write, and MAX_WRITE_ATTEMPTS is NOT
#     exhausted (assert on the LLM call count / trajectory shape).
# ---------------------------------------------------------------------------
def test_unusable_response_reasks_once_then_resolves(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response(""))
    responses.append(_make_text_response(PASSING_FIX))

    call_count = {"n": 0}

    def fake_call(messages, tools):
        call_count["n"] += 1
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
    # Only ONE update_ticket call: the in-attempt re-ask did not consume a
    # MAX_WRITE_ATTEMPTS slot.
    assert len(update_ticket_entries) == 1
    # 2 pre-compose calls (query_logs round + search_runbooks round) + 2
    # compose calls (initial unusable + re-ask).
    assert call_count["n"] == 6


# ---------------------------------------------------------------------------
# 13. A response containing harmony control-token debris is rejected as
#     unusable and NEVER reaches update_ticket. Critical safety test.
# ---------------------------------------------------------------------------
def test_control_token_debris_never_reaches_update_ticket(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response("<|message|><|start|>assistant garbage"))
    responses.append(_make_text_response(PASSING_FIX))
    _install_responses(monkeypatch, responses)

    update_ticket_calls = []
    real_update_ticket = orchestrator_module.update_ticket

    def spy_update_ticket(**kwargs):
        update_ticket_calls.append(kwargs)
        return real_update_ticket(**kwargs)

    monkeypatch.setattr(orchestrator_module, "update_ticket", spy_update_ticket)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert len(update_ticket_calls) == 1
    assert "<|" not in update_ticket_calls[0]["proposed_fix"]
    assert update_ticket_calls[0]["proposed_fix"] == PASSING_FIX


# ---------------------------------------------------------------------------
# 13b. Control-token debris must never leak into state.trajectory either --
#      it is returned verbatim to API callers via /tickets/{ticket_id}/resolve
#      (main.py). Assert over the WHOLE serialized trajectory, not just one
#      field, so this can't pass by checking the wrong key.
# ---------------------------------------------------------------------------
def test_control_token_debris_never_reaches_trajectory(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response("<|message|><|start|>assistant garbage"))
    responses.append(_make_text_response(PASSING_FIX))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    serialized = json.dumps(state.trajectory)
    assert "<|" not in serialized


# ---------------------------------------------------------------------------
# 14. Two consecutive unusable responses (initial + re-ask) consume ONE
#     attempt; four total unusable responses consume both attempts and
#     escalate.
# ---------------------------------------------------------------------------
def test_repeated_unusable_responses_consume_attempts_and_escalate(monkeypatch):
    responses = _install_resolving_setup(monkeypatch)
    responses.append(_make_text_response(""))  # attempt 1, initial
    responses.append(_make_text_response(""))  # attempt 1, re-ask
    responses.append(_make_text_response("<|bad|>"))  # attempt 2, initial
    responses.append(_make_text_response("<|bad|>"))  # attempt 2, re-ask
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert approval_module.list_pending_actions() == []

    update_ticket_entries = [
        e for e in state.trajectory
        if e.get("tool_call") and e["tool_call"]["name"] == "update_ticket"
    ]
    # Each attempt is recorded exactly once in the trajectory (the in-attempt
    # re-ask doesn't add its own entry), so 2 attempts -> 2 entries.
    assert len(update_ticket_entries) == 2
    assert all(
        e["observation"]["summary"] == "model returned no usable fix text"
        for e in update_ticket_entries
    )
