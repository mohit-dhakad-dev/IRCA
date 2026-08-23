"""Tests for agent/single_pass.py (Phase 4 Part B — single-pass baseline).

Group A is fully offline and deterministic: it stubs out
agent.single_pass.call_llm_with_tools so no network call is ever made.

Group B (marked `live`) actually calls the Groq API and is deselected by
default via pytest.ini (`addopts = -m "not live"`).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agent.single_pass as single_pass_module
from agent.single_pass import FINAL_INSTRUCTION, TOOL_REGISTRY, run_single_pass
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Stub helpers mimicking the groq ChatCompletion response shape.
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

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (self.tool_calls or [])
            ],
        }


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


# ---------------------------------------------------------------------------
# GROUP A — offline, deterministic, no network.
# ---------------------------------------------------------------------------
def test_ticket_id_is_injected_into_executed_args(monkeypatch):
    captured = {}

    def spy_query_logs(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "data": {}, "summary": "ok"}

    monkeypatch.setitem(single_pass_module.TOOL_REGISTRY, "query_logs", spy_query_logs)

    tool_call = _StubToolCall(
        "call_1", "query_logs", json.dumps({"service": "checkout", "window": "2h", "level": "ERROR"})
    )
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("root cause: X, evidence: Y, confidence: high"),
    ]
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )

    result = run_single_pass("T001")

    assert captured.get("ticket_id") == "T001"
    assert result["tool_calls_made"][0]["arguments"]["ticket_id"] == "T001"


def test_injected_ticket_id_in_model_arguments_is_silently_overwritten(monkeypatch):
    """Injection-defense test: if injected log text talks the model into
    emitting a different ticket_id, the executor must overwrite it, not
    merge it — the real tool call must run against the true ticket_id.
    """
    captured = {}

    def spy_query_logs(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "data": {}, "summary": "ok"}

    monkeypatch.setitem(single_pass_module.TOOL_REGISTRY, "query_logs", spy_query_logs)

    tool_call = _StubToolCall(
        "call_1",
        "query_logs",
        json.dumps({"service": "checkout", "window": "2h", "level": "ERROR", "ticket_id": "T999"}),
    )
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )

    result = run_single_pass("T001")

    assert captured.get("ticket_id") == "T001"
    assert captured.get("ticket_id") != "T999"
    assert result["tool_calls_made"][0]["arguments"]["ticket_id"] == "T001"


def test_unknown_tool_name_returns_error_without_raising(monkeypatch):
    called = {"query_logs": False, "query_metrics": False}

    def guard_logs(**kwargs):
        called["query_logs"] = True
        return {"status": "ok", "data": {}, "summary": "ok"}

    def guard_metrics(**kwargs):
        called["query_metrics"] = True
        return {"status": "ok", "data": {}, "summary": "ok"}

    monkeypatch.setitem(single_pass_module.TOOL_REGISTRY, "query_logs", guard_logs)
    monkeypatch.setitem(single_pass_module.TOOL_REGISTRY, "query_metrics", guard_metrics)

    tool_call = _StubToolCall("call_1", "delete_everything", json.dumps({}))
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )

    result = run_single_pass("T001")

    assert result["tool_calls_made"][0]["status"] == "error"
    assert not called["query_logs"]
    assert not called["query_metrics"]


def test_malformed_arguments_json_returns_error_without_raising(monkeypatch):
    tool_call = _StubToolCall("call_1", "query_logs", "{not json")
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )

    result = run_single_pass("T001")

    assert result["tool_calls_made"][0]["status"] == "error"


def test_llm_error_branch(monkeypatch):
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: {"error": "boom"}
    )

    result = run_single_pass("T001")

    assert result["tool_calls_made"] == []
    assert "boom" in result["final_answer"]


def test_no_tool_calls_path(monkeypatch):
    monkeypatch.setattr(
        single_pass_module,
        "call_llm_with_tools",
        lambda *a, **kw: _make_text_response("root cause: X"),
    )

    result = run_single_pass("T001")

    assert result["tool_calls_made"] == []
    assert result["final_answer"] == "root cause: X"


def test_unknown_ticket_id_does_not_call_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(
        single_pass_module,
        "call_llm_with_tools",
        lambda *a, **kw: calls.append(1) or _make_text_response("unused"),
    )

    result = run_single_pass("NOPE")

    assert result["tool_calls_made"] == []
    assert "NOPE" in result["final_answer"]
    assert calls == []


def test_tool_calls_made_shape_has_no_result_key(monkeypatch):
    tool_call = _StubToolCall(
        "call_1", "query_logs", json.dumps({"service": "checkout", "window": "2h", "level": "ERROR"})
    )
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )

    result = run_single_pass("T001")

    for entry in result["tool_calls_made"]:
        assert set(entry.keys()) == {"name", "arguments", "status"}


@pytest.mark.parametrize("raw_arguments", ["[]", "3", "null", "\"x\""])
def test_non_dict_arguments_returns_error_without_raising(monkeypatch, raw_arguments):
    called = {"query_logs": False, "query_metrics": False}

    def guard_logs(**kwargs):
        called["query_logs"] = True
        return {"status": "ok", "data": {}, "summary": "ok"}

    def guard_metrics(**kwargs):
        called["query_metrics"] = True
        return {"status": "ok", "data": {}, "summary": "ok"}

    monkeypatch.setitem(single_pass_module.TOOL_REGISTRY, "query_logs", guard_logs)
    monkeypatch.setitem(single_pass_module.TOOL_REGISTRY, "query_metrics", guard_metrics)

    tool_call = _StubToolCall("call_1", "query_logs", raw_arguments)
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )

    result = run_single_pass("T001")

    assert len(result["tool_calls_made"]) == 1
    assert result["tool_calls_made"][0]["status"] == "error"
    assert not called["query_logs"]
    assert not called["query_metrics"]


def test_second_call_messages_omit_sdk_internal_fields(monkeypatch):
    tool_call = _StubToolCall(
        "call_1", "query_logs", json.dumps({"service": "checkout", "window": "2h", "level": "ERROR"})
    )
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    captured_messages = []

    def fake_call_llm_with_tools(messages, *a, **kw):
        response = responses.pop(0)
        if not responses:
            # This is the second invocation; capture the messages built for it.
            captured_messages.append(messages)
        return response

    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", fake_call_llm_with_tools
    )

    run_single_pass("T001")

    assert len(captured_messages) == 1
    messages = captured_messages[0]

    assistant_entries = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_entries) == 1
    assistant_entry = assistant_entries[0]

    assert set(assistant_entry.keys()) == {"role", "content", "tool_calls"}

    assistant_tool_call_ids = set()
    for tc in assistant_entry["tool_calls"]:
        assert set(tc.keys()) == {"id", "type", "function"}
        assert set(tc["function"].keys()) == {"name", "arguments"}
        assert tc["function"]["arguments"] == tool_call.function.arguments
        assert isinstance(tc["function"]["arguments"], str)
        assistant_tool_call_ids.add(tc["id"])

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    answered_ids = {m["tool_call_id"] for m in tool_messages}
    assert answered_ids == assistant_tool_call_ids
    # Exactly one tool message per assistant tool_call id.
    assert len(tool_messages) == len(assistant_entry["tool_calls"])


def test_final_call_appends_final_instruction(monkeypatch):
    """The appended FINAL_INSTRUCTION message is what keeps final_answer
    non-empty; tool_choice="none" is unusable on this provider/model (Groq
    rejects the request with a 400), so a regression back to it would
    silently produce "LLM follow-up call failed" answers.
    """
    tool_call = _StubToolCall(
        "call_1", "query_logs", json.dumps({"service": "checkout", "window": "2h", "level": "ERROR"})
    )
    responses = [
        _make_tool_call_response([tool_call]),
        _make_text_response("done"),
    ]
    captured_calls = []

    def fake_call_llm_with_tools(messages, *a, **kw):
        # Copy: `messages` is mutated in place by run_single_pass after this
        # call, so capturing the list itself would let later appends leak
        # into earlier captures.
        captured_calls.append((list(messages), kw))
        return responses.pop(0)

    monkeypatch.setattr(
        single_pass_module, "call_llm_with_tools", fake_call_llm_with_tools
    )

    run_single_pass("T001")

    assert len(captured_calls) == 2
    (first_messages, first_kwargs), (second_messages, second_kwargs) = captured_calls

    assert second_kwargs.get("tool_choice", "auto") in ("auto", None)

    assert second_messages[-1] == {"role": "user", "content": FINAL_INSTRUCTION}

    tool_message_indices = [
        i for i, m in enumerate(second_messages) if m.get("role") == "tool"
    ]
    final_instruction_index = len(second_messages) - 1
    assert tool_message_indices, "expected tool messages before the final instruction"
    assert final_instruction_index > max(tool_message_indices)

    assert not any(
        m.get("content") == FINAL_INSTRUCTION for m in first_messages
    )


def test_diagnose_endpoint_unknown_ticket_returns_404():
    response = client.post("/tickets/NOPE/diagnose")
    assert response.status_code == 404


def test_diagnose_endpoint_known_ticket_returns_200(monkeypatch):
    monkeypatch.setattr(
        single_pass_module,
        "call_llm_with_tools",
        lambda *a, **kw: _make_text_response("root cause: X"),
    )

    response = client.post("/tickets/T001/diagnose")

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"ticket_id", "tool_calls_made", "final_answer"}


# ---------------------------------------------------------------------------
# GROUP B — live, hits the real Groq API. Deselected by default.
# ---------------------------------------------------------------------------
@pytest.mark.live
@pytest.mark.parametrize("ticket_id", ["T001", "T006", "T012"])
def test_run_single_pass_uses_tools_live(ticket_id):
    """Hits the real Groq API — costs real API calls, requires GROQ_API_KEY.

    Deselected by default via pytest.ini; run explicitly with `pytest -m live`.
    """
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    result = run_single_pass(ticket_id)

    final_answer = result["final_answer"]
    assert not final_answer.startswith("LLM call failed:"), (
        "the initial LLM call failed (API/config problem), not a model "
        f"behavior finding: {final_answer!r}"
    )
    assert not final_answer.startswith("LLM follow-up call failed:"), (
        "the follow-up LLM call failed (API/config problem, e.g. Groq "
        f"rejecting the request), not a model behavior finding: {final_answer!r}"
    )
    assert len(final_answer.strip()) > 50, (
        "final_answer is empty or near-empty; the final call produced no "
        f"usable text: {final_answer!r}"
    )

    assert result["ticket_id"] == ticket_id
    assert result["tool_calls_made"], "expected the model to use tools rather than guess"
    for entry in result["tool_calls_made"]:
        assert entry["name"] in TOOL_REGISTRY
        assert entry["arguments"]["ticket_id"] == ticket_id
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"]
