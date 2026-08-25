"""Tests for agent/tool_executor.py.

Fully offline: search_runbooks is monkeypatched with a stub in the registry
so these tests never touch chroma or the embedding model. query_logs /
query_metrics are the real fake tools already exercised elsewhere.
"""

from __future__ import annotations

import json

import pytest

import agent.tool_executor as tool_executor_module
from agent.tool_executor import TICKET_SCOPED_TOOLS, TOOL_REGISTRY, execute_tool_call

TICKET_ID = "T001"


class _StubFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, name, arguments):
        self.function = _StubFunction(name, arguments)


def _call(name: str, arguments: dict) -> _StubToolCall:
    return _StubToolCall(name, json.dumps(arguments))


@pytest.fixture
def stub_search_runbooks(monkeypatch):
    received = {}

    def _stub(**kwargs):
        received.update(kwargs)
        return {"status": "ok", "data": {"query": kwargs.get("query"), "chunks": [], "top_score": 0.0}, "summary": "stub"}

    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub)
    return received


def test_search_runbooks_gets_ticket_id_injected(stub_search_runbooks):
    # search_runbooks is a TICKET_ID_AWARE_READ_TOOL, not a TICKET_SCOPED_TOOL:
    # it takes an executor-injected ticket_id purely to deliver the
    # adversarial injection fixtures (tools/injection_fixtures.py), never as
    # write-scoping. See agent/tool_executor.py.
    record = execute_tool_call(_call("search_runbooks", {"query": "db pool exhausted"}), TICKET_ID)
    assert record["status"] == "ok"
    assert stub_search_runbooks["ticket_id"] == TICKET_ID


def test_search_runbooks_overwrites_model_supplied_ticket_id(stub_search_runbooks):
    record = execute_tool_call(
        _call("search_runbooks", {"query": "db pool exhausted", "ticket_id": "T999"}),
        TICKET_ID,
    )
    assert stub_search_runbooks["ticket_id"] == TICKET_ID
    assert record["arguments"]["ticket_id"] == TICKET_ID
    assert record["arguments"] == {"query": "db pool exhausted", "ticket_id": TICKET_ID}


@pytest.fixture
def stub_search_past_incidents(monkeypatch):
    received = {}

    def _stub(**kwargs):
        received.update(kwargs)
        return {"status": "ok", "data": {"query": kwargs.get("query"), "incidents": [], "top_score": 0.0}, "summary": "stub"}

    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "search_past_incidents", _stub)
    return received


def test_search_past_incidents_gets_ticket_id_injected(stub_search_past_incidents):
    # Same TICKET_ID_AWARE_READ_TOOL rationale as search_runbooks above.
    record = execute_tool_call(_call("search_past_incidents", {"query": "db pool exhausted"}), TICKET_ID)
    assert record["status"] == "ok"
    assert stub_search_past_incidents["ticket_id"] == TICKET_ID


def test_search_past_incidents_overwrites_model_supplied_ticket_id(stub_search_past_incidents):
    record = execute_tool_call(
        _call("search_past_incidents", {"query": "db pool exhausted", "ticket_id": "T999"}),
        TICKET_ID,
    )
    assert stub_search_past_incidents["ticket_id"] == TICKET_ID
    assert record["arguments"]["ticket_id"] == TICKET_ID
    assert record["arguments"] == {"query": "db pool exhausted", "ticket_id": TICKET_ID}


def test_ticket_scoped_tool_gets_ticket_id_injected():
    record = execute_tool_call(
        _call("query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"}),
        TICKET_ID,
    )
    assert record["arguments"]["ticket_id"] == TICKET_ID


def test_ticket_scoped_tool_overwrites_model_supplied_wrong_ticket_id():
    record = execute_tool_call(
        _call(
            "query_logs",
            {"service": "checkout", "window": "1h", "level": "ERROR", "ticket_id": "T999"},
        ),
        TICKET_ID,
    )
    assert record["arguments"]["ticket_id"] == TICKET_ID


def test_unknown_tool_returns_error():
    record = execute_tool_call(_call("delete_everything", {}), TICKET_ID)
    assert record["status"] == "error"
    assert "Unknown tool" in record["result"]["summary"]


def test_malformed_json_arguments_returns_error():
    tool_call = _StubToolCall("query_logs", "{not valid json")
    record = execute_tool_call(tool_call, TICKET_ID)
    assert record["status"] == "error"
    assert "Could not parse arguments" in record["result"]["summary"]


def test_ticket_scoped_and_registry_stay_in_sync():
    # This test exists to catch registry/scoping drift: a future tool added
    # to TOOL_REGISTRY but forgotten in TICKET_SCOPED_TOOLS would otherwise
    # surface as a mystery TypeError at runtime instead of failing here.
    # Every tool must be consciously placed in one bucket or the other.
    assert TICKET_SCOPED_TOOLS <= set(TOOL_REGISTRY)
    # update_ticket is ticket-scoped because it is a WRITE (approval-gated) tool.
    non_scoped = set(TOOL_REGISTRY) - TICKET_SCOPED_TOOLS
    assert non_scoped == {"search_runbooks", "search_past_incidents"}
    # Pin the full registry so a future tool cannot be added unnoticed.
    assert set(TOOL_REGISTRY) == {
        "query_logs",
        "query_metrics",
        "search_runbooks",
        "search_past_incidents",
        "update_ticket",
    }
