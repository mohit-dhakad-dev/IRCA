"""Tests for the write-action approval gate (Session 7, Part B).

Fully offline: no LLM, no network. Exercises tools.ticket_tools.update_ticket,
the /approvals endpoints, and the static invariant that the write path does
not exist outside the approval flow.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agent.approval as approval_module
import tools.ticket_store as ticket_store_module
from agent.approval import list_pending_actions
from agent.tool_executor import execute_tool_call
from main import app
from tools.ticket_store import RESOLVED_TICKETS
from tools.ticket_tools import update_ticket

client = TestClient(app)

VIOLATING_FIX = "Set log rotation to 500 MB per file"
PASSING_FIX = "Cap each log file at 50 MB and retain 3 files"


@pytest.fixture(autouse=True)
def _clear_stores():
    approval_module.clear_store()
    ticket_store_module.clear_store()
    yield
    approval_module.clear_store()
    ticket_store_module.clear_store()


def test_violating_fix_is_rejected_and_nothing_queued():
    result = update_ticket(
        ticket_id="T001",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=VIOLATING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    assert result["status"] == "verification_failed"
    assert "reason" in result["data"]
    assert list_pending_actions() == []


def test_valid_fix_queues_pending_action_without_mutating_ticket_store():
    result = update_ticket(
        ticket_id="T001",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    assert result["status"] == "awaiting_approval"
    action_id = result["data"]["action_id"]
    assert action_id

    pending = approval_module.get_pending_action(action_id)
    assert pending.status == "pending"

    # Core assertion of this phase: queueing a proposal must never itself
    # mutate the resolution store -- only an explicit human approval may.
    assert RESOLVED_TICKETS == {}


def test_approve_pending_action_applies_write():
    result = update_ticket(
        ticket_id="T001",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id = result["data"]["action_id"]

    response = client.post(f"/approvals/{action_id}/approve")
    assert response.status_code == 200

    resolution = RESOLVED_TICKETS.get("T001")
    assert resolution is not None
    assert resolution["root_cause"] == "disk_log_rotation_gap"
    assert resolution["fix"] == PASSING_FIX
    assert resolution["citation_doc_id"] == "RB-DISK-001"

    action = approval_module.get_pending_action(action_id)
    assert action.status == "approved"


def test_reject_pending_action_leaves_store_empty():
    result = update_ticket(
        ticket_id="T002",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id = result["data"]["action_id"]

    response = client.post(f"/approvals/{action_id}/reject")
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    action = approval_module.get_pending_action(action_id)
    assert action.status == "rejected"
    assert RESOLVED_TICKETS == {}


def test_double_decision_returns_409_and_does_not_mutate():
    result = update_ticket(
        ticket_id="T003",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id = result["data"]["action_id"]

    approve_resp = client.post(f"/approvals/{action_id}/approve")
    assert approve_resp.status_code == 200

    reject_after_approve = client.post(f"/approvals/{action_id}/reject")
    assert reject_after_approve.status_code == 409
    assert RESOLVED_TICKETS.get("T003") is not None  # unchanged from the approval

    result2 = update_ticket(
        ticket_id="T004",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id2 = result2["data"]["action_id"]
    reject_resp = client.post(f"/approvals/{action_id2}/reject")
    assert reject_resp.status_code == 200

    approve_after_reject = client.post(f"/approvals/{action_id2}/approve")
    assert approve_after_reject.status_code == 409
    assert "T004" not in RESOLVED_TICKETS

    result3 = update_ticket(
        ticket_id="T005",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id3 = result3["data"]["action_id"]

    first_approve = client.post(f"/approvals/{action_id3}/approve")
    assert first_approve.status_code == 200
    first_record = dict(RESOLVED_TICKETS["T005"])

    second_approve = client.post(f"/approvals/{action_id3}/approve")
    assert second_approve.status_code == 409
    assert RESOLVED_TICKETS["T005"] == first_record  # no second write, resolved_at unchanged


def test_decisions_on_unknown_action_id_are_404():
    assert client.post("/approvals/does-not-exist/approve").status_code == 404
    assert client.post("/approvals/does-not-exist/reject").status_code == 404


def test_apply_write_raises_for_non_approved_status():
    result = update_ticket(
        ticket_id="T005",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id = result["data"]["action_id"]
    pending_action = approval_module.get_pending_action(action_id)

    with pytest.raises(ValueError):
        ticket_store_module.apply_write(pending_action)
    assert RESOLVED_TICKETS == {}

    pending_action.status = "rejected"
    with pytest.raises(ValueError):
        ticket_store_module.apply_write(pending_action)
    assert RESOLVED_TICKETS == {}


def test_executor_injects_ticket_id_overwriting_model_supplied_value():
    class _StubFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _StubToolCall:
        def __init__(self, name, arguments):
            self.function = _StubFunction(name, arguments)

    tool_call = _StubToolCall(
        "update_ticket",
        json.dumps(
            {
                "ticket_id": "T999",
                "proposed_root_cause": "disk_log_rotation_gap",
                "proposed_fix": PASSING_FIX,
                "citation_doc_id": "RB-DISK-001",
            }
        ),
    )

    record = execute_tool_call(tool_call, ticket_id="T001")
    assert record["status"] == "awaiting_approval"
    action_id = record["result"]["data"]["action_id"]

    action = approval_module.get_pending_action(action_id)
    assert action.ticket_id == "T001"  # model's "T999" was overwritten


def test_ticket_tools_does_not_import_ticket_store():
    # WEAK secondary tripwire: an AST import check can only catch static
    # `import`/`from ... import` statements -- it cannot catch
    # importlib.import_module("tools.ticket_store") or similar dynamic
    # construction of the module name. The real enforcement of "the write
    # path does not exist outside the approval flow" is the runtime
    # assertion elsewhere in this file that RESOLVED_TICKETS is still empty
    # immediately after update_ticket returns.
    source = Path("tools/ticket_tools.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "tools.ticket_store"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module != "tools.ticket_store"
            if module == "tools":
                assert not any(alias.name == "ticket_store" for alias in node.names)


def test_get_approvals_lists_queued_action_and_unknown_is_404():
    result = update_ticket(
        ticket_id="T006",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix=PASSING_FIX,
        citation_doc_id="RB-DISK-001",
    )
    action_id = result["data"]["action_id"]

    list_resp = client.get("/approvals")
    assert list_resp.status_code == 200
    ids = [a["action_id"] for a in list_resp.json()]
    assert action_id in ids

    assert client.get(f"/approvals/{action_id}").status_code == 200
    assert client.get("/approvals/does-not-exist").status_code == 404
