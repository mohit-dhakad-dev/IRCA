import json
from pathlib import Path

import pytest

from eval.ragas_inputs import _find_terminal_answer, build_all, build_ragas_input, load_chunk_index, load_tickets

RAW_DIR = Path(__file__).resolve().parent.parent / "eval" / "results" / "raw"


@pytest.fixture(scope="module")
def chunk_index():
    return load_chunk_index()


@pytest.fixture(scope="module")
def tickets():
    return load_tickets()


def test_t009(chunk_index, tickets):
    row = build_ragas_input("T009", raw_dir=RAW_DIR, tickets=tickets, chunk_index=chunk_index)

    assert row.citations == ["RB-DISK-001"]
    assert row.gold_runbook_id == "RB-DISK-001.md"
    assert len(row.contexts) == 5
    assert any("Rotate or prune logs immediately" in c for c in row.contexts)
    assert "Rotate or prune logs immediately" in row.ground_truth
    assert not row.ground_truth.startswith("RB-DISK")

    raw = json.loads((RAW_DIR / "T009.json").read_text())
    tickets_json = json.loads((RAW_DIR.parent.parent.parent / "data" / "tickets.json").read_text())
    expected_question = next(t["ticket_text"] for t in tickets_json if t["id"] == "T009")
    assert row.question == expected_question

    assert "rotat" in row.answer.lower()
    assert row.action_id == raw["state"]["pending_action_id"]


def test_t038(chunk_index, tickets):
    row = build_ragas_input("T038", raw_dir=RAW_DIR, tickets=tickets, chunk_index=chunk_index)

    assert row.citations == ["RB-DB-001"]
    assert row.gold_runbook_id == "RB-DB-001.md"
    assert len(row.contexts) == 5

    title_prefix = row.contexts[0].split(" — ")[0]
    for ctx in row.contexts:
        assert ctx.startswith(title_prefix)

    assert row.ground_truth != ""


def test_t044(chunk_index, tickets):
    row = build_ragas_input("T044", raw_dir=RAW_DIR, tickets=tickets, chunk_index=chunk_index)

    assert row.citations == ["RB-DEPLOY-001"]
    assert row.gold_runbook_id == "RB-DEPLOY-001.md"
    assert len(row.contexts) == 5
    assert row.answer != ""


def test_t011_terminal_action_id_rule(chunk_index, tickets):
    # This pins real fixture behaviour but is NOT by itself a discriminator
    # against a regression to "last update_ticket call" -- in the T011 raw
    # fixture the accepted call also happens to be last in trajectory order.
    # See test_find_terminal_answer_prefers_action_id_match_over_last_call
    # below for the test that actually fails under that regression.
    row = build_ragas_input("T011", raw_dir=RAW_DIR, tickets=tickets, chunk_index=chunk_index)

    raw = json.loads((RAW_DIR / "T011.json").read_text())
    assert row.action_id == raw["state"]["pending_action_id"]
    assert row.action_id is not None
    assert "below 380 connections" not in row.answer


def test_find_terminal_answer_prefers_action_id_match_over_last_call():
    state = {
        "pending_action_id": "act-1",
        "trajectory": [
            {
                "tool_call": {
                    "name": "update_ticket",
                    "arguments": {"proposed_fix": "ACCEPTED FIX"},
                },
                "observation": {
                    "status": "ok",
                    "data": {"action_id": "act-1"},
                },
            },
            {
                "tool_call": {
                    "name": "update_ticket",
                    "arguments": {"proposed_fix": "REJECTED LATER FIX"},
                },
                "observation": {
                    "status": "verification_failed",
                    "data": {"action_id": None},
                },
            },
        ],
    }

    answer, action_id = _find_terminal_answer("TEST", state)

    assert answer == "ACCEPTED FIX"
    assert action_id == "act-1"


def test_find_terminal_answer_raises_when_no_match():
    state = {
        "pending_action_id": "act-1",
        "trajectory": [
            {
                "tool_call": {"name": "update_ticket", "arguments": {"proposed_fix": "X"}},
                "observation": {"status": "verification_failed", "data": {"action_id": None}},
            },
        ],
    }

    with pytest.raises(ValueError):
        _find_terminal_answer("TEST", state)


def test_find_terminal_answer_raises_when_multiple_matches():
    state = {
        "pending_action_id": "act-1",
        "trajectory": [
            {
                "tool_call": {"name": "update_ticket", "arguments": {"proposed_fix": "FIRST"}},
                "observation": {"status": "ok", "data": {"action_id": "act-1"}},
            },
            {
                "tool_call": {"name": "update_ticket", "arguments": {"proposed_fix": "SECOND"}},
                "observation": {"status": "ok", "data": {"action_id": "act-1"}},
            },
        ],
    }

    with pytest.raises(ValueError):
        _find_terminal_answer("TEST", state)


def test_build_all_counts():
    rows = build_all(raw_dir=RAW_DIR)
    assert len(rows) == 49

    all_ids = sorted(p.stem for p in RAW_DIR.glob("*.json"))
    assert len(all_ids) == 63
    escalated = [
        t for t in all_ids
        if json.loads((RAW_DIR / f"{t}.json").read_text())["state"]["status"] != "resolved"
    ]
    assert len(escalated) == 14
