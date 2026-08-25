"""Tests for eval/diagnose_underresolution.py -- fully offline, no live
sweep dependency. Raw result files are hand-built under tmp_path in the
same schema eval/run_benchmark.py writes (schema_version/ticket_id/run/
ticket/state/usage)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.diagnose_underresolution import analyze


def _base_state(**overrides) -> dict:
    state = {
        "ticket_id": "T000",
        "description": "desc",
        "status": "escalated",
        "plan": [],
        "hypothesis": "some hypothesis",
        "confidence": 0.9,
        "evidence_sources": ["query_logs", "search_runbooks"],
        "citations": ["RB-DB-001.md#0"],
        "iteration": 5,
        "max_iterations": 8,
        "trajectory": [
            {
                "iteration": 1,
                "thought": "t",
                "tool_call": {"name": "search_runbooks", "args": {}},
                "observation": {
                    "status": "ok",
                    "data": {"chunks": [{"doc_id": "RB-DB-001.md#0"}]},
                },
                "hypothesis_after": None,
            }
        ],
        "pending_action_id": None,
    }
    state.update(overrides)
    return state


def _write_raw(path: Path, ticket_id: str, state: dict | None, expected_behavior: str,
                runner_error=None) -> None:
    result = {
        "schema_version": 1,
        "ticket_id": ticket_id,
        "run": {"started_at": "x", "wall_clock_seconds": 0.1, "runner_error": runner_error},
        "ticket": {
            "category": "medium",
            "gold_root_cause": "x",
            "gold_runbook_id": "RB-DB-001.md",
            "required_tools": [],
            "expected_behavior": expected_behavior,
            "min_confidence_evidence_sources": 2,
        },
        "state": state,
        "usage": {},
    }
    (path / f"{ticket_id}.json").write_text(json.dumps(result), encoding="utf-8")


def test_confidence_alone(tmp_path):
    state = _base_state(ticket_id="T001", confidence=0.5)
    _write_raw(tmp_path, "T001", state, "resolve_with_approval")

    result = analyze(tmp_path)
    entry = result["analyzed"][0]
    assert entry["failed_conditions"] == ["confidence"]
    assert result["summary"]["confidence_alone"] == 1
    assert result["summary"]["evidence_count_alone"] == 0


def test_runbook_credit_alone(tmp_path):
    state = _base_state(
        ticket_id="T002",
        evidence_sources=["query_logs", "query_metrics"],
        trajectory=[
            {
                "iteration": 1,
                "thought": "t",
                "tool_call": {"name": "query_logs", "args": {}},
                "observation": {"status": "ok", "data": {}},
                "hypothesis_after": None,
            }
        ],
        citations=[],
    )
    _write_raw(tmp_path, "T002", state, "resolve_with_approval")

    result = analyze(tmp_path)
    entry = result["analyzed"][0]
    # runbook_credited AND has_citations AND citations_grounded all fail
    # (no citations, no search_runbooks credited) but confidence and
    # evidence_count both pass.
    assert "confidence" not in entry["failed_conditions"]
    assert "evidence_count" not in entry["failed_conditions"]
    assert "runbook_credited" in entry["failed_conditions"]


def test_multi_failure_appears_in_all_buckets_and_signature(tmp_path):
    state = _base_state(
        ticket_id="T003",
        confidence=0.3,
        evidence_sources=["query_logs"],
        citations=[],
        trajectory=[],
    )
    _write_raw(tmp_path, "T003", state, "resolve_with_approval")

    result = analyze(tmp_path)
    entry = result["analyzed"][0]
    failed = set(entry["failed_conditions"])
    assert "confidence" in failed
    assert "evidence_count" in failed
    assert "runbook_credited" in failed
    assert "has_citations" in failed
    summary = result["summary"]
    assert summary["per_condition_fail_counts"]["confidence"] == 1
    assert summary["per_condition_fail_counts"]["evidence_count"] == 1
    sig = tuple(entry["failed_conditions"])
    signatures = {tuple(row["signature"]) for row in summary["failure_signature_ranked"]}
    assert sig in signatures


def test_conditions_1_and_4_both_reported(tmp_path):
    """A ticket failing confidence (1) and runbook_credited (4) reports BOTH,
    not just the first found."""
    state = _base_state(
        ticket_id="T004",
        confidence=0.2,
        evidence_sources=["query_logs", "query_metrics"],
        trajectory=[],
        citations=[],
    )
    _write_raw(tmp_path, "T004", state, "resolve_with_approval")

    result = analyze(tmp_path)
    entry = result["analyzed"][0]
    assert "confidence" in entry["failed_conditions"]
    assert "runbook_credited" in entry["failed_conditions"]


def test_crashed_ticket_skipped_but_counted(tmp_path):
    _write_raw(
        tmp_path, "T005", None, "resolve_with_approval",
        runner_error={"type": "ValueError", "message": "boom", "traceback": "..."},
    )

    result = analyze(tmp_path)
    assert result["analyzed"] == []
    assert result["summary"]["crashed_count"] == 1
    assert result["summary"]["total_raw_files"] == 1


def test_expected_escalate_not_analyzed(tmp_path):
    state = _base_state(ticket_id="T006", confidence=0.2)
    _write_raw(tmp_path, "T006", state, "escalate")

    result = analyze(tmp_path)
    assert result["analyzed"] == []
    assert result["summary"]["skipped_not_selected"] == 1


def test_search_runbooks_called_but_not_credited_detector(tmp_path):
    state = _base_state(
        ticket_id="T007",
        evidence_sources=["query_logs", "query_metrics"],
        citations=[],
        trajectory=[
            {
                "iteration": 1,
                "thought": "t",
                "tool_call": {"name": "search_runbooks", "args": {}},
                "observation": {"status": "no_confident_match", "data": {}},
                "hypothesis_after": None,
            }
        ],
    )
    _write_raw(tmp_path, "T007", state, "resolve_with_approval")

    result = analyze(tmp_path)
    entry = result["analyzed"][0]
    assert entry["runbook_called_not_credited"] is True
    assert result["summary"]["runbook_called_not_credited_count"] == 1
