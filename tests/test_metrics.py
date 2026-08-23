"""Offline, hand-built-fixture tests for eval/metrics.py. No dependency on
eval/results/raw and no network calls."""

from __future__ import annotations

from eval.known_issues import KnownIssue
from eval.metrics import (
    RateCard,
    citation_presence,
    classify_outcome,
    correct_escalation,
    correct_resolution_status_only,
    correct_resolution_strict,
    efficiency,
    estimated_cost_usd,
    hypothesis_matches_gold,
    hypothesis_semantic_verdict,
    loop_tool_calls,
    observed_retrieval_rank,
    parameter_validity,
    retrieval_recall_at_3_observed,
    state_consistency,
    task_success_status_only,
    task_success_strict_lexical,
    terminal_write_entry_index,
    tool_selection_accuracy,
    unnecessary_tool_calls,
    write_gate_appended_correctly,
)


# --- hypothesis_matches_gold -------------------------------------------


def test_hypothesis_matches_gold_positive():
    result = hypothesis_matches_gold(
        "The db connection pool is exhausted under load.",
        "db_connection_pool_exhaustion",
    )
    assert result["matched"] is True
    assert result["missing_tokens"] == []
    assert result["rule"] == "all-significant-tokens"


def test_hypothesis_matches_gold_negative_missing_token():
    result = hypothesis_matches_gold(
        "The database is slow.",
        "db_connection_pool_exhaustion",
    )
    assert result["matched"] is False
    assert "exhaustion" in result["missing_tokens"] or "exhaust" in result["missing_tokens"] or result["missing_tokens"]


def test_hypothesis_matches_gold_none_gold():
    result = hypothesis_matches_gold("anything", None)
    assert result["matched"] is False
    assert result["rule"] == "n/a-no-gold"


def test_hypothesis_matches_gold_suffix_variant():
    # gold token "exhaustion" must match hypothesis phrasing "exhausted"/"exhausting".
    result = hypothesis_matches_gold(
        "Connections are exhausting the pool.",
        "connection_pool_exhaustion",
    )
    assert result["matched"] is True


# --- correct_resolution (status-only / strict) / correct_escalation / task_success ---


def _ticket(expected_behavior="resolve", gold_root_cause="disk_log_rotation_gap", **kw):
    t = {
        "category": "disk",
        "gold_root_cause": gold_root_cause,
        "gold_runbook_id": "RB-DISK-001.md",
        "required_tools": ["query_logs", "query_metrics"],
        "expected_behavior": expected_behavior,
        "min_confidence_evidence_sources": 1,
    }
    t.update(kw)
    return t


def test_correct_resolution_status_only_ignores_hypothesis_content():
    matching_state = {"status": "resolved", "hypothesis": "disk log rotation gap causing overflow"}
    unrelated_state = {"status": "resolved", "hypothesis": "unrelated cause"}
    assert correct_resolution_status_only(matching_state, _ticket()) is True
    assert correct_resolution_status_only(unrelated_state, _ticket()) is True


def test_correct_resolution_strict_true():
    state = {"status": "resolved", "hypothesis": "disk log rotation gap causing overflow"}
    assert correct_resolution_strict(state, _ticket()) is True


def test_correct_resolution_strict_false_wrong_hypothesis():
    state = {"status": "resolved", "hypothesis": "unrelated cause"}
    assert correct_resolution_strict(state, _ticket()) is False


def test_correct_escalation_true():
    state = {"status": "escalated"}
    ticket = _ticket(expected_behavior="escalate", gold_root_cause=None)
    assert correct_escalation(state, ticket) is True


def test_task_success_status_only_dispatches_on_expected_behavior():
    resolve_ticket = _ticket(expected_behavior="resolve")
    escalate_ticket = _ticket(expected_behavior="escalate", gold_root_cause=None)

    resolved_state = {"status": "resolved", "hypothesis": "unrelated cause"}
    escalated_state = {"status": "escalated"}

    assert task_success_status_only(resolved_state, resolve_ticket) is True
    assert task_success_status_only(escalated_state, resolve_ticket) is False
    assert task_success_status_only(escalated_state, escalate_ticket) is True
    assert task_success_status_only(resolved_state, escalate_ticket) is False


def test_task_success_strict_lexical_dispatches_on_expected_behavior():
    resolve_ticket = _ticket(expected_behavior="resolve")
    escalate_ticket = _ticket(expected_behavior="escalate", gold_root_cause=None)

    resolved_state = {"status": "resolved", "hypothesis": "disk log rotation gap"}
    escalated_state = {"status": "escalated"}

    assert task_success_strict_lexical(resolved_state, resolve_ticket) is True
    assert task_success_strict_lexical(escalated_state, resolve_ticket) is False
    assert task_success_strict_lexical(escalated_state, escalate_ticket) is True
    assert task_success_strict_lexical(resolved_state, escalate_ticket) is False


def test_status_only_and_strict_lexical_agree_on_every_escalation_fixture():
    # Escalation has no hypothesis component, so the two measures must
    # always agree -- for both a genuinely escalate-expected ticket and a
    # resolve-expected ticket that instead escalated.
    escalate_ticket = _ticket(expected_behavior="escalate", gold_root_cause=None)
    resolve_ticket = _ticket(expected_behavior="resolve")
    escalated_state = {"status": "escalated"}

    assert (
        task_success_status_only(escalated_state, escalate_ticket)
        == task_success_strict_lexical(escalated_state, escalate_ticket)
    )
    assert (
        task_success_status_only(escalated_state, resolve_ticket)
        == task_success_strict_lexical(escalated_state, resolve_ticket)
    )


def test_hypothesis_semantic_verdict_is_always_pending_not_a_verdict():
    matching_state = {"status": "resolved", "hypothesis": "disk log rotation gap causing overflow"}
    missing_token_state = {"status": "resolved", "hypothesis": "log rotation gap"}
    ticket = _ticket()

    for state in (matching_state, missing_token_state):
        result = hypothesis_semantic_verdict(state, ticket)
        assert result["verdict"] == "pending"
        assert "not built" in result["reason"]
        assert isinstance(result["strict_lexical"], bool)
        assert isinstance(result["missing_tokens"], list)
        # never a boolean pass/fail masquerading as the verdict
        assert result["verdict"] not in (True, False)


# --- terminal_write_entry_index / loop_tool_calls -----------------------


def _entry(iteration, tool_name=None, observation=None):
    return {
        "iteration": iteration,
        "thought": "t",
        "tool_call": {"name": tool_name, "arguments": {}} if tool_name else None,
        "observation": observation,
        "hypothesis_after": None,
    }


def test_terminal_write_entry_identified_by_iteration_not_name_alone():
    # An IN-LOOP update_ticket call at an earlier iteration must NOT be
    # treated as terminal.
    state = {
        "iteration": 3,
        "trajectory": [
            _entry(0, "query_logs"),
            _entry(1, "update_ticket"),  # in-loop, must be ignored
            _entry(2, "search_runbooks"),
            _entry(3, "update_ticket"),  # terminal
        ],
    }
    idx = terminal_write_entry_index(state)
    assert idx == 3


def test_terminal_write_entry_none_when_absent():
    state = {
        "iteration": 2,
        "trajectory": [_entry(0, "query_logs"), _entry(1, "search_runbooks")],
    }
    assert terminal_write_entry_index(state) is None


def test_loop_tool_calls_excludes_terminal_entry():
    state = {
        "iteration": 2,
        "trajectory": [
            _entry(0, "query_logs"),
            _entry(1, "search_runbooks"),
            _entry(2, "update_ticket"),
        ],
    }
    calls = loop_tool_calls(state)
    names = [c["tool_call"]["name"] for c in calls]
    assert names == ["query_logs", "search_runbooks"]


def test_loop_tool_calls_includes_in_loop_update_ticket():
    # An update_ticket call NOT at the terminal iteration is a loop call.
    state = {
        "iteration": 3,
        "trajectory": [
            _entry(0, "query_logs"),
            _entry(1, "update_ticket"),
            _entry(3, "update_ticket"),
        ],
    }
    calls = loop_tool_calls(state)
    names = [c["tool_call"]["name"] for c in calls]
    assert names == ["query_logs", "update_ticket"]


def test_tool_selection_accuracy_and_unnecessary_calls():
    state = {
        "iteration": 2,
        "trajectory": [
            _entry(0, "query_logs"),
            _entry(1, "search_past_incidents"),
            _entry(2, "update_ticket"),
        ],
    }
    ticket = _ticket()  # required_tools = query_logs, query_metrics
    assert tool_selection_accuracy(state, ticket) == 0.5
    assert unnecessary_tool_calls(state, ticket) == 1


def test_tool_selection_accuracy_none_when_no_loop_calls():
    state = {"iteration": 0, "trajectory": []}
    assert tool_selection_accuracy(state, _ticket()) is None


def test_parameter_validity_not_measurable():
    state = {
        "iteration": 1,
        "trajectory": [_entry(0, "query_logs", observation={"status": "ok"})],
    }
    result = parameter_validity(state)
    assert result["measurable"] is False
    assert result["total_calls"] == 1
    assert result["observed_failures"] == 0


# --- state_consistency ---------------------------------------------------


def test_state_consistency_true():
    state = {
        "iteration": 3,
        "trajectory": [_entry(0), _entry(1), _entry(2), _entry(3, "update_ticket")],
    }
    assert state_consistency(state) is True


def test_state_consistency_false_missing_iteration():
    state = {
        "iteration": 3,
        "trajectory": [_entry(0), _entry(0), _entry(2), _entry(3, "update_ticket")],
    }
    assert state_consistency(state) is False


# --- write_gate_appended_correctly ----------------------------------------


def test_write_gate_appended_correctly_true():
    state = {
        "iteration": 2,
        "pending_action_id": "abc123",
        "trajectory": [_entry(0), _entry(1), _entry(2, "update_ticket")],
    }
    assert write_gate_appended_correctly(state) is True


def test_write_gate_appended_correctly_false_no_pending_action():
    state = {
        "iteration": 2,
        "pending_action_id": None,
        "trajectory": [_entry(0), _entry(1), _entry(2, "update_ticket")],
    }
    assert write_gate_appended_correctly(state) is False


def test_write_gate_appended_correctly_none_when_never_reached():
    state = {
        "iteration": 8,
        "pending_action_id": None,
        "trajectory": [_entry(i) for i in range(8)],
    }
    assert write_gate_appended_correctly(state) is None


# --- observed retrieval rank / recall@3 -----------------------------------


def _search_runbooks_entry(iteration, doc_ids, status="ok"):
    return _entry(
        iteration,
        "search_runbooks",
        observation={
            "status": status,
            "data": {"chunks": [{"doc_id": d} for d in doc_ids]},
        },
    )


def test_observed_retrieval_rank_and_recall_at_3():
    state = {
        "iteration": 1,
        "trajectory": [_search_runbooks_entry(0, ["RB-NETWORK-001", "RB-DISK-001", "RB-DB-001"])],
    }
    ticket = _ticket(gold_root_cause="disk_log_rotation_gap")
    assert observed_retrieval_rank(state, ticket) == 2
    assert retrieval_recall_at_3_observed(state, ticket) is True


def test_observed_retrieval_rank_none_when_gold_absent():
    state = {
        "iteration": 1,
        "trajectory": [_search_runbooks_entry(0, ["RB-NETWORK-001"])],
    }
    ticket = _ticket(gold_root_cause="disk_log_rotation_gap")
    assert observed_retrieval_rank(state, ticket) is None
    assert retrieval_recall_at_3_observed(state, ticket) is None


# --- citation_presence -----------------------------------------------------


def test_citation_presence_true_when_resolved_with_citations():
    state = {"status": "resolved", "citations": ["RB-DISK-001"]}
    assert citation_presence(state) is True


def test_citation_presence_none_when_not_resolved():
    state = {"status": "escalated", "citations": []}
    assert citation_presence(state) is None


def test_citation_presence_false_when_resolved_without_citations():
    state = {"status": "resolved", "citations": []}
    assert citation_presence(state) is False


# --- efficiency / cost -----------------------------------------------------


def test_estimated_cost_usd_arithmetic():
    usage = {"total_tokens_in": 1_000_000, "total_tokens_out": 500_000}
    rate = RateCard(input_per_mtok=0.15, output_per_mtok=0.60)
    cost = estimated_cost_usd(usage, rate)
    assert round(cost, 4) == round(0.15 + 0.30, 4)


def test_efficiency_aggregates_fields():
    raw = {
        "run": {"wall_clock_seconds": 12.5},
        "usage": {"llm_call_count": 4, "total_tokens_in": 2000, "total_tokens_out": 500},
        "state": {
            "iteration": 1,
            "trajectory": [_entry(0, "query_logs"), _entry(1, "update_ticket")],
        },
    }
    result = efficiency(raw)
    assert result["llm_call_count"] == 4
    assert result["tool_call_count"] == 1
    assert result["wall_clock_seconds"] == 12.5
    assert result["estimated_cost_usd"] > 0


# --- classify_outcome -------------------------------------------------------


def test_classify_outcome_crashed():
    raw = {"ticket_id": "T999", "run": {"runner_error": "boom"}, "state": None}
    result = classify_outcome(raw, _ticket())
    assert result["bucket"] == "crashed"
    assert result["task_success_status_only"] is False
    assert result["task_success_strict_lexical"] is False


def test_classify_outcome_success():
    raw = {
        "ticket_id": "T001",
        "run": {"runner_error": None},
        "state": {"status": "resolved", "hypothesis": "disk log rotation gap"},
    }
    result = classify_outcome(raw, _ticket())
    assert result["bucket"] == "success"
    assert result["task_success_status_only"] is True
    assert result["task_success_strict_lexical"] is True
    assert result["hypothesis_match"]["matched"] is True


def test_classify_outcome_success_status_only_but_strict_lexical_false():
    # A resolved ticket whose hypothesis is missing exactly one gold token
    # must still bucket as "success" (status-only), while separately
    # reporting strict_lexical=False and the missing token -- the gap
    # between the two measures is surfaced, not hidden or bucketed on.
    raw = {
        "ticket_id": "T042",
        "run": {"runner_error": None},
        "state": {"status": "resolved", "hypothesis": "log rotation gap"},
    }
    result = classify_outcome(raw, _ticket())
    assert result["bucket"] == "success"
    assert result["task_success_status_only"] is True
    assert result["task_success_strict_lexical"] is False
    assert result["hypothesis_match"]["matched"] is False
    assert result["hypothesis_match"]["missing_tokens"]


def test_classify_outcome_known_issue():
    raw = {
        "ticket_id": "T019",
        "run": {"runner_error": None},
        "state": {"status": "escalated", "hypothesis": None},
    }
    ticket = _ticket(expected_behavior="resolve")
    known = KnownIssue(
        ticket_ids=("T019",),
        cause="asks user instead of investigating",
        detail="detail",
        evidence="evidence",
        expect_status="escalated",
    )
    result = classify_outcome(raw, ticket, known_issue=known)
    assert result["bucket"] == "known_issue"
    assert result["known_issue_id"] == "T019"
    assert result["task_success_status_only"] is False


def test_classify_outcome_stale_known_issue():
    raw = {
        "ticket_id": "T019",
        "run": {"runner_error": None},
        "state": {"status": "resolved", "hypothesis": "disk log rotation gap"},
    }
    ticket = _ticket(expected_behavior="resolve")
    known = KnownIssue(
        ticket_ids=("T019",),
        cause="asks user instead of investigating",
        detail="detail",
        evidence="evidence",
        expect_status="escalated",
    )
    result = classify_outcome(raw, ticket, known_issue=known)
    assert result["bucket"] == "stale_known_issue"
    assert result["task_success_status_only"] is True


def test_classify_outcome_wrong_status_is_unexplained_failure():
    # T029's documented failure mode expects "escalated"; if it instead
    # errors out with a DIFFERENT failing status, it must NOT be silently
    # absorbed into the known bucket.
    raw = {
        "ticket_id": "T029",
        "run": {"runner_error": None},
        "state": {"status": "error", "hypothesis": None},
    }
    ticket = _ticket(expected_behavior="resolve")
    known = KnownIssue(
        ticket_ids=("T025", "T029"),
        cause="loop-guard recovery",
        detail="detail",
        evidence="evidence",
        expect_status="escalated",
    )
    result = classify_outcome(raw, ticket, known_issue=known)
    assert result["bucket"] == "unexplained_failure"
    assert result["known_issue_id"] is None


def test_classify_outcome_unexplained_failure_no_known_issue():
    raw = {
        "ticket_id": "T500",
        "run": {"runner_error": None},
        "state": {"status": "escalated", "hypothesis": None},
    }
    ticket = _ticket(expected_behavior="resolve")
    result = classify_outcome(raw, ticket)
    assert result["bucket"] == "unexplained_failure"


def test_classify_outcome_crashed_flows_through_other_metrics_without_raising():
    raw = {"ticket_id": "T999", "run": {"runner_error": "boom"}, "state": None}
    ticket = _ticket()
    state = raw["state"]

    assert correct_resolution_status_only(state, ticket) is False
    assert correct_resolution_strict(state, ticket) is False
    assert correct_escalation(state, ticket) is False
    assert task_success_status_only(state, ticket) is False
    assert task_success_strict_lexical(state, ticket) is False
    verdict = hypothesis_semantic_verdict(state, ticket)
    assert verdict["verdict"] == "pending"
    assert terminal_write_entry_index(state) is None
    assert loop_tool_calls(state) == []
    assert tool_selection_accuracy(state, ticket) is None
    assert unnecessary_tool_calls(state, ticket) == 0
    assert parameter_validity(state) == {
        "observed_failures": 0,
        "total_calls": 0,
        "measurable": False,
    }
    assert state_consistency(state) is False
    assert write_gate_appended_correctly(state) is None
    assert observed_retrieval_rank(state, ticket) is None
    assert retrieval_recall_at_3_observed(state, ticket) is None
    assert citation_presence(state) is None
