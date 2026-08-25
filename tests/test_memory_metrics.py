import eval.memory_metrics as mm


def _warrant(rows):
    return {"schema_version": 1, "threshold": 0.40, "embedding_model": "m", "n_past_incidents": 8, "n_tickets": len(rows), "tickets": rows}


def _row(id_, classification, gold="db_connection_pool_exhaustion"):
    return {
        "id": id_,
        "category": "easy",
        "gold_root_cause": gold,
        "memory_status": "ok" if classification != "no_signal" else "no_confident_match",
        "top_incident_id": "INC-1",
        "top_resolved_root_cause": gold,
        "top_score": 0.6,
        "classification": classification,
    }


def _state(called_memory: bool, initiated_by: str | None = "model", extra_call: str | None = None):
    """extra_call, if given, adds a second search_past_incidents call with
    that initiated_by value (used to test mixed model+loop provenance).
    initiated_by=None omits the key entirely, to model old result files."""
    trajectory = [
        {
            "iteration": 0,
            "tool_call": {"name": "query_logs", "arguments": {}},
            "observation": {"status": "ok"},
        }
    ]
    if called_memory:
        call = {"name": "search_past_incidents", "arguments": {}}
        if initiated_by is not None:
            call["initiated_by"] = initiated_by
        trajectory.append(
            {
                "iteration": 0,
                "tool_call": call,
                "observation": {"status": "ok"},
            }
        )
    if extra_call is not None:
        trajectory.append(
            {
                "iteration": 0,
                "tool_call": {
                    "name": "search_past_incidents",
                    "arguments": {},
                    "initiated_by": extra_call,
                },
                "observation": {"status": "ok"},
            }
        )
    trajectory.append(
        {
            "iteration": 1,
            "tool_call": {"name": "update_ticket", "arguments": {}},
            "observation": {"status": "ok"},
        }
    )
    return {
        "ticket_id": "T",
        "status": "resolved",
        "hypothesis": "x",
        "iteration": 1,
        "trajectory": trajectory,
        "pending_action_id": "abc",
        "citations": [],
    }


def _raw(ticket_id, state):
    return {"ticket_id": ticket_id, "state": state}


def test_recall_counts_called_warranted_tickets():
    warrant = _warrant([_row("T001", "warranted"), _row("T002", "warranted")])
    results = [_raw("T001", _state(True)), _raw("T002", _state(False))]
    out = mm.memory_invocation_recall(warrant, results)
    assert out == {
        "warranted_total": 2,
        "called_by_model": 1,
        "called_by_loop": 0,
        "called_by_any": 1,
        "unknown_provenance": [],
        "rate_model": 0.5,
        "rate_loop": 0.0,
        "rate_any": 0.5,
        "missed_by_model": ["T002"],
        "not_scored": [],
    }


def test_recall_empty_denominator_is_none_not_zero():
    warrant = _warrant([_row("T001", "no_signal")])
    results = [_raw("T001", _state(False))]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["warranted_total"] == 0
    assert out["rate_model"] is None
    assert out["rate_loop"] is None
    assert out["rate_any"] is None


def test_recall_reports_not_scored_and_excludes_from_denominator():
    warrant = _warrant([_row("T001", "warranted"), _row("T002", "warranted")])
    # T002 never ran (missing from results).
    results = [_raw("T001", _state(True))]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["warranted_total"] == 1
    assert out["called_by_model"] == 1
    assert out["rate_model"] == 1.0
    assert out["not_scored"] == ["T002"]


def test_recall_crashed_ticket_is_not_scored():
    warrant = _warrant([_row("T001", "warranted")])
    results = [_raw("T001", None)]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["warranted_total"] == 0
    assert out["not_scored"] == ["T001"]
    assert out["rate_model"] is None


def test_precision_counts_calls_landing_on_warranted():
    warrant = _warrant(
        [
            _row("T001", "warranted"),
            _row("T002", "no_signal"),
        ]
    )
    results = [_raw("T001", _state(True)), _raw("T002", _state(True))]
    out = mm.memory_invocation_precision(warrant, results)
    assert out["called_total_model"] == 2
    assert out["called_and_warranted_model"] == 1
    assert out["rate_model"] == 0.5
    assert out["unwarranted_calls_model"] == ["T002"]
    assert out["not_scored"] == []


def test_precision_empty_denominator_is_none_not_zero():
    warrant = _warrant([_row("T001", "warranted")])
    results = [_raw("T001", _state(False))]
    out = mm.memory_invocation_precision(warrant, results)
    assert out["called_total_model"] == 0
    assert out["rate_model"] is None


def test_hazard_exposure_counts_calls_on_hazard_tickets():
    warrant = _warrant(
        [
            _row("T001", "hazard", gold=None),
            _row("T002", "hazard", gold=None),
        ]
    )
    results = [_raw("T001", _state(True)), _raw("T002", _state(False))]
    out = mm.memory_hazard_exposure(warrant, results)
    assert out["called_by_model"] == 1
    assert out["hazard_total"] == 2
    assert out["tickets_model"] == ["T001"]
    assert out["not_scored"] == []


def test_hazard_exposure_not_scored():
    warrant = _warrant([_row("T001", "hazard", gold=None)])
    results = []
    out = mm.memory_hazard_exposure(warrant, results)
    assert out["hazard_total"] == 0
    assert out["not_scored"] == ["T001"]


# --- provenance-aware behaviour (agent/orchestrator.py's initiated_by split) ---


def test_recall_mixed_model_and_loop_call_counted_once_in_each_bucket():
    warrant = _warrant([_row("T001", "warranted")])
    # One model-initiated call and one loop-initiated call in the same trajectory.
    results = [_raw("T001", _state(True, initiated_by="model", extra_call="loop"))]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["called_by_model"] == 1
    assert out["called_by_loop"] == 1
    assert out["called_by_any"] == 1
    assert out["missed_by_model"] == []


def test_hazard_exposure_mixed_model_and_loop_call_counted_once_in_each_bucket():
    warrant = _warrant([_row("T001", "hazard", gold=None)])
    results = [_raw("T001", _state(True, initiated_by="model", extra_call="loop"))]
    out = mm.memory_hazard_exposure(warrant, results)
    assert out["called_by_model"] == 1
    assert out["called_by_loop"] == 1
    assert out["called_by_any"] == 1
    assert out["tickets_model"] == ["T001"]
    assert out["tickets_loop"] == ["T001"]


def test_recall_missing_initiated_by_lands_in_unknown_provenance_not_model():
    """Pre-change result files have no initiated_by key at all. Those calls
    must land in unknown_provenance and must NOT be counted as
    called_by_model."""
    warrant = _warrant([_row("T001", "warranted")])
    results = [_raw("T001", _state(True, initiated_by=None))]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["called_by_model"] == 0
    assert out["called_by_loop"] == 0
    assert out["called_by_any"] == 1
    assert out["unknown_provenance"] == ["T001"]
    assert out["missed_by_model"] == ["T001"]
    assert out["rate_model"] == 0.0


def test_recall_unrecognised_initiated_by_value_lands_in_unknown_provenance():
    warrant = _warrant([_row("T001", "warranted")])
    results = [_raw("T001", _state(True, initiated_by="something_else"))]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["called_by_model"] == 0
    assert out["unknown_provenance"] == ["T001"]


def test_recall_model_rate_unaffected_by_loop_initiated_calls():
    """A ticket where the ONLY memory call is loop-initiated must not count
    toward called_by_model or rate_model."""
    warrant = _warrant([_row("T001", "warranted"), _row("T002", "warranted")])
    results = [
        _raw("T001", _state(True, initiated_by="loop")),
        _raw("T002", _state(True, initiated_by="model")),
    ]
    out = mm.memory_invocation_recall(warrant, results)
    assert out["called_by_model"] == 1
    assert out["rate_model"] == 0.5
    assert out["called_by_loop"] == 1
    assert out["rate_loop"] == 0.5
    assert out["missed_by_model"] == ["T001"]
