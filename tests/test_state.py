import pytest
from pydantic import ValidationError

from agent.state import TaskState


def _make_state(**overrides):
    defaults = {"ticket_id": "TKT-00000001", "description": "Checkout returning 500s"}
    defaults.update(overrides)
    return TaskState(**defaults)


def test_defaults_on_minimal_construction():
    state = _make_state()

    assert state.status == "new"
    assert state.hypothesis is None
    assert state.confidence == 0.0
    assert state.evidence_sources == []
    assert state.iteration == 0
    assert state.max_iterations == 8
    assert state.called_tool_signatures == set()
    assert state.trajectory == []
    assert state.plan == []


def test_confidence_bounds_rejected():
    with pytest.raises(ValidationError):
        _make_state(confidence=1.5)

    with pytest.raises(ValidationError):
        _make_state(confidence=-0.1)


def test_confidence_bounds_accepted():
    assert _make_state(confidence=0.75).confidence == 0.75
    assert _make_state(confidence=1.0).confidence == 1.0


@pytest.mark.parametrize("status", ["new", "running", "resolved", "escalated", "error"])
def test_status_accepted_values(status):
    assert _make_state(status=status).status == status


def test_status_rejects_bogus_value():
    with pytest.raises(ValidationError):
        _make_state(status="bogus")


def test_called_tool_signatures_dedupes():
    state = _make_state()
    signature = ("query_logs", '{"service": "checkout"}')

    state.called_tool_signatures.add(signature)
    state.called_tool_signatures.add(signature)

    assert len(state.called_tool_signatures) == 1


def test_trajectory_entry_round_trips_documented_shape():
    entry = {
        "iteration": 1,
        "thought": "Check checkout service logs",
        "tool_call": {"name": "query_logs", "args": {"service": "checkout"}},
        "observation": {"errors": ["500 Internal Server Error"]},
        "hypothesis_after": "Checkout service is failing due to downstream timeout",
    }
    state = _make_state(trajectory=[entry])

    assert state.trajectory == [entry]
    assert set(state.trajectory[0].keys()) == {
        "iteration",
        "thought",
        "tool_call",
        "observation",
        "hypothesis_after",
    }


def test_called_tool_signatures_excluded_from_serialization():
    state = _make_state()
    state.called_tool_signatures.add(("query_logs", '{"service": "checkout"}'))

    dumped = state.model_dump()
    assert "called_tool_signatures" not in dumped

    import json

    dumped_json = json.loads(state.model_dump_json())
    assert "called_tool_signatures" not in dumped_json


def test_iteration_count_field_removed():
    assert "iteration_count" not in TaskState.model_fields
