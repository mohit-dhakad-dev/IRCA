import json
from types import SimpleNamespace

import pytest

from agent.state import TaskState
from eval import run_benchmark


def _canned_state(ticket_id="T001"):
    state = TaskState(ticket_id=ticket_id, description="desc", status="resolved")
    state.hypothesis = "db_connection_pool_exhaustion"
    state.confidence = 0.9
    state.evidence_sources = ["query_logs", "search_runbooks"]
    state.citations = ["RB-DB-001"]
    state.trajectory.append(
        {
            "iteration": 0,
            "thought": "t",
            "tool_call": {"name": "query_logs", "arguments": {}},
            "observation": {"status": "ok", "data": {}, "summary": "s"},
            "hypothesis_after": state.hypothesis,
        }
    )
    return state


def test_happy_path_writes_expected_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    out_path = tmp_path / "T001.json"
    assert out_path.exists()
    result = json.loads(out_path.read_text())

    assert result["schema_version"] == 1
    assert result["ticket_id"] == "T001"
    assert set(result.keys()) == {"schema_version", "ticket_id", "run", "ticket", "state", "usage"}
    assert result["run"]["runner_error"] is None
    assert isinstance(result["run"]["wall_clock_seconds"], float)
    assert result["run"]["wall_clock_seconds"] >= 0

    expected_state = _canned_state("T001").model_dump()
    assert result["state"] == expected_state


def test_crash_path_writes_file_with_runner_error(tmp_path, monkeypatch):
    def _raise(ticket_id):
        raise ValueError("boom")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", _raise)

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text())
    assert result["state"] is None
    assert result["run"]["runner_error"]["type"] == "ValueError"
    assert result["run"]["runner_error"]["message"] == "boom"
    assert "Traceback" in result["run"]["runner_error"]["traceback"]


def test_missing_live_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--tickets", "T001", "--out", str(tmp_path)])
    assert rc != 0
    assert list(tmp_path.iterdir()) == []


def test_subset_writes_exactly_n_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--live", "--subset", "3", "--out", str(tmp_path)])
    assert rc == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 3


def test_tickets_flag_writes_exactly_named_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--live", "--tickets", "T001,T002", "--out", str(tmp_path)])
    assert rc == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["T001.json", "T002.json"]


def test_subset_and_tickets_mutually_exclusive_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(
        ["--live", "--subset", "2", "--tickets", "T001", "--out", str(tmp_path)]
    )
    assert rc != 0
    assert list(tmp_path.iterdir()) == []


def test_usage_shim_accumulates_tokens(tmp_path, monkeypatch):
    calls = []

    def fake_llm(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        return {"error": "boom"}

    def fake_run_agent_loop(ticket_id):
        run_benchmark.orchestrator.call_llm_with_tools({}, [])
        run_benchmark.orchestrator.call_llm_with_tools({}, [])
        return _canned_state(ticket_id)

    monkeypatch.setattr(run_benchmark.orchestrator, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", fake_run_agent_loop)

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text())
    usage = result["usage"]
    assert usage["llm_call_count"] == 2
    assert usage["total_tokens_in"] == 10
    assert usage["total_tokens_out"] == 5
    assert usage["per_call"] == [{"in": 10, "out": 5}, {"in": 0, "out": 0}]


def test_usage_shim_handles_error_dict_without_raising(tmp_path, monkeypatch):
    def fake_llm(*args, **kwargs):
        return {"error": "nope"}

    def fake_run_agent_loop(ticket_id):
        run_benchmark.orchestrator.call_llm_with_tools({}, [])
        return _canned_state(ticket_id)

    monkeypatch.setattr(run_benchmark.orchestrator, "call_llm_with_tools", fake_llm)
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", fake_run_agent_loop)

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "T001.json").read_text())
    usage = result["usage"]
    assert usage["llm_call_count"] == 1
    assert usage["total_tokens_in"] == 0
    assert usage["total_tokens_out"] == 0
    assert usage["per_call"] == [{"in": 0, "out": 0}]


def test_no_stray_temp_files_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--live", "--tickets", "T001,T002", "--out", str(tmp_path)])
    assert rc == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["T001.json", "T002.json"]


def test_no_stray_temp_files_after_crash(tmp_path, monkeypatch):
    def _raise(ticket_id):
        raise ValueError("boom")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", _raise)
    rc = run_benchmark.main(["--live", "--tickets", "T001,T002", "--out", str(tmp_path)])
    assert rc == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["T001.json", "T002.json"]


def test_progress_line_reports_resolved_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid)
    )
    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "status=resolved" in captured.err
    assert "status=done" not in captured.err


def test_progress_line_reports_escalated_status(tmp_path, monkeypatch, capsys):
    def _escalated_state(ticket_id):
        state = _canned_state(ticket_id)
        state.status = "escalated"
        return state

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", _escalated_state)
    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "status=escalated" in captured.err
    assert "status=done" not in captured.err


def test_progress_line_reports_crashed_status(tmp_path, monkeypatch, capsys):
    def _raise(ticket_id):
        raise ValueError("boom")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", _raise)
    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "status=crashed" in captured.err
    assert "status=done" not in captured.err
