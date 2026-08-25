import json
from types import SimpleNamespace

import pytest

import agent.approval as approval_module
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
    result = json.loads(out_path.read_text(encoding="utf-8"))

    assert result["schema_version"] == 2
    assert result["ticket_id"] == "T001"
    assert set(result.keys()) == {
        "schema_version",
        "ticket_id",
        "run",
        "ticket",
        "state",
        "pending_action",
        "usage",
    }
    assert result["pending_action"] is None
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

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
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

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
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

    result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
    usage = result["usage"]
    assert usage["llm_call_count"] == 1
    assert usage["total_tokens_in"] == 0
    assert usage["total_tokens_out"] == 0
    assert usage["per_call"] == [{"in": 0, "out": 0}]


def test_run_one_persists_pending_action_when_write_was_queued(tmp_path, monkeypatch):
    """Session 10 Step 2 follow-up: run_one must look up the PendingAction
    from agent.approval's in-memory store (still populated immediately after
    run_agent_loop returns) and persist its human-visible fields under the
    top-level "pending_action" key -- offline, no LLM call."""
    approval_module.clear_store()
    try:
        action = approval_module.create_pending_action(
            ticket_id="T001",
            proposed_root_cause="db_connection_pool_exhaustion",
            proposed_fix="Raise the pool max and add a queue-depth alert.",
            citation_doc_id="RB-DB-001.md",
        )

        def _stub_loop(tid):
            state = _canned_state(tid)
            state.pending_action_id = action.action_id
            return state

        monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", _stub_loop)

        rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(tmp_path)])
        assert rc == 0

        result = json.loads((tmp_path / "T001.json").read_text(encoding="utf-8"))
        assert result["pending_action"] == {
            "ticket_id": "T001",
            "action_id": action.action_id,
            "proposed_root_cause": "db_connection_pool_exhaustion",
            "proposed_fix": "Raise the pool max and add a queue-depth alert.",
            "citation_doc_id": "RB-DB-001.md",
        }
    finally:
        approval_module.clear_store()


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


def test_archives_previous_sweep_before_new_run(tmp_path, monkeypatch, capsys):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)

    stale_content = {"stale": True, "ticket_id": "T099"}
    (out_dir / "T099.json").write_text(json.dumps(stale_content), encoding="utf-8")
    (out_dir / ".gitkeep").write_text("", encoding="utf-8")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(out_dir)])
    assert rc == 0

    # out_dir now contains only the new sweep's file (plus .gitkeep, untouched).
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == [".gitkeep", "T001.json"]

    # exactly one archive subdirectory was created, holding the old file
    # with its original content intact.
    archive_dirs = list(archive_root.iterdir())
    assert len(archive_dirs) == 1
    archived_files = list(archive_dirs[0].iterdir())
    assert [p.name for p in archived_files] == ["T099.json"]
    assert json.loads(archived_files[0].read_text(encoding="utf-8")) == stale_content

    captured = capsys.readouterr()
    assert "Archived 1 file(s)" in captured.err
    assert str(archive_dirs[0]) in captured.err


def test_gitkeep_not_archived(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)
    (out_dir / ".gitkeep").write_text("", encoding="utf-8")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(out_dir)])
    assert rc == 0

    assert (out_dir / ".gitkeep").exists()
    # no stale json files existed, so no archive dir should have been made
    assert not archive_root.exists()


def test_no_archive_flag_leaves_stale_files_in_place(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)

    stale_content = {"stale": True, "ticket_id": "T002"}
    (out_dir / "T002.json").write_text(json.dumps(stale_content), encoding="utf-8")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(
        ["--live", "--no-archive", "--tickets", "T001,T002", "--out", str(out_dir)]
    )
    assert rc == 0

    # no archiving happened at all
    assert not archive_root.exists()
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["T001.json", "T002.json"]
    # the stale T002.json was overwritten by the new sweep, not left as-is
    result = json.loads((out_dir / "T002.json").read_text(encoding="utf-8"))
    assert result != stale_content
    assert result["ticket_id"] == "T002"


def test_empty_out_dir_no_archive_dir_and_no_stderr_line(tmp_path, monkeypatch, capsys):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(out_dir)])
    assert rc == 0

    assert not archive_root.exists()
    captured = capsys.readouterr()
    assert "Archived" not in captured.err


def test_live_guard_rejection_does_not_archive(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)

    stale_content = {"stale": True, "ticket_id": "T003"}
    (out_dir / "T003.json").write_text(json.dumps(stale_content), encoding="utf-8")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))
    rc = run_benchmark.main(["--tickets", "T001", "--out", str(out_dir)])
    assert rc != 0

    assert not archive_root.exists()
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["T003.json"]
    assert json.loads((out_dir / "T003.json").read_text(encoding="utf-8")) == stale_content


def test_timestamp_collision_creates_distinct_archive_dirs(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)
    monkeypatch.setattr(run_benchmark, "_utc_timestamp", lambda: "20260101T000000Z")

    first_stale = {"stale": True, "ticket_id": "T010"}
    (out_dir / "T010.json").write_text(json.dumps(first_stale), encoding="utf-8")

    monkeypatch.setattr(run_benchmark.orchestrator, "run_agent_loop", lambda tid: _canned_state(tid))

    rc1 = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(out_dir)])
    assert rc1 == 0

    # second sweep, forced to the same timestamp, with a new stale file.
    second_stale = {"stale": True, "ticket_id": "T011"}
    (out_dir / "T011.json").write_text(json.dumps(second_stale), encoding="utf-8")
    # the T001.json written by the first sweep is also "stale" from the
    # second sweep's point of view since it predates this run.
    rc2 = run_benchmark.main(["--live", "--tickets", "T002", "--out", str(out_dir)])
    assert rc2 == 0

    archive_dirs = sorted(p.name for p in archive_root.iterdir())
    assert len(archive_dirs) == 2
    assert archive_dirs[0] == "20260101T000000Z"
    assert archive_dirs[1] == "20260101T000000Z-2"

    # first archive dir's content is byte-for-byte intact -- not clobbered
    # by the second archive operation landing on the same timestamp.
    first_dir_files = {p.name: p.read_text(encoding="utf-8") for p in (archive_root / archive_dirs[0]).iterdir()}
    assert first_dir_files == {"T010.json": json.dumps(first_stale)}

    second_dir_files = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (archive_root / archive_dirs[1]).iterdir()}
    assert set(second_dir_files) == {"T001.json", "T011.json"}
    assert second_dir_files["T011.json"] == second_stale
    assert second_dir_files["T001.json"]["ticket_id"] == "T001"
    assert second_dir_files["T001.json"]["state"] == _canned_state("T001").model_dump()


def test_partial_move_failure_aborts_before_any_ticket_runs(tmp_path, monkeypatch, capsys):
    archive_root = tmp_path / "archive_root"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(run_benchmark, "ARCHIVE_ROOT", archive_root)

    (out_dir / "T020.json").write_text(json.dumps({"ticket_id": "T020"}), encoding="utf-8")
    (out_dir / "T021.json").write_text(json.dumps({"ticket_id": "T021"}), encoding="utf-8")

    real_move = run_benchmark.shutil.move
    call_count = {"n": 0}

    def flaky_move(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("disk full")
        return real_move(src, dst)

    monkeypatch.setattr(run_benchmark.shutil, "move", flaky_move)

    run_calls = []
    monkeypatch.setattr(
        run_benchmark.orchestrator,
        "run_agent_loop",
        lambda tid: (run_calls.append(tid), _canned_state(tid))[1],
    )

    rc = run_benchmark.main(["--live", "--tickets", "T001", "--out", str(out_dir)])

    assert rc != 0
    assert run_calls == []  # zero tickets executed

    captured = capsys.readouterr()
    assert "moved (1)" in captured.err
    assert "remaining in" in captured.err
    assert "T020.json" in captured.err or "T021.json" in captured.err
