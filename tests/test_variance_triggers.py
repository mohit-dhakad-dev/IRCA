"""Tests for eval/variance_triggers.py: the paired A/B harness's arm/repeat/
directory layout and summary shape. run_benchmark.run_one is monkeypatched
so NO LLM call ever happens -- fully offline and deterministic.
"""

from __future__ import annotations

import json
import os

import pytest

import eval.run_benchmark as run_benchmark
import eval.variance_triggers as variance_triggers
from eval.variance_triggers import _ENV_VAR, ARMS, compare_arms, main, run_ab

TICKET_IDS = ["T004", "T014"]


def _make_fake_run_one(env_seen_by_call: list, statuses: dict):
    """Fake run_one(ticket, out_dir) that writes a minimal raw result file
    (mirroring run_benchmark.run_one's on-disk shape) instead of calling the
    real agent loop, and records the IRCA_MEMORY_TRIGGERS value visible at
    call time.

    `statuses` maps ticket_id -> status string used for every repeat of that
    ticket, so tests can construct a ticket that resolves 3/5 times by
    swapping the fake between calls if needed; the default fixture keeps it
    constant per ticket for simplicity.
    """

    def fake_run_one(ticket, out_dir):
        env_seen_by_call.append(os.environ.get(_ENV_VAR))
        ticket_id = ticket["id"]
        status = statuses.get(ticket_id, "resolved")
        state = {
            "ticket_id": ticket_id,
            "status": status,
            "hypothesis": None,
            "trajectory": [],
            "memory_nudge_issued": False,
            "memory_autoconsulted": False,
        }
        result = {
            "schema_version": 2,
            "ticket_id": ticket_id,
            "run": {"started_at": "x", "wall_clock_seconds": 0.0, "runner_error": None},
            "ticket": {"expected_behavior": "escalate" if status == "escalated" else "resolve"},
            "state": state,
            "pending_action": None,
            "usage": {},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ticket_id}.json").write_text(json.dumps(result), encoding="utf-8")
        return {"ticket_id": ticket_id, "crashed": False, "status": status, "wall_clock_seconds": 0.0}

    return fake_run_one


@pytest.fixture
def fake_tickets(tmp_path, monkeypatch):
    tickets = [
        {"id": "T004", "expected_behavior": "resolve"},
        {"id": "T014", "expected_behavior": "resolve"},
    ]
    tickets_path = tmp_path / "tickets.json"
    tickets_path.write_text(json.dumps(tickets), encoding="utf-8")
    monkeypatch.setattr(variance_triggers, "TICKETS_PATH", tickets_path)
    return tickets


def test_run_ab_directory_layout(tmp_path, monkeypatch, fake_tickets):
    env_seen = []
    monkeypatch.setattr(
        run_benchmark, "run_one", _make_fake_run_one(env_seen, {"T004": "resolved", "T014": "escalated"})
    )
    out_root = tmp_path / "variance_out"

    summary = run_ab(TICKET_IDS, repeats=2, out_root=out_root)

    for arm in ARMS:
        for rep in (1, 2):
            for tid in TICKET_IDS:
                path = out_root / arm / f"rep{rep}" / f"{tid}.json"
                assert path.exists(), path

    assert summary["repeats"] == 2
    assert set(summary["arms"].keys()) == set(ARMS)


def test_env_var_set_per_arm_and_restored(tmp_path, monkeypatch, fake_tickets):
    env_seen = []
    monkeypatch.setattr(run_benchmark, "run_one", _make_fake_run_one(env_seen, {}))
    monkeypatch.delenv(_ENV_VAR, raising=False)

    run_ab(TICKET_IDS, repeats=1, out_root=tmp_path / "out")

    # 2 tickets * 1 repeat = 2 calls per arm, 2 arms = 4 calls total.
    assert len(env_seen) == 4
    on_calls = env_seen[:2]
    off_calls = env_seen[2:]
    assert all(v is None for v in on_calls)
    assert all(v == "0" for v in off_calls)
    # Restored to the pre-run (unset) state afterward.
    assert _ENV_VAR not in os.environ


def test_env_var_restored_even_if_run_one_raises(tmp_path, monkeypatch, fake_tickets):
    monkeypatch.delenv(_ENV_VAR, raising=False)

    def raising_run_one(ticket, out_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_benchmark, "run_one", raising_run_one)

    with pytest.raises(RuntimeError):
        run_ab(TICKET_IDS, repeats=1, out_root=tmp_path / "out")

    assert _ENV_VAR not in os.environ


def test_env_var_restored_to_previous_value(tmp_path, monkeypatch, fake_tickets):
    env_seen = []
    monkeypatch.setattr(run_benchmark, "run_one", _make_fake_run_one(env_seen, {}))
    monkeypatch.setenv(_ENV_VAR, "some-prior-value")

    run_ab(TICKET_IDS, repeats=1, out_root=tmp_path / "out")

    assert os.environ.get(_ENV_VAR) == "some-prior-value"


def test_summary_reports_per_ticket_distributions_not_means(tmp_path, monkeypatch, fake_tickets):
    """A ticket resolving 3/5 times must show up as visible counts, not a
    single averaged float."""
    calls = {"T004": 0}

    def fake_run_one(ticket, out_dir):
        ticket_id = ticket["id"]
        if ticket_id == "T004":
            calls["T004"] += 1
            status = "resolved" if calls["T004"] <= 3 else "escalated"
        else:
            status = "resolved"
        state = {
            "ticket_id": ticket_id,
            "status": status,
            "hypothesis": None,
            "trajectory": [],
            "memory_nudge_issued": False,
            "memory_autoconsulted": False,
        }
        result = {
            "schema_version": 2,
            "ticket_id": ticket_id,
            "run": {"started_at": "x", "wall_clock_seconds": 0.0, "runner_error": None},
            "ticket": {"expected_behavior": "resolve"},
            "state": state,
            "pending_action": None,
            "usage": {},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ticket_id}.json").write_text(json.dumps(result), encoding="utf-8")
        return {"ticket_id": ticket_id, "crashed": False, "status": status, "wall_clock_seconds": 0.0}

    monkeypatch.setattr(run_benchmark, "run_one", fake_run_one)

    summary = run_ab(["T004"], repeats=5, out_root=tmp_path / "out")

    t004_on = summary["arms"]["triggers_on"]["per_ticket"]["T004"]
    assert t004_on["n_repeats"] == 5
    assert t004_on["task_success_status_only_count"] == 3
    assert t004_on["status_distribution"] == {"resolved": 3, "escalated": 2}
    assert len(t004_on["runs"]) == 5


def test_live_flag_required():
    assert main([]) == 2


def test_compare_arms_reports_k_of_n_no_pvalue(tmp_path, monkeypatch, fake_tickets):
    env_seen = []
    monkeypatch.setattr(
        run_benchmark, "run_one", _make_fake_run_one(env_seen, {"T004": "resolved", "T014": "escalated"})
    )
    summary = run_ab(TICKET_IDS, repeats=2, out_root=tmp_path / "out")

    comparison = compare_arms(summary)

    for arm in ARMS:
        arm_result = comparison["arms"][arm]
        assert arm_result["n_total_runs"] == 4  # 2 tickets * 2 repeats
        assert "of" in arm_result["memory_consultation"]
        assert "of" in arm_result["task_success"]
        assert "p_value" not in arm_result
        assert "significant" not in arm_result
