import json

import eval.report as report_mod


def _write_ticket_file(path, ticket_id, ticket, state, runner_error=None, started_at="2026-01-01T00:00:00Z"):
    raw = {
        "schema_version": 1,
        "ticket_id": ticket_id,
        "run": {"started_at": started_at, "wall_clock_seconds": 1.5, "runner_error": runner_error},
        "ticket": {
            "category": ticket["category"],
            "gold_root_cause": ticket["gold_root_cause"],
            "gold_runbook_id": ticket["gold_runbook_id"],
            "required_tools": ticket["required_tools"],
            "expected_behavior": ticket["expected_behavior"],
            "min_confidence_evidence_sources": ticket["min_confidence_evidence_sources"],
        },
        "state": state,
        "usage": {"llm_call_count": 3, "total_tokens_in": 1000, "total_tokens_out": 200, "per_call": []},
    }
    (path / f"{ticket_id}.json").write_text(json.dumps(raw), encoding="utf-8")


def _state(status, hypothesis, iteration=1, citations=None):
    return {
        "ticket_id": "T",
        "status": status,
        "hypothesis": hypothesis,
        "iteration": iteration,
        "trajectory": [
            {
                "iteration": 0,
                "tool_call": {"name": "query_logs", "arguments": {}},
                "observation": {"status": "ok"},
            },
        ],
        "pending_action_id": "abc",
        "citations": citations or [],
    }


def _fake_pytest_run_factory(status="PARTIAL"):
    class _Proc:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(*args, **kwargs):
        if status == "FAIL":
            return _Proc(1, "1 failed, 2 passed in 0.10s")
        # Default PARTIAL fixture mirrors the real suite: some tests skipped
        # (injection placeholder not built), nothing failed.
        return _Proc(0, "3 passed, 1 skipped in 0.10s")

    return fake_run


def _setup(tmp_path, monkeypatch, safety_status="PARTIAL"):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"

    tickets = [
        {
            "id": "T_SUCCESS",
            "ticket_text": "x",
            "category": "easy",
            "gold_root_cause": "db_connection_pool_exhaustion",
            "gold_runbook_id": "RB-DB-001.md",
            "required_tools": ["query_logs"],
            "min_confidence_evidence_sources": 1,
            "expected_behavior": "resolve_with_approval",
            "notes": "",
        },
        {
            "id": "T_KNOWN",
            "ticket_text": "x",
            "category": "ambiguous",
            "gold_root_cause": "deploy_bad_config",
            "gold_runbook_id": "RB-DEPLOY-001.md",
            "required_tools": ["query_logs"],
            "min_confidence_evidence_sources": 1,
            "expected_behavior": "escalate",
            "notes": "",
        },
        {
            "id": "T_STRICT_MISMATCH",
            "ticket_text": "x",
            "category": "multi_step",
            "gold_root_cause": "db_connection_pool_exhaustion",
            "gold_runbook_id": "RB-DB-001.md",
            "required_tools": ["query_logs"],
            "min_confidence_evidence_sources": 1,
            "expected_behavior": "resolve_with_approval",
            "notes": "",
        },
        {
            "id": "T_CRASHED",
            "ticket_text": "x",
            "category": "tool_heavy",
            "gold_root_cause": "network_timeout",
            "gold_runbook_id": "RB-NETWORK-001.md",
            "required_tools": ["query_logs"],
            "min_confidence_evidence_sources": 1,
            "expected_behavior": "resolve_with_approval",
            "notes": "",
        },
    ]
    tickets_by_id = {t["id"]: t for t in tickets}

    _write_ticket_file(
        raw_dir, "T_SUCCESS", tickets_by_id["T_SUCCESS"],
        _state("resolved", "db connection pool exhaustion caused the errors"),
    )
    _write_ticket_file(
        raw_dir, "T_KNOWN", tickets_by_id["T_KNOWN"],
        _state("resolved", "something unrelated"),
    )
    _write_ticket_file(
        raw_dir, "T_STRICT_MISMATCH", tickets_by_id["T_STRICT_MISMATCH"],
        _state("resolved", "totally unrelated wording with no gold tokens at all"),
    )
    _write_ticket_file(
        raw_dir, "T_CRASHED", tickets_by_id["T_CRASHED"],
        None,
        runner_error={"type": "RuntimeError", "message": "boom", "traceback": "..."},
    )

    monkeypatch.setattr(report_mod.subprocess, "run", _fake_pytest_run_factory(safety_status))
    monkeypatch.setattr(report_mod, "find_known_issue", lambda ticket_id: _known_issue() if ticket_id == "T_KNOWN" else None)

    return raw_dir, out_dir, tickets_by_id


class _FakeKnownIssue:
    ticket_ids = ("T_KNOWN",)
    cause = "test-fixture known cause"
    detail = "detail"
    evidence = "evidence"
    expect_status = "escalated"


def _known_issue():
    return _FakeKnownIssue()


def test_build_report_and_render(tmp_path, monkeypatch):
    raw_dir, out_dir, tickets_by_id = _setup(tmp_path, monkeypatch)

    safety = report_mod.run_safety_gate()
    assert safety["status"] == "PARTIAL"

    report = report_mod.build_report(raw_dir, tickets_by_id, safety=safety)

    # Crashed ticket present in its own count, never silently dropped.
    assert report["crashed"]["n"] == 1
    assert report["crashed"]["tickets"][0]["ticket_id"] == "T_CRASHED"
    assert report["task_success"]["task_success_status_only"]["n_total"] == 4

    # Known-issue ticket bucketed correctly, with documented cause -- since
    # T_KNOWN's expected_behavior is "escalate" but state.status is
    # "resolved", it fails status-only and should land in known_issues IF the
    # fake KnownIssue's expect_status matches state.status ("resolved" !=
    # "escalated" in our fixture, so this exercises unexplained instead)...
    all_ids_in_known = {k["ticket_id"] for k in report["known_issues"]}
    all_ids_in_unexplained = set(report["unexplained_failures"])
    assert "T_KNOWN" in all_ids_in_known or "T_KNOWN" in all_ids_in_unexplained
    if "T_KNOWN" in all_ids_in_known:
        found = next(k for k in report["known_issues"] if k["ticket_id"] == "T_KNOWN")
        assert found["documented_cause"] == "test-fixture known cause"
        assert "T_KNOWN" not in all_ids_in_unexplained

    # Both success measures differ on the strict-lexical-mismatch fixture.
    strict_ticket = next(t for t in report["per_ticket"] if t["ticket_id"] == "T_STRICT_MISMATCH")
    assert strict_ticket["task_success_status_only"] is True
    assert strict_ticket["task_success_strict_lexical"] is False

    md = report_mod.render_markdown(report)

    # Safety block near the top, before efficiency.
    safety_idx = md.index("SAFETY GATE")
    efficiency_idx = md.index("## Efficiency")
    assert safety_idx < efficiency_idx
    assert "PARTIAL" in md.split("\n")[md[:safety_idx].count("\n") + 1] or "PARTIAL" in md

    # Generated summary reflects fixture data, not canned prose.
    assert "4" in report["summary"]  # n_total
    assert str(report["crashed"]["n"]) in report["summary"]

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    assert md_path.exists()
    assert json_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("provenance", "safety", "task_success", "category_breakdown", "known_issues", "crashed", "efficiency"):
        assert key in parsed


def test_safety_fail_status_and_exit(tmp_path, monkeypatch):
    raw_dir, out_dir, tickets_by_id = _setup(tmp_path, monkeypatch, safety_status="FAIL")
    safety = report_mod.run_safety_gate()
    assert safety["status"] == "FAIL"
    report = report_mod.build_report(raw_dir, tickets_by_id, safety=safety)
    md = report_mod.render_markdown(report)
    assert "FAIL" in md.split("SAFETY GATE:")[1][:20]


def test_main_writes_files_and_exit_code(tmp_path, monkeypatch):
    raw_dir, out_dir, tickets_by_id = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(report_mod, "load_tickets", lambda tickets_path=None: tickets_by_id)

    rc = report_mod.main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()


def test_stale_denorm_detection():
    live = {"category": "easy", "gold_root_cause": "x", "gold_runbook_id": "RB-1.md",
            "required_tools": ["a"], "expected_behavior": "resolve_with_approval",
            "min_confidence_evidence_sources": 1}
    raw = {"ticket": {"category": "hard", "gold_root_cause": "x", "gold_runbook_id": "RB-1.md",
                       "required_tools": ["a"], "expected_behavior": "resolve_with_approval",
                       "min_confidence_evidence_sources": 1}}
    mismatches = report_mod.check_stale_denorm(raw, live)
    assert any("category" in m for m in mismatches)


def test_rate_card_selection_unknown_provider():
    label, card = report_mod.select_rate_card("https://api.some-other-host.example/v1")
    assert card is None
    assert "unknown" in label.lower()


def test_rate_card_selection_groq():
    label, card = report_mod.select_rate_card("https://api.groq.com/openai/v1")
    assert card is not None
    assert "groq" in label.lower()


def test_safety_skip_not_folded_into_denominator(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "3 passed, 1 skipped in 0.10s"
        stderr = ""

    monkeypatch.setattr(report_mod.subprocess, "run", lambda *a, **k: _Proc())

    safety = report_mod.run_safety_gate()

    assert safety["status"] == "PARTIAL"
    assert safety["unauthorized_write_block_rate"] == "3/3 enforced tests passed (0 failed)"
    assert "3/4" not in safety["unauthorized_write_block_rate"]
    assert safety["skipped_note"] == "1 test skipped: injection placeholder (adversarial ticket set not built)"

    report = {
        "safety": safety,
        "summary": "x",
        "provenance": {"n_tickets_scored": 0, "earliest_started_at": None, "latest_started_at": None},
        "rate_card": {"provider": "n/a", "used": False},
        "task_success": {
            "task_success_status_only": {"n_success": 0, "n_total": 0, "rate": None},
            "task_success_strict_lexical": {"n_success": 0, "n_total": 0, "rate": None},
            "hypothesis_semantic": "PENDING (Session 9b)",
            "gap_explanation": "n/a",
        },
        "category_breakdown": {
            cat: {
                "n": 0,
                "task_success_status_only": {"n_success": 0, "rate": None},
                "task_success_strict_lexical": {"n_success": 0, "rate": None},
            }
            for cat in report_mod.CATEGORIES
        },
        "known_issues": [],
        "stale_known_issues": [],
        "unexplained_failures": [],
        "crashed": {"n": 0, "tickets": []},
        "efficiency": {
            field: {"mean": None, "p50": None, "p95": None, "n": 0}
            for field in (
                "llm_call_count", "tool_call_count", "total_tokens_in",
                "total_tokens_out", "wall_clock_seconds", "estimated_cost_usd",
            )
        },
        "tool_use": {
            "tool_selection_accuracy": {"mean": None, "n": 0},
            "unnecessary_tool_calls_total": 0,
            "parameter_validity": {"observed_failures": 0, "total_calls": 0, "measurable": False, "note": ""},
        },
        "state_tracking": {
            "state_consistency": {"n_true": 0, "n": 0, "rate": None},
            "write_gate_appended_correctly": {"n_true": 0, "n": 0, "rate": None, "note": ""},
        },
        "rag": {
            "retrieval_recall_at_3_observed": {"n_true": 0, "n": 0, "rate": None},
            "citation_presence_rate": {"n_true": 0, "n": 0, "rate": None},
            "retrieval_recall_at_3_corpus": None,
        },
        "stale_gold_warnings": [],
        "missing_gold_tickets": [],
        "per_ticket": [],
    }
    md = report_mod.render_markdown(report)
    assert "3/4" not in md
    assert "3/3 enforced tests passed (0 failed)" in md
    assert "1 test skipped: injection placeholder" in md
    assert "PARTIAL" in md


def test_safety_failed_beats_skipped_and_sets_fail(monkeypatch):
    class _Proc:
        returncode = 1
        stdout = "1 failed, 2 passed in 0.10s"
        stderr = ""

    monkeypatch.setattr(report_mod.subprocess, "run", lambda *a, **k: _Proc())

    safety = report_mod.run_safety_gate()

    assert safety["status"] == "FAIL"
    assert safety["unauthorized_write_block_rate"] == "2/3 enforced tests passed (1 failed)"


def test_rate_card_baseten_confirmed():
    label, card = report_mod.select_rate_card("https://inference.baseten.co/v1")
    assert card is not None
    assert card.input_per_mtok == 0.15
    assert card.output_per_mtok == 0.60
    assert "baseten" in label.lower()
    assert "2026-08-23" in label
