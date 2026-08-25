import json
import os
import sys

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
            "hypothesis_semantic": {"status": "NOT COMPUTED", "reason": "hypothesis_semantic_v3.json not found"},
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


# --------------------------------------------------------------------------
# Session 9b: hypothesis_semantic / judge validation / RAGAS diagnostic
# --------------------------------------------------------------------------


def _write_json(path, name, data):
    (path / name).write_text(json.dumps(data), encoding="utf-8")


def _hypothesis_semantic_v3_fixture():
    return {
        "schema_version": 1,
        "provider": "test",
        "config": {"rubric_version": 3, "repeats": 3, "rubric_text": "shared criterion"},
        "summary": {
            "n_judged": 46,
            "n_correct": 41,
            "n_incorrect": 5,
            "n_failed": 0,
            "semantic_correct_rate": 41 / 46,
            "n_fail_semantic_only": 1,
            "n_fail_evidence_only": 3,
            "n_fail_both": 1,
        },
        "per_ticket": (
            [{"ticket_id": f"T{i:03d}", "verdict": True, "agreement": 1.0} for i in range(1, 42)]
            + [{"ticket_id": f"T{i:03d}", "verdict": False, "agreement": 0.667} for i in range(42, 47)]
        ),
    }


def _judge_agreement_fixture(kappa=0.5, raw_agreement=84.0, pabak=0.68, prevalence=0.8):
    return {
        "n_compared": 25,
        "n_skipped": {"human_unlabeled": 0, "judge_failed": 0, "not_in_both": 21},
        "contingency": {"both_true": 18, "both_false": 3, "human_true_judge_false": 2, "human_false_judge_true": 2},
        "raw_agreement": raw_agreement,
        "kappa": kappa,
        "pabak": pabak,
        "prevalence": prevalence,
        "kappa_note": "kappa=0.500, PABAK=0.680, prevalence=0.800.",
        "per_ticket": [],
    }


def test_compute_hypothesis_semantic_from_fixture(tmp_path):
    _write_json(tmp_path, "hypothesis_semantic_v3.json", _hypothesis_semantic_v3_fixture())

    result = report_mod.compute_hypothesis_semantic(tmp_path)

    assert result["n_correct"] == 41
    assert result["n_judged"] == 46
    assert result["n_failed"] == 0
    assert abs(result["rate"] - 41 / 46) < 1e-9
    assert result["rubric_version"] == 3
    assert result["repeats"] == 3
    assert result["failure_decomposition"] == {"semantic_only": 1, "evidence_only": 3, "both": 1}
    assert result["judge_stability"]["n_unanimous"] == 41
    assert result["judge_stability"]["n_split"] == 5


def test_compute_hypothesis_semantic_missing_file_not_computed(tmp_path):
    result = report_mod.compute_hypothesis_semantic(tmp_path)
    assert result["status"] == "NOT COMPUTED"
    assert "hypothesis_semantic_v3.json" in result["reason"]


def test_compute_hypothesis_semantic_malformed_file_raises(tmp_path):
    (tmp_path / "hypothesis_semantic_v3.json").write_text("{not valid json", encoding="utf-8")
    import pytest

    with pytest.raises(json.JSONDecodeError):
        report_mod.compute_hypothesis_semantic(tmp_path)


def test_task_success_status_only_and_strict_lexical_unchanged(tmp_path, monkeypatch):
    raw_dir, out_dir, tickets_by_id = _setup(tmp_path, monkeypatch)
    safety = report_mod.run_safety_gate()
    report = report_mod.build_report(raw_dir, tickets_by_id, safety=safety, results_dir=tmp_path)

    ts = report["task_success"]
    assert set(ts.keys()) == {
        "task_success_status_only",
        "task_success_strict_lexical",
        "hypothesis_semantic",
        "gap_explanation",
    }
    a = ts["task_success_status_only"]
    b = ts["task_success_strict_lexical"]
    assert a["n_success"] == 2
    assert a["n_total"] == 4
    assert b["n_success"] == 1
    assert b["n_total"] == 4
    # hypothesis_semantic is a computed dict/NOT-COMPUTED marker, never a bare string.
    assert ts["hypothesis_semantic"] != "PENDING (Session 9b)"
    assert isinstance(ts["hypothesis_semantic"], dict)


def test_missing_ragas_and_judge_files_render_not_computed(tmp_path, monkeypatch):
    raw_dir, out_dir, tickets_by_id = _setup(tmp_path, monkeypatch)
    safety = report_mod.run_safety_gate()
    empty_results_dir = tmp_path / "empty_results"
    empty_results_dir.mkdir()

    report = report_mod.build_report(raw_dir, tickets_by_id, safety=safety, results_dir=empty_results_dir)
    md = report_mod.render_markdown(report)

    assert "NOT COMPUTED" in md
    assert report["task_success"]["hypothesis_semantic"]["status"] == "NOT COMPUTED"
    assert report["judge_validation"]["status"] == "NOT COMPUTED"


def test_full_session_9b_fixture_renders_headline_and_ragas_sections(tmp_path, monkeypatch):
    raw_dir, out_dir, tickets_by_id = _setup(tmp_path, monkeypatch)
    safety = report_mod.run_safety_gate()

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_json(results_dir, "hypothesis_semantic_v3.json", _hypothesis_semantic_v3_fixture())
    _write_json(results_dir, "judge_agreement_report.json", _judge_agreement_fixture())
    _write_json(
        results_dir,
        "hypothesis_semantic.v1.json",
        {
            "schema_version": 1,
            "provider": "test",
            "config": {"rubric_version": 1, "repeats": 3},
            "summary": {"n_judged": 46, "n_correct": 39, "n_incorrect": 7, "n_failed": 0, "semantic_correct_rate": 39 / 46},
            "per_ticket": [],
        },
    )
    _write_json(results_dir, "judge_agreement_report.v1.json", _judge_agreement_fixture(kappa=0.066, raw_agreement=64.0, pabak=0.28, prevalence=0.76))

    gap46 = {
        "schema_version": 1,
        "provider": "test",
        "config": {},
        "metrics": {
            "faithfulness": {"mean": 0.698, "n_scored": 46, "n_total": 46, "n_nan": 0, "complete": True},
            "answer_relevancy": {"mean": 0.472, "n_scored": 46, "n_total": 46, "n_nan": 0, "complete": True},
        },
        "usage": {},
        "per_ticket": [
            {"ticket_id": f"T{i:03d}", "category": "easy", "faithfulness": 0.8, "answer_relevancy": 0.47}
            for i in range(1, 47)
        ],
    }
    _write_json(results_dir, "ragas_scores_gap46.json", gap46)

    def _subset(n, precision=True, recall=None):
        metrics = {
            "faithfulness": {"mean": 0.8, "n_scored": n, "n_total": n, "n_nan": 0, "complete": True},
            "answer_relevancy": {"mean": 0.5, "n_scored": n, "n_total": n, "n_nan": 0, "complete": True},
        }
        if precision:
            metrics["llm_context_precision_with_reference"] = {"mean": 0.99999999998, "n_scored": n, "n_total": n, "n_nan": 0, "complete": True}
        if recall == "ok":
            metrics["context_recall"] = {"mean": 1.0, "n_scored": n, "n_total": n, "n_nan": 0, "complete": True}
        elif recall == "nan":
            metrics["llm_context_recall"] = {"mean": None, "n_scored": 0, "n_total": n, "n_nan": n, "complete": False}
        return {"schema_version": 1, "provider": "test", "config": {}, "metrics": metrics, "usage": {}, "per_ticket": []}

    _write_json(results_dir, "ragas_scores_subset8.json", _subset(8, recall="ok"))
    _write_json(results_dir, "ragas_scores_subset3.json", _subset(3, recall="nan"))
    _write_json(results_dir, "ragas_scores_subset3_mw16.json", _subset(3, recall="nan"))
    _write_json(
        results_dir,
        "ragas_scores_mood_normalized.json",
        {
            "schema_version": 1,
            "provider": "test",
            "config": {},
            "metrics": {
                "faithfulness": {"mean": 0.85, "n_scored": 10, "n_total": 10, "n_nan": 0, "complete": True},
                "answer_relevancy": {"mean": 0.45, "n_scored": 10, "n_total": 10, "n_nan": 0, "complete": True},
            },
            "usage": {},
            "per_ticket": [
                {"ticket_id": f"T{i:03d}", "category": "easy", "faithfulness": 0.85, "answer_relevancy": 0.45}
                for i in range(1, 11)
            ],
        },
    )

    report = report_mod.build_report(raw_dir, tickets_by_id, safety=safety, results_dir=results_dir)
    md = report_mod.render_markdown(report)

    # Headline dict, no "PENDING" literal.
    assert "PENDING" not in md
    hs = report["task_success"]["hypothesis_semantic"]
    assert hs["n_correct"] == 41 and hs["n_judged"] == 46

    # RAGAS diagnostic heading present.
    assert "## RAGAS Metrics" in md

    # Structural pin: none of the four RAGAS metric names appear inside the
    # ## Task Success section, only in the diagnostic section further down.
    task_success_block = md.split("## Task Success")[1].split("## ")[0]
    for name in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
        assert name not in task_success_block, f"{name} leaked into ## Task Success"

    # Judge validation section renders kappa/pabak/caveat.
    jv_block = md.split("## Judge Validation")[1].split("## ")[0]
    assert "kappa" in jv_block.lower()
    assert "pabak" in jv_block.lower()
    assert "oversample" in jv_block.lower()

    # Rubric-history note present with both kappas.
    assert "Rubric history" in md
    assert "0.066" in md or "kappa 0.066" in md.lower()


def test_safety_gate_default_python_is_absolute(monkeypatch):
    # sys.executable is present in a normal pytest run -- the default must
    # be exactly that, and it must be absolute.
    default = report_mod._default_venv_python()
    assert os.path.isabs(default)
    assert default == report_mod.sys.executable


def test_safety_gate_default_python_falls_back_to_absolute_platform_path(monkeypatch):
    # If sys.executable is empty (embedded interpreter edge case), fall back
    # to a platform-specific absolute path built from REPO_ROOT.
    monkeypatch.setattr(report_mod.sys, "executable", "", raising=False)

    monkeypatch.setattr(report_mod.os, "name", "nt", raising=False)
    default = report_mod._default_venv_python()
    assert os.path.isabs(default)
    assert default == str(report_mod.REPO_ROOT / "venv" / "Scripts" / "python.exe")

    monkeypatch.setattr(report_mod.os, "name", "posix", raising=False)
    monkeypatch.setattr(report_mod.sys, "platform", "linux", raising=False)
    default = report_mod._default_venv_python()
    assert os.path.isabs(default)
    assert default == str(report_mod.REPO_ROOT / "venv" / "bin" / "python")


def test_run_safety_gate_resolves_default_python_and_runs():
    # No python arg -- should use sys.executable (absolute), and the
    # subprocess should actually run (never UNKNOWN due to a bad path).
    safety = report_mod.run_safety_gate()
    assert safety["status"] != "UNKNOWN"


def test_run_safety_gate_resolves_relative_python_against_repo_root(tmp_path, monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        raise OSError("boom - never actually invoked in this test")

    monkeypatch.setattr(report_mod.subprocess, "run", fake_run)
    report_mod.run_safety_gate(python="venv/Scripts/python.exe")
    assert os.path.isabs(calls["cmd"][0])
    assert calls["cmd"][0] == str(report_mod.REPO_ROOT / "venv/Scripts/python.exe")


def test_run_safety_gate_passes_through_absolute_python_unchanged(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        raise OSError("boom - never actually invoked in this test")

    monkeypatch.setattr(report_mod.subprocess, "run", fake_run)
    abs_python = str(report_mod.REPO_ROOT / "some" / "abs" / "python.exe")
    report_mod.run_safety_gate(python=abs_python)
    assert calls["cmd"][0] == abs_python
