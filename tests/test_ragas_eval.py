"""Tests for eval/ragas_eval.py's pure helpers, run under the PROJECT venv
where ragas is NOT installed. run_evaluation() itself (the only function
that imports ragas) is exercised only under venv-ragas, manually -- see the
module docstring in eval/ragas_eval.py."""

import json
import math

import eval.ragas_eval as ragas_eval


def test_import_without_ragas():
    # If this module imported ragas at top level, this import would already
    # have failed under the project venv (ragas is not installed there).
    assert hasattr(ragas_eval, "run_evaluation")
    assert hasattr(ragas_eval, "load_inputs")


def test_summarize_metric_with_nan():
    values = [0.5, float("nan"), 0.9]
    summary = ragas_eval.summarize_metric(values)
    assert summary["n_scored"] == 2
    assert summary["n_total"] == 3
    assert summary["n_nan"] == 1
    assert summary["complete"] is False
    assert math.isclose(summary["mean"], (0.5 + 0.9) / 2)


def test_summarize_metric_all_nan():
    values = [float("nan"), None, float("nan")]
    summary = ragas_eval.summarize_metric(values)
    assert summary["mean"] is None
    assert summary["n_scored"] == 0
    assert summary["n_total"] == 3
    assert summary["n_nan"] == 3
    assert summary["complete"] is False


def test_summarize_metric_all_present():
    values = [0.1, 0.2, 0.3]
    summary = ragas_eval.summarize_metric(values)
    assert summary["n_scored"] == 3
    assert summary["n_nan"] == 0
    assert summary["complete"] is True
    assert math.isclose(summary["mean"], 0.2)


def test_dry_run_estimator_call_count_for_5_contexts():
    row = {
        "question": "why did it break",
        "answer": "the pool was exhausted",
        "ground_truth": "db connection pool exhaustion",
        "contexts": ["a", "b", "c", "d", "e"],
    }
    calls = ragas_eval.estimate_calls_for_row(row)
    total = sum(calls.values())
    assert total == 2 + 1 + 1 + 5
    assert calls["faithfulness"] == 2
    assert calls["answer_relevancy"] == 1
    assert calls["context_recall"] == 1
    assert calls["llm_context_precision_with_reference"] == 5


def test_gap_set_filtering_selects_46(tmp_path):
    gap_set_ids = ragas_eval.load_gap_set_ids()
    assert len(gap_set_ids) == 46

    rows = [{"ticket_id": f"T{i:03d}"} for i in range(1, 61)]
    filtered = ragas_eval.filter_rows(rows, gap_set_ids=gap_set_ids)
    assert len(filtered) == 46
    assert {r["ticket_id"] for r in filtered} == gap_set_ids


def test_summary_text_flags_incomplete():
    summary = ragas_eval.summarize_metric([0.5, float("nan")])
    text = ragas_eval.metric_summary_text("faithfulness", summary)
    assert "1/2" in text
    assert "INCOMPLETE" in text


def test_metric_names_pin_corrected_context_recall_key():
    # LLMContextRecall.name is "context_recall" (verified against ragas
    # 0.4.3 under venv-ragas), NOT "llm_context_recall" despite the class
    # name -- this pins the fix for that lookup-key bug.
    assert ragas_eval.METRIC_NAMES == [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]


def test_validate_metric_keys_passes_when_all_present():
    available = {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9,
        "llm_context_precision_with_reference": 0.8,
        "context_recall": 1.0,
    }
    # Should not raise.
    ragas_eval.validate_metric_keys(available, ragas_eval.METRIC_NAMES)


def test_validate_metric_keys_raises_naming_missing_key():
    available = {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9,
        "llm_context_precision_with_reference": 0.8,
        # "context_recall" missing, simulating the bug this guard exists to
        # catch immediately instead of silently reporting mean=None.
    }
    try:
        ragas_eval.validate_metric_keys(available, ragas_eval.METRIC_NAMES)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "context_recall" in str(exc)


def test_validate_pairing_passes_for_correct_pairing():
    pairs = [
        ("faithfulness", "faithfulness"),
        ("answer_relevancy", "answer_relevancy"),
        ("llm_context_precision_with_reference", "llm_context_precision_with_reference"),
        ("context_recall", "context_recall"),
    ]
    # Should not raise.
    ragas_eval.validate_pairing(pairs, ragas_eval.METRIC_NAMES)


def test_validate_pairing_raises_when_keys_dont_cover_metric_names():
    pairs = [
        ("faithfulness", "faithfulness"),
        ("answer_relevancy", "answer_relevancy"),
        ("llm_context_precision_with_reference", "llm_context_precision_with_reference"),
        # "context_recall" missing entirely.
    ]
    try:
        ragas_eval.validate_pairing(pairs, ragas_eval.METRIC_NAMES)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "context_recall" in str(exc)


def test_parse_metrics_arg_selects_requested_subset_in_canonical_order():
    selected = ragas_eval.parse_metrics_arg("answer_relevancy,faithfulness")
    # Canonical METRIC_NAMES order, not input order.
    assert selected == ["faithfulness", "answer_relevancy"]


def test_parse_metrics_arg_default_is_all_four():
    assert ragas_eval.parse_metrics_arg(None) == ragas_eval.METRIC_NAMES
    assert ragas_eval.parse_metrics_arg("") == ragas_eval.METRIC_NAMES


def test_parse_metrics_arg_raises_on_unknown_key():
    try:
        ragas_eval.parse_metrics_arg("faithfulness,not_a_real_metric")
        assert False, "expected ValueError"
    except ValueError as exc:
        msg = str(exc)
        assert "not_a_real_metric" in msg
        for name in ragas_eval.METRIC_NAMES:
            assert name in msg


def test_dropped_metrics_is_complement_of_selection():
    selected = ["faithfulness", "answer_relevancy"]
    dropped = ragas_eval.dropped_metrics(selected)
    assert dropped == ["llm_context_precision_with_reference", "context_recall"]
    assert set(dropped) | set(selected) == set(ragas_eval.METRIC_NAMES)
    assert set(dropped) & set(selected) == set()


def test_dry_run_call_projection_respects_metric_selection():
    row = {
        "question": "why did it break",
        "answer": "the pool was exhausted",
        "ground_truth": "db connection pool exhaustion",
        "contexts": ["a", "b", "c", "d", "e"],
    }
    selected = ragas_eval.parse_metrics_arg("faithfulness,answer_relevancy")
    estimate = ragas_eval.estimate_dry_run([row], selected)
    assert estimate["total_calls"] == 2 + 1
    assert set(estimate["per_metric_calls"]) == {"faithfulness", "answer_relevancy"}

    default_estimate = ragas_eval.estimate_dry_run([row])
    assert default_estimate["total_calls"] == 9
    assert set(default_estimate["per_metric_calls"]) == set(ragas_eval.METRIC_NAMES)


def test_default_checkpoint_path_derives_from_out_stem():
    path = ragas_eval.default_checkpoint_path("eval/results/ragas_scores.json")
    assert path.name == ".ragas_checkpoint_ragas_scores.jsonl"
    assert path.parent == ragas_eval.REPO_ROOT / "eval" / "results"


def _row(ticket_id, **overrides):
    row = {
        "ticket_id": ticket_id,
        "category": "database",
        "faithfulness": 0.5,
        "answer_relevancy": 0.6,
        "llm_context_precision_with_reference": 0.7,
        "context_recall": 0.8,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "llm_calls": 5},
    }
    row.update(overrides)
    return row


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    row1 = _row("T001")
    row2 = _row("T002")
    ragas_eval.append_checkpoint_row(path, row1)
    ragas_eval.append_checkpoint_row(path, row2)

    loaded = ragas_eval.read_checkpoint(path)
    assert loaded == [row1, row2]


def test_read_checkpoint_missing_file_returns_empty(tmp_path):
    assert ragas_eval.read_checkpoint(tmp_path / "does_not_exist.jsonl") == []


def test_resume_with_3_of_5_done_returns_other_2_in_order(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    for tid in ["T003", "T001", "T005"]:
        ragas_eval.append_checkpoint_row(path, _row(tid))

    existing = ragas_eval.read_checkpoint(path)
    done_ids = ragas_eval.checkpointed_ticket_ids(existing)
    all_ids = ["T001", "T002", "T003", "T004", "T005"]
    remaining = ragas_eval.remaining_ticket_ids(all_ids, done_ids)
    assert remaining == ["T002", "T004"]


def test_merge_produces_correct_aggregate_and_nan_accounting():
    rows = [
        _row("T001", faithfulness=0.5),
        _row("T002", faithfulness=float("nan")),
        _row("T003", faithfulness=0.9),
    ]
    merged = ragas_eval.merge_checkpoint_rows(rows, ragas_eval.METRIC_NAMES)
    faith = merged["metrics"]["faithfulness"]
    assert faith["n_scored"] == 2
    assert faith["n_total"] == 3
    assert math.isclose(faith["mean"], (0.5 + 0.9) / 2)
    assert merged["usage"]["prompt_tokens"] == 300
    assert merged["usage"]["completion_tokens"] == 150
    assert merged["usage"]["llm_calls"] == 15
    assert [r["ticket_id"] for r in merged["per_ticket"]] == ["T001", "T002", "T003"]


def test_malformed_checkpoint_line_raises_naming_ticket_and_path(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    good = _row("T001")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(good) + "\n")
        bad = dict(good)
        bad["ticket_id"] = "T002"
        del bad["usage"]
        f.write(json.dumps(bad) + "\n")

    try:
        ragas_eval.read_checkpoint(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        msg = str(exc)
        assert "T002" in msg
        assert str(path) in msg


def test_malformed_checkpoint_invalid_json_raises_naming_line_and_path(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json\n")

    try:
        ragas_eval.read_checkpoint(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        msg = str(exc)
        assert str(path) in msg
        assert "line 1" in msg


def test_no_resume_truncates_existing_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    ragas_eval.append_checkpoint_row(path, _row("T001"))
    assert ragas_eval.read_checkpoint(path) != []

    ragas_eval.truncate_checkpoint(path)
    assert ragas_eval.read_checkpoint(path) == []


def test_validate_pairing_raises_on_duplicate_canonical_key():
    pairs = [
        ("faithfulness", "faithfulness"),
        ("faithfulness", "answer_relevancy"),  # duplicate canonical key
        ("llm_context_precision_with_reference", "llm_context_precision_with_reference"),
        ("context_recall", "context_recall"),
    ]
    try:
        ragas_eval.validate_pairing(pairs, ragas_eval.METRIC_NAMES)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "faithfulness" in str(exc)
        assert "Duplicate" in str(exc)
