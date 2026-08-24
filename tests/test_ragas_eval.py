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
