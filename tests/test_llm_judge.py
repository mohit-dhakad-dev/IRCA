import json
from types import SimpleNamespace

import pytest

import eval.llm_judge as llm_judge


def _fake_response(semantically_correct: bool, reasoning: str = "because reasons"):
    args = json.dumps(
        {"semantically_correct": semantically_correct, "reasoning": reasoning}
    )
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=args))
    message = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_no_tool_call_response():
    message = SimpleNamespace(tool_calls=[])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_well_formed_tool_call_parses_verdict(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: _fake_response(True)
    )
    verdict, reasoning, error = llm_judge.judge_once(
        "ticket text", "easy", "hypothesis text", "gold_slug"
    )
    assert verdict is True
    assert reasoning == "because reasons"
    assert error is None


def test_error_response_yields_none_verdict_with_error(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: {"error": "rate limited"}
    )
    verdict, reasoning, error = llm_judge.judge_once(
        "ticket text", "easy", "hypothesis text", "gold_slug"
    )
    assert verdict is None
    assert error == "rate limited"


def test_no_tool_call_yields_none_verdict_with_error(monkeypatch):
    monkeypatch.setattr(
        llm_judge,
        "call_llm_with_tools",
        lambda *a, **k: _fake_no_tool_call_response(),
    )
    verdict, reasoning, error = llm_judge.judge_once(
        "ticket text", "easy", "hypothesis text", "gold_slug"
    )
    assert verdict is None
    assert error is not None


def test_failure_is_never_coerced_to_false(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: {"error": "boom"}
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", repeats=1)
    assert result["verdict"] is None
    assert result["verdicts"] == [None]
    assert result["error"] == "boom"


def test_majority_vote_true_true_false(monkeypatch):
    responses = [_fake_response(True), _fake_response(True), _fake_response(False)]
    calls = iter(responses)
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: next(calls)
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", repeats=3)
    assert result["verdict"] is True
    assert result["agreement"] == pytest.approx(2 / 3)


def test_no_majority_when_tied(monkeypatch):
    responses = [_fake_response(True), _fake_response(False)]
    calls = iter(responses)
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: next(calls)
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", repeats=2)
    assert result["verdict"] is None
    assert result["agreement"] == pytest.approx(0.5)


def test_semantic_correct_rate_over_judged_only(monkeypatch):
    class Args:
        only = "A,B,C,D"
        gap_set = False
        subset = None
        repeats = 1

    args = Args()

    responses = {
        "A": _fake_response(True),
        "B": _fake_response(True),
        "C": _fake_response(False),
        "D": {"error": "boom"},
    }

    monkeypatch.setattr(
        llm_judge,
        "load_tickets",
        lambda: {
            tid: {
                "id": tid,
                "ticket_text": f"text {tid}",
                "gold_root_cause": "slug",
                "category": "easy",
            }
            for tid in "ABCD"
        },
    )
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: f"hyp {tid}")
    monkeypatch.setattr(llm_judge, "load_raw_category", lambda tid: "easy")
    monkeypatch.setattr(
        llm_judge,
        "call_llm_with_tools",
        lambda *a, tid_map=responses, **k: None,
    )

    def fake_judge_once(ticket_text, category, hypothesis, gold_root_cause):
        tid = hypothesis.split()[-1]
        resp = responses[tid]
        return llm_judge._extract_verdict(resp)

    monkeypatch.setattr(llm_judge, "judge_once", fake_judge_once)

    report = llm_judge.run(args)
    assert report["summary"]["n_judged"] == 3
    assert report["summary"]["n_correct"] == 2
    assert report["summary"]["n_failed"] == 1
    assert report["summary"]["semantic_correct_rate"] == pytest.approx(2 / 3)


def test_prompt_contains_required_fields_and_is_blind_to_prior_metrics():
    prompt = llm_judge.build_prompt(
        ticket_text="Checkout API returns 500s",
        category="easy",
        hypothesis="DB pool exhaustion caused the 500s",
        gold_root_cause="db_connection_pool_exhaustion",
    )
    assert "DB pool exhaustion caused the 500s" in prompt
    assert "db_connection_pool_exhaustion" in prompt
    assert "easy" in prompt
    assert "Checkout API returns 500s" in prompt
    assert "strict_lexical" not in prompt
    assert "status_only" not in prompt
    assert "task_success" not in prompt


def test_resolve_hypothesis_auto_prefers_raw(monkeypatch):
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: "raw hyp")
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: "gap hyp")
    hypothesis, source = llm_judge.resolve_hypothesis("T001", "auto")
    assert hypothesis == "raw hyp"
    assert source == "raw"


def test_resolve_hypothesis_auto_falls_back_to_gap_set(monkeypatch):
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: None)
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: "gap hyp")
    hypothesis, source = llm_judge.resolve_hypothesis("T001", "auto")
    assert hypothesis == "gap hyp"
    assert source == "gap_set"


def test_resolve_hypothesis_auto_raises_when_neither_available(monkeypatch):
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: None)
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: None)
    with pytest.raises(ValueError) as excinfo:
        llm_judge.resolve_hypothesis("T999", "auto")
    msg = str(excinfo.value)
    assert "T999" in msg
    assert "raw" in msg
    assert "gap_set" in msg


def test_resolve_hypothesis_forced_raw_fails_loudly_without_fallback(monkeypatch):
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: None)
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: "gap hyp")
    with pytest.raises(ValueError) as excinfo:
        llm_judge.resolve_hypothesis("T001", "raw")
    assert "T001" in str(excinfo.value)


def test_resolve_hypothesis_forced_gap_set_fails_loudly_without_raw(monkeypatch):
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: "raw hyp")
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: None)
    with pytest.raises(ValueError) as excinfo:
        llm_judge.resolve_hypothesis("T001", "gap-set")
    assert "T001" in str(excinfo.value)


def test_run_records_hypothesis_source_per_ticket_and_counts(monkeypatch):
    class Args:
        only = "A,B"
        gap_set = False
        subset = None
        repeats = 1
        hypothesis_source = "auto"

    args = Args()

    monkeypatch.setattr(
        llm_judge,
        "load_tickets",
        lambda: {
            tid: {
                "id": tid,
                "ticket_text": f"text {tid}",
                "gold_root_cause": "slug",
                "category": "easy",
            }
            for tid in "AB"
        },
    )
    monkeypatch.setattr(llm_judge, "load_raw_category", lambda tid: "easy")

    raw_map = {"A": "raw hypothesis A"}
    gap_map = {"B": "gap set hypothesis B"}
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: raw_map.get(tid))
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: gap_map.get(tid))
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: _fake_response(True)
    )

    report = llm_judge.run(args)

    by_id = {t["ticket_id"]: t for t in report["per_ticket"]}
    assert by_id["A"]["hypothesis_source"] == "raw"
    assert by_id["A"]["hypothesis"] == "raw hypothesis A"
    assert by_id["B"]["hypothesis_source"] == "gap_set"
    assert by_id["B"]["hypothesis"] == "gap set hypothesis B"

    assert report["config"]["hypothesis_source"] == "auto"
    assert report["config"]["hypothesis_source_counts"] == {"raw": 1, "gap_set": 1}


def test_gap_set_selects_46_tickets():
    class Args:
        only = None
        gap_set = True
        subset = None
        repeats = 1

    ids = llm_judge.resolve_ticket_ids(Args())
    assert len(ids) == 46
    expected = set(llm_judge.load_gap_set_ids())
    assert set(ids) == expected
