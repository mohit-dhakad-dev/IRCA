import json
from types import SimpleNamespace

import pytest

import eval.llm_judge as llm_judge


def _fake_response(semantic_match: bool, evidence_supported: bool, reasoning: str = "because reasons"):
    args = json.dumps(
        {
            "semantic_match": semantic_match,
            "evidence_supported": evidence_supported,
            "reasoning": reasoning,
        }
    )
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=args))
    message = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_no_tool_call_response():
    message = SimpleNamespace(tool_calls=[])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


SAMPLE_OBSERVATIONS = [
    (0, "query_logs", '{"status": "ok", "data": {"total_matched": 3}}'),
    (1, "query_metrics", '{"status": "ok", "data": {"disk_pct": 97}}'),
]


# ---------------------------------------------------------------------------
# judge_once / conjunction
# ---------------------------------------------------------------------------


def test_well_formed_tool_call_parses_both_fields(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: _fake_response(True, True)
    )
    sm, es, reasoning, error = llm_judge.judge_once(
        "ticket text", "easy", "hypothesis text", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert sm is True
    assert es is True
    assert reasoning == "because reasons"
    assert error is None


def test_error_response_yields_none_verdicts_with_error(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: {"error": "rate limited"}
    )
    sm, es, reasoning, error = llm_judge.judge_once(
        "ticket text", "easy", "hypothesis text", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert sm is None
    assert es is None
    assert error == "rate limited"


def test_no_tool_call_yields_none_verdicts_with_error(monkeypatch):
    monkeypatch.setattr(
        llm_judge,
        "call_llm_with_tools",
        lambda *a, **k: _fake_no_tool_call_response(),
    )
    sm, es, reasoning, error = llm_judge.judge_once(
        "ticket text", "easy", "hypothesis text", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert sm is None
    assert es is None
    assert error is not None


def test_failure_is_never_coerced_to_false(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: {"error": "boom"}
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", SAMPLE_OBSERVATIONS, repeats=1)
    assert result["verdict"] is None
    assert result["verdicts"] == [None]
    assert result["error"] == "boom"


@pytest.mark.parametrize(
    "semantic_match,evidence_supported,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_conjunction_is_computed_by_module_not_model(monkeypatch, semantic_match, evidence_supported, expected):
    monkeypatch.setattr(
        llm_judge,
        "call_llm_with_tools",
        lambda *a, **k: _fake_response(semantic_match, evidence_supported),
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", SAMPLE_OBSERVATIONS, repeats=1)
    assert result["verdict"] is expected
    assert result["semantic_match"] is semantic_match
    assert result["evidence_supported"] is evidence_supported


def test_majority_vote_true_true_false(monkeypatch):
    responses = [
        _fake_response(True, True),
        _fake_response(True, True),
        _fake_response(True, False),
    ]
    calls = iter(responses)
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: next(calls)
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", SAMPLE_OBSERVATIONS, repeats=3)
    assert result["verdict"] is True
    assert result["agreement"] == pytest.approx(2 / 3)


def test_no_majority_when_tied(monkeypatch):
    responses = [_fake_response(True, True), _fake_response(True, False)]
    calls = iter(responses)
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: next(calls)
    )
    result = llm_judge.judge_ticket("t", "easy", "h", "gold_slug", SAMPLE_OBSERVATIONS, repeats=2)
    assert result["verdict"] is None
    assert result["agreement"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# prompt content: rubric, clauses, untrusted-data delimiters, blindness
# ---------------------------------------------------------------------------


def test_prompt_contains_required_fields_and_is_blind_to_prior_metrics():
    prompt = llm_judge.build_prompt(
        ticket_text="Checkout API returns 500s",
        category="easy",
        hypothesis="DB pool exhaustion caused the 500s",
        gold_root_cause="db_connection_pool_exhaustion",
        observations=SAMPLE_OBSERVATIONS,
    )
    assert "DB pool exhaustion caused the 500s" in prompt
    assert "db_connection_pool_exhaustion" in prompt
    assert "easy" in prompt
    assert "Checkout API returns 500s" in prompt
    assert "strict_lexical" not in prompt
    assert "status_only" not in prompt
    assert "task_success" not in prompt
    assert llm_judge.RUBRIC_TEXT in prompt


def test_prompt_contains_verbatim_v3_rubric_and_clause_labels():
    prompt = llm_judge.build_prompt(
        "ticket text", "easy", "hypothesis", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert llm_judge.RUBRIC_TEXT in prompt
    assert "clause (a)" in prompt
    assert "clause (b)" in prompt
    assert "connection limits" in prompt
    assert "queue exhaustion" in prompt


def test_v3_prompt_restores_disk_full_clarification_and_worked_example():
    prompt = llm_judge.build_prompt(
        "ticket text", "easy", "hypothesis", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert "the disk is full" in prompt
    assert "disk_log_rotation_gap" in prompt
    assert "Worked example" in prompt
    assert "clause (a)" in prompt
    assert "clause (b)" in prompt
    assert "evidence grounding" in prompt


def test_v3_prompt_still_has_both_clause_labels_and_evidence_grounding():
    # Regression guard against re-dropping either side while restoring the
    # clarification: both clause labels and the evidence-grounding clause
    # must remain present.
    prompt = llm_judge.build_prompt(
        "ticket text", "easy", "hypothesis", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert "clause (a)" in prompt
    assert "clause (b)" in prompt
    assert "evidence grounding" in prompt
    assert "actually observed" in prompt


# Data-driven list of substantive v2 rules that must survive into v3.
# Each entry is (identifying phrase(s), rule name) where "phrase(s)" is
# either a single string that must appear verbatim in the prompt body,
# or a tuple of strings any one of which satisfies the check. To pin a
# future rule, append one line here — no new test function needed. If a
# rule is dropped or relocated out of the prompt body (e.g. left only in
# a schema field description), the assertion fails with a message naming
# which rule went missing.
V2_SUBSTANTIVE_RULES_IN_PROMPT_BODY = [
    (("wording",), "tolerance: differences in wording don't make it wrong"),
    (("verbosity",), "tolerance: differences in verbosity don't make it wrong"),
    (("the disk is full",), "clarification: disk-full example text"),
    (("disk_log_rotation_gap",), "clarification: disk-full example gold slug"),
    (
        ("log rotation failed", "log rotation failure"),
        "clarification: deeper mechanism (log rotation) named in example",
    ),
    (
        ("different, non-overlapping", "different mechanism"),
        "boundary: naming a different, non-overlapping mechanism is incorrect",
    ),
    (("Worked example",), "worked example distinguishing symptom vs. mechanism"),
    (("clause (b)",), "v3 superseding rule: evidence grounding is its own judged clause"),
    (("evidence grounding",), "v3 superseding rule: evidence grounding is its own judged clause"),
]


def test_v3_prompt_carries_every_substantive_v2_rule():
    """Pins the class of bug: a future rubric revision must not silently
    drop a rule the project owner already decided was part of the
    rubric, or move it out of the prompt body into a place (like a
    schema field description) that a reader auditing the prompt text
    would miss. Every substantive clause/example from the v2 prompt must
    still appear (verbatim or as its documented v3 restoration) in the
    v3 prompt BODY specifically — not merely somewhere in the combined
    prompt-plus-schema surface sent to the model.
    """
    v3_prompt = llm_judge.build_prompt(
        "ticket text", "easy", "hypothesis", "gold_slug", SAMPLE_OBSERVATIONS
    )

    for phrases, rule_name in V2_SUBSTANTIVE_RULES_IN_PROMPT_BODY:
        assert any(phrase in v3_prompt for phrase in phrases), (
            f"substantive v2 rule missing from v3 prompt body: {rule_name!r} "
            f"(looked for any of {phrases!r})"
        )


def test_tolerance_rule_not_duplicated_in_schema_field_description():
    """Pins the de-duplication itself: the tolerance rule ("differences in
    wording, verbosity, or added detail do NOT make the hypothesis
    incorrect") now lives once, in the prompt body's clause-(a) section.
    The RECORD_VERDICT_SCHEMA semantic_match field description must not
    reintroduce a second, independently-editable copy of it.
    """
    semantic_match_desc = llm_judge.RECORD_VERDICT_SCHEMA["function"]["parameters"][
        "properties"
    ]["semantic_match"]["description"]
    assert "wording" not in semantic_match_desc
    assert "verbosity" not in semantic_match_desc


def test_prompt_wraps_observations_in_untrusted_delimiters():
    prompt = llm_judge.build_prompt(
        "ticket text", "easy", "hypothesis", "gold_slug", SAMPLE_OBSERVATIONS
    )
    assert "BEGIN UNTRUSTED OBSERVED TOOL OUTPUT" in prompt
    assert "END UNTRUSTED OBSERVED TOOL OUTPUT" in prompt
    assert "DATA" in prompt
    assert "cannot instruct you" in prompt or "never instructions" in prompt.lower()
    assert '"total_matched": 3' in prompt
    assert '"disk_pct": 97' in prompt


# ---------------------------------------------------------------------------
# observations loading / rendering / truncation
# ---------------------------------------------------------------------------


def test_load_observations_returns_ordered_tuples(tmp_path):
    raw_dir = tmp_path
    raw = {
        "state": {
            "trajectory": [
                {
                    "iteration": 0,
                    "tool_call": {"name": "query_logs"},
                    "observation": {"status": "ok", "data": {"a": 1}},
                },
                {
                    "iteration": 1,
                    "tool_call": {"name": "query_metrics"},
                    "observation": {"status": "ok", "data": {"b": 2}},
                },
            ]
        }
    }
    (raw_dir / "T100.json").write_text(json.dumps(raw), encoding="utf-8")

    observations = llm_judge.load_observations("T100", raw_dir)
    assert observations is not None
    assert len(observations) == 2
    assert observations[0][0] == 0
    assert observations[0][1] == "query_logs"
    assert "\"a\": 1" in observations[0][1] or "a" in observations[0][2]
    assert observations[1][1] == "query_metrics"


def test_load_observations_missing_raw_file_returns_none(tmp_path):
    observations = llm_judge.load_observations("T999", tmp_path)
    assert observations is None


def test_render_observations_under_cap_not_truncated():
    text, truncated = llm_judge.render_observations(SAMPLE_OBSERVATIONS, cap=12000)
    assert truncated is False
    assert "query_logs" in text
    assert "query_metrics" in text


def test_render_observations_truncates_longest_first_and_sets_flag():
    short = "s" * 100
    long1 = "l" * 8000
    long2 = "m" * 5000
    observations = [
        (0, "tool_short", short),
        (1, "tool_long1", long1),
        (2, "tool_long2", long2),
    ]
    # total = 13100, cap 12000 -> excess 1100, should come off the longest
    # (long1, 8000 chars) first, not off the chronologically-last entry.
    text, truncated = llm_judge.render_observations(observations, cap=12000)
    assert truncated is True
    assert "...[truncated 1100 chars]" in text
    # the short and long2 entries should be untouched (fully present)
    assert short in text
    assert long2 in text
    # long1 should be shortened by exactly the excess
    assert ("l" * (8000 - 1100)) in text
    assert ("l" * 8000) not in text


def test_render_observations_empty_list():
    text, truncated = llm_judge.render_observations([], cap=12000)
    assert truncated is False
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# missing raw record fails loudly (not semantics-only fallback)
# ---------------------------------------------------------------------------


def test_run_fails_loudly_when_raw_record_missing_for_evidence(monkeypatch, tmp_path):
    class Args:
        only = "T404"
        gap_set = False
        subset = None
        repeats = 1
        hypothesis_source = "auto"

    args = Args()

    monkeypatch.setattr(
        llm_judge,
        "load_tickets",
        lambda: {
            "T404": {
                "id": "T404",
                "ticket_text": "text",
                "gold_root_cause": "slug",
                "category": "easy",
            }
        },
    )
    monkeypatch.setattr(llm_judge, "load_raw_hypothesis", lambda tid: None)
    monkeypatch.setattr(llm_judge, "load_gap_set_hypothesis", lambda tid: "gap hyp")
    monkeypatch.setattr(llm_judge, "load_raw_category", lambda tid: None)
    monkeypatch.setattr(llm_judge, "load_observations", lambda tid, raw_dir=None: None)
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call LLM"))
    )

    report = llm_judge.run(args)
    row = report["per_ticket"][0]
    assert row["verdict"] is None
    assert row["semantic_match"] is None
    assert row["evidence_supported"] is None
    assert "T404" in row["error"]
    assert "raw" in row["error"].lower()
    # NOT a semantics-only fallback: no verdict was ever computed
    assert row["verdicts"] == []


def test_gap_set_hypothesis_source_errors_clearly_under_v3():
    class Args:
        only = None
        gap_set = True
        subset = None
        repeats = 1
        hypothesis_source = "gap-set"

    with pytest.raises(ValueError) as excinfo:
        llm_judge.run(Args())
    msg = str(excinfo.value).lower()
    assert "gap-set" in msg
    assert "evidence" in msg or "clause (b)" in msg or "clause b" in msg


# ---------------------------------------------------------------------------
# decomposition summary
# ---------------------------------------------------------------------------


def test_summary_decomposes_incorrect_verdicts_by_clause(monkeypatch):
    class Args:
        only = "A,B,C,D"
        gap_set = False
        subset = None
        repeats = 1
        hypothesis_source = "auto"

    args = Args()

    # A: semantic fails only (sm=False, es=True) -> semantic_only
    # B: evidence fails only (sm=True, es=False) -> evidence_only
    # C: both fail (sm=False, es=False) -> both
    # D: both pass -> correct, not counted
    responses = {
        "A": _fake_response(False, True),
        "B": _fake_response(True, False),
        "C": _fake_response(False, False),
        "D": _fake_response(True, True),
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
    monkeypatch.setattr(llm_judge, "load_observations", lambda tid, raw_dir=None: SAMPLE_OBSERVATIONS)

    def fake_judge_once(ticket_text, category, hypothesis, gold_root_cause, observations):
        tid = hypothesis.split()[-1]
        resp = responses[tid]
        return llm_judge._extract_verdict(resp)

    monkeypatch.setattr(llm_judge, "judge_once", fake_judge_once)

    report = llm_judge.run(args)
    summary = report["summary"]
    assert summary["n_fail_semantic_only"] == 1
    assert summary["n_fail_evidence_only"] == 1
    assert summary["n_fail_both"] == 1
    assert summary["n_correct"] == 1


# ---------------------------------------------------------------------------
# hypothesis source resolution (unchanged by v3)
# ---------------------------------------------------------------------------


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
    monkeypatch.setattr(llm_judge, "load_observations", lambda tid, raw_dir=None: SAMPLE_OBSERVATIONS)
    monkeypatch.setattr(
        llm_judge, "call_llm_with_tools", lambda *a, **k: _fake_response(True, True)
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


def test_rubric_version_is_3():
    assert llm_judge.RUBRIC_VERSION == 3


def test_v2_rubric_and_prompt_still_reachable():
    assert llm_judge.RUBRIC_VERSION_V2 == 2
    prompt = llm_judge.build_prompt_v2(
        "ticket text", "easy", "hypothesis", "gold_slug"
    )
    assert llm_judge.RUBRIC_DEFINITION_V2 in prompt
    assert "semantically_correct" in prompt
