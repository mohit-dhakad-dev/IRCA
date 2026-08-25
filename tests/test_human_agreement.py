import json

import pytest

from eval.human_agreement import (
    build_template,
    build_template_with_provenance,
    cohens_kappa,
    run_compare,
    run_template,
    select_sample,
    select_sample_disagreement,
)
from eval.llm_judge import RUBRIC_TEXT, RUBRIC_VERSION

TEMPLATE_ROW_KEYS = {
    "ticket_id",
    "category",
    "gold_root_cause",
    "hypothesis",
    "ticket_text",
    "observations",
    "semantically_correct",
    "notes",
}


# ---------------------------------------------------------------------------
# cohens_kappa
# ---------------------------------------------------------------------------


def test_kappa_known_example():
    # 20 items: both_true=8, both_false=8, human_true_judge_false=2,
    # human_false_judge_true=2.
    # p_o = 16/20 = 0.8
    # a_true = (8+2)/20 = 0.5, a_false = 0.5
    # b_true = (8+2)/20 = 0.5, b_false = 0.5
    # p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    # kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
    pairs = (
        [(True, True)] * 8
        + [(False, False)] * 8
        + [(True, False)] * 2
        + [(False, True)] * 2
    )
    kappa, p_o, note = cohens_kappa(pairs)
    assert p_o == pytest.approx(0.8)
    assert kappa == pytest.approx(0.6)


def test_pabak_equals_2po_minus_1_same_example():
    pairs = (
        [(True, True)] * 8
        + [(False, False)] * 8
        + [(True, False)] * 2
        + [(False, True)] * 2
    )
    kappa, p_o, note = cohens_kappa(pairs)
    pabak = 2 * p_o - 1
    assert pabak == pytest.approx(0.6)
    assert pabak == pytest.approx(p_o * 2 - 1)


def test_kappa_degenerate_all_true():
    pairs = [(True, True)] * 10
    kappa, p_o, note = cohens_kappa(pairs)
    assert kappa is None
    assert p_o == pytest.approx(1.0)
    pabak = 2 * p_o - 1
    assert pabak == pytest.approx(1.0)
    assert "PABAK" in note


def test_kappa_prevalence_problem():
    # n=100: both_true=90, both_false=0, human_true_judge_false=5,
    # human_false_judge_true=5.
    # p_o = 0.90
    # a_true = 95/100 = 0.95, a_false = 0.05
    # b_true = 95/100 = 0.95, b_false = 0.05
    # p_e = 0.95^2 + 0.05^2 = 0.905
    # kappa = (0.90 - 0.905) / (1 - 0.905) = -0.0526...
    pairs = (
        [(True, True)] * 90
        + [(True, False)] * 5
        + [(False, True)] * 5
    )
    kappa, p_o, note = cohens_kappa(pairs)
    pabak = 2 * p_o - 1
    assert p_o == pytest.approx(0.90)
    assert kappa == pytest.approx(-0.0526, abs=1e-3)
    assert pabak == pytest.approx(0.80)
    # kappa and pabak diverge substantially
    assert abs(pabak - kappa) > 0.5
    assert "prevalence" in note.lower()


# ---------------------------------------------------------------------------
# compare / skip accounting
# ---------------------------------------------------------------------------


def test_compare_skip_accounting(tmp_path):
    human_path = tmp_path / "human.json"
    judge_path = tmp_path / "judge.json"
    out_path = tmp_path / "report.json"

    human_data = {
        "tickets": [
            {"ticket_id": "T1", "semantically_correct": True},
            {"ticket_id": "T2", "semantically_correct": None},
            {"ticket_id": "T3", "semantically_correct": True},
        ]
    }
    judge_data = {
        "per_ticket": [
            {"ticket_id": "T1", "verdict": True},
            {"ticket_id": "T2", "verdict": True},
            {"ticket_id": "T3", "verdict": None},
        ]
    }
    human_path.write_text(json.dumps(human_data), encoding="utf-8")
    judge_path.write_text(json.dumps(judge_data), encoding="utf-8")

    class Args:
        human = str(human_path)
        judge = str(judge_path)
        out = str(out_path)

    run_compare(Args())

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["n_compared"] == 1
    assert report["n_skipped"] == {
        "human_unlabeled": 1,
        "judge_failed": 1,
        "not_in_both": 0,
    }
    assert report["contingency"]["both_true"] == 1


# ---------------------------------------------------------------------------
# template blindness (+ observations rendering)
# ---------------------------------------------------------------------------


def test_template_blindness(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"

    gap_set = {
        "n": 2,
        "tickets": [
            {
                "ticket_id": "T1",
                "category": "easy",
                "gold_root_cause": "cause_a",
                "hypothesis": "hyp a",
            },
            {
                "ticket_id": "T2",
                "category": "easy",
                "gold_root_cause": "cause_b",
                "hypothesis": "hyp b",
            },
        ],
    }
    ragas = {
        "per_ticket": [
            {"ticket_id": "T1", "faithfulness": 0.2},
            {"ticket_id": "T2", "faithfulness": 0.5},
        ]
    }
    tickets = [
        {"id": "T1", "ticket_text": "text 1"},
        {"id": "T2", "ticket_text": "text 2"},
    ]
    gap_set_path.write_text(json.dumps(gap_set), encoding="utf-8")
    ragas_path.write_text(json.dumps(ragas), encoding="utf-8")
    tickets_path.write_text(json.dumps(tickets), encoding="utf-8")

    template = build_template(gap_set_path, ragas_path, tickets_path, n=2)

    dumped = json.dumps(template)
    for forbidden in ["verdict", "reasoning", "judge"]:
        assert forbidden not in dumped

    assert len(template) == 2
    for row in template:
        assert row["semantically_correct"] is None
        assert set(row.keys()) == TEMPLATE_ROW_KEYS
        # no raw record exists for these synthetic tickets -> unavailable note
        assert "no raw trajectory record" in row["observations"]
        assert "T1" in row["observations"] or "T2" in row["observations"]


def test_template_includes_rendered_observations_from_raw_record(tmp_path, monkeypatch):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    gap_set = {
        "n": 1,
        "tickets": [
            {"ticket_id": "T1", "category": "easy", "gold_root_cause": "cause_a", "hypothesis": "hyp a"},
        ],
    }
    ragas = {"per_ticket": [{"ticket_id": "T1", "faithfulness": 0.2}]}
    tickets = [{"id": "T1", "ticket_text": "text 1"}]
    raw = {
        "state": {
            "trajectory": [
                {
                    "iteration": 0,
                    "tool_call": {"name": "query_logs"},
                    "observation": {"status": "ok", "data": {"disk_pct": 97}},
                }
            ]
        }
    }
    (raw_dir / "T1.json").write_text(json.dumps(raw), encoding="utf-8")
    gap_set_path.write_text(json.dumps(gap_set), encoding="utf-8")
    ragas_path.write_text(json.dumps(ragas), encoding="utf-8")
    tickets_path.write_text(json.dumps(tickets), encoding="utf-8")

    import eval.human_agreement as human_agreement

    monkeypatch.setattr(human_agreement, "RAW_DIR", raw_dir)

    template = build_template(gap_set_path, ragas_path, tickets_path, n=1)
    assert len(template) == 1
    assert "query_logs" in template[0]["observations"]
    assert "disk_pct" in template[0]["observations"]


def test_template_blindness_disagreement_strategy_and_provenance(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"
    judge_path = tmp_path / "judge.json"

    gap_set = {
        "n": 4,
        "tickets": [
            {"ticket_id": "T1", "category": "easy", "gold_root_cause": "a", "hypothesis": "h1"},
            {"ticket_id": "T2", "category": "easy", "gold_root_cause": "b", "hypothesis": "h2"},
            {"ticket_id": "T3", "category": "rag_heavy", "gold_root_cause": "c", "hypothesis": "h3"},
            {"ticket_id": "T4", "category": "rag_heavy", "gold_root_cause": "d", "hypothesis": "h4"},
        ],
    }
    judge = {
        "per_ticket": [
            {"ticket_id": "T1", "verdict": False, "agreement": 1.0, "reasonings": ["x"]},
            {"ticket_id": "T2", "verdict": True, "agreement": 0.5, "verdicts": [True, False]},
            {"ticket_id": "T3", "verdict": True, "agreement": 1.0},
            {"ticket_id": "T4", "verdict": True, "agreement": 1.0},
        ]
    }
    tickets = [
        {"id": "T1", "ticket_text": "text 1"},
        {"id": "T2", "ticket_text": "text 2"},
        {"id": "T3", "ticket_text": "text 3"},
        {"id": "T4", "ticket_text": "text 4"},
    ]
    gap_set_path.write_text(json.dumps(gap_set), encoding="utf-8")
    ragas_path.write_text(json.dumps({"per_ticket": []}), encoding="utf-8")
    tickets_path.write_text(json.dumps(tickets), encoding="utf-8")
    judge_path.write_text(json.dumps(judge), encoding="utf-8")

    template, provenance = build_template_with_provenance(
        gap_set_path, ragas_path, tickets_path, n=4,
        strategy="disagreement", judge_path=judge_path,
    )

    dumped = json.dumps(template)
    for forbidden in ["verdict", "reasoning", "agreement", "faithfulness", "verdicts"]:
        assert forbidden not in dumped

    ids = {t["ticket_id"] for t in template}
    assert ids == {"T1", "T2", "T3", "T4"}
    for row in template:
        assert row["semantically_correct"] is None
        assert set(row.keys()) == TEMPLATE_ROW_KEYS

    assert provenance["strategy"] == "disagreement"
    assert provenance["n_mandatory_disagreement"] == 2


# ---------------------------------------------------------------------------
# template sampling
# ---------------------------------------------------------------------------


def test_template_sampling_allocation_and_low_faithfulness():
    gap_tickets = []
    faithfulness = {}
    # fake categories: catA has 6 tickets, catB has 4 tickets -> 10 total
    for i in range(6):
        tid = f"A{i}"
        gap_tickets.append({"ticket_id": tid, "category": "catA"})
        faithfulness[tid] = float(i)  # A0 lowest, A5 highest
    for i in range(4):
        tid = f"B{i}"
        gap_tickets.append({"ticket_id": tid, "category": "catB"})
        faithfulness[tid] = float(i)

    n = 5
    sample = select_sample(gap_tickets, faithfulness, n)
    assert len(sample) == n

    cats = [t["category"] for t in sample]
    # proportional: catA 6/10*5=3, catB 4/10*5=2
    assert cats.count("catA") == 3
    assert cats.count("catB") == 2

    ids = {t["ticket_id"] for t in sample}
    # lowest-faithfulness within each category first
    assert {"A0", "A1", "A2"}.issubset(ids)
    assert {"B0", "B1"}.issubset(ids)


def test_template_fails_loudly_when_ragas_missing(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "missing_ragas.json"
    tickets_path = tmp_path / "tickets.json"

    gap_set_path.write_text(json.dumps({"n": 0, "tickets": []}), encoding="utf-8")
    tickets_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="RAGAS eval"):
        build_template(gap_set_path, ragas_path, tickets_path, n=5, strategy="faithfulness")


# ---------------------------------------------------------------------------
# disagreement sampling strategy
# ---------------------------------------------------------------------------


def _make_gap_tickets():
    tickets = []
    # 4 easy, 4 tool_heavy -> total 8 candidates in the true pool (plus
    # mandatory disagreement ones below)
    for i in range(4):
        tickets.append({"ticket_id": f"E{i}", "category": "easy"})
    for i in range(4):
        tickets.append({"ticket_id": f"H{i}", "category": "tool_heavy"})
    tickets.append({"ticket_id": "F1", "category": "easy"})
    tickets.append({"ticket_id": "F2", "category": "tool_heavy"})
    tickets.append({"ticket_id": "U1", "category": "easy"})
    return tickets


def _make_judge_rows():
    rows = []
    for i in range(4):
        rows.append({"ticket_id": f"E{i}", "verdict": True, "agreement": 1.0})
    for i in range(4):
        rows.append({"ticket_id": f"H{i}", "verdict": True, "agreement": 1.0})
    rows.append({"ticket_id": "F1", "verdict": False, "agreement": 1.0})
    rows.append({"ticket_id": "F2", "verdict": False, "agreement": 1.0})
    rows.append({"ticket_id": "U1", "verdict": True, "agreement": 0.5})
    return rows


def test_disagreement_includes_all_false_and_nonunanimous():
    gap_tickets = _make_gap_tickets()
    judge_rows = _make_judge_rows()

    selected, provenance = select_sample_disagreement(gap_tickets, judge_rows, n=6)
    ids = {t["ticket_id"] for t in selected}

    assert {"F1", "F2", "U1"}.issubset(ids)
    assert provenance["n_mandatory_disagreement"] == 3


def test_disagreement_fills_remaining_slots_proportionally_by_category():
    gap_tickets = _make_gap_tickets()
    judge_rows = _make_judge_rows()

    selected, provenance = select_sample_disagreement(gap_tickets, judge_rows, n=7)
    ids = {t["ticket_id"] for t in selected}

    # mandatory: F1, F2, U1 (3). remaining = 4 slots from true pool
    # (E0-3 easy=4, H0-3 tool_heavy=4 -> total 8, proportional 4/8 each -> 2 each)
    assert provenance["n_mandatory_disagreement"] == 3
    assert provenance["n_filled_true"] == 4
    assert len(selected) == 7

    easy_filled = sum(1 for t in selected if t["category"] == "easy" and t["ticket_id"].startswith("E"))
    tool_filled = sum(1 for t in selected if t["category"] == "tool_heavy" and t["ticket_id"].startswith("H"))
    assert easy_filled == 2
    assert tool_filled == 2

    # deterministic tie-break: lowest ticket_id ascending
    assert {"E0", "E1"}.issubset(ids)
    assert {"H0", "H1"}.issubset(ids)


def test_disagreement_mandatory_exceeds_n_includes_all_and_warns(capsys):
    gap_tickets = _make_gap_tickets()
    judge_rows = _make_judge_rows()

    selected, provenance = select_sample_disagreement(gap_tickets, judge_rows, n=2)
    ids = {t["ticket_id"] for t in selected}

    # mandatory set (3) exceeds n (2); nothing dropped
    assert {"F1", "F2", "U1"}.issubset(ids)
    assert len(selected) == 3
    assert provenance["n_mandatory_disagreement"] == 3
    assert provenance["n_exceeded"] == 1


def test_disagreement_missing_judge_file_fails_loudly(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"
    missing_judge_path = tmp_path / "no_judge.json"

    gap_set_path.write_text(json.dumps({"n": 0, "tickets": []}), encoding="utf-8")
    ragas_path.write_text(json.dumps({"per_ticket": []}), encoding="utf-8")
    tickets_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"no_judge\.json"):
        build_template(
            gap_set_path, ragas_path, tickets_path, n=5,
            strategy="disagreement", judge_path=missing_judge_path,
        )


def test_faithfulness_strategy_still_reproduces_original_behaviour(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"

    gap_tickets = []
    faithfulness = {}
    for i in range(6):
        tid = f"A{i}"
        gap_tickets.append({"ticket_id": tid, "category": "catA", "gold_root_cause": "g", "hypothesis": "h"})
        faithfulness[tid] = float(i)
    for i in range(4):
        tid = f"B{i}"
        gap_tickets.append({"ticket_id": tid, "category": "catB", "gold_root_cause": "g", "hypothesis": "h"})
        faithfulness[tid] = float(i)

    gap_set_path.write_text(json.dumps({"n": 10, "tickets": gap_tickets}), encoding="utf-8")
    ragas_path.write_text(
        json.dumps({"per_ticket": [{"ticket_id": tid, "faithfulness": f} for tid, f in faithfulness.items()]}),
        encoding="utf-8",
    )
    tickets_path.write_text(
        json.dumps([{"id": t["ticket_id"], "ticket_text": "x"} for t in gap_tickets]),
        encoding="utf-8",
    )

    template, provenance = build_template_with_provenance(
        gap_set_path, ragas_path, tickets_path, n=5, strategy="faithfulness",
    )
    assert provenance is None
    ids = {t["ticket_id"] for t in template}
    assert {"A0", "A1", "A2"}.issubset(ids)
    assert {"B0", "B1"}.issubset(ids)


def test_run_template_writes_sampling_provenance_and_warning(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"
    judge_path = tmp_path / "judge.json"
    out_path = tmp_path / "out.json"

    gap_tickets = _make_gap_tickets()
    judge_rows = _make_judge_rows()

    gap_set_path.write_text(json.dumps({"n": len(gap_tickets), "tickets": [
        {**t, "gold_root_cause": "g", "hypothesis": "h"} for t in gap_tickets
    ]}), encoding="utf-8")
    ragas_path.write_text(json.dumps({"per_ticket": []}), encoding="utf-8")
    tickets_path.write_text(
        json.dumps([{"id": t["ticket_id"], "ticket_text": "x"} for t in gap_tickets]),
        encoding="utf-8",
    )
    judge_path.write_text(json.dumps({"per_ticket": judge_rows}), encoding="utf-8")

    class Args:
        gap_set = str(gap_set_path)
        ragas = str(ragas_path)
        tickets = str(tickets_path)
        judge = str(judge_path)
        out = str(out_path)
        n = 7
        strategy = "disagreement"

    run_template(Args())

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert "sampling" in written
    assert written["sampling"]["strategy"] == "disagreement"
    assert written["sampling"]["n_requested"] == 7
    assert "WARNING" in written
    assert "NOT representative" in written["WARNING"]

    # blindness: no judge-derived key appears anywhere in the emitted file
    # (checked by key name, not substring, since e.g. the strategy name
    # "disagreement" legitimately contains the substring "agreement")
    def _collect_keys(obj, keys):
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(k)
                _collect_keys(v, keys)
        elif isinstance(obj, list):
            for item in obj:
                _collect_keys(item, keys)

    all_keys = set()
    _collect_keys(written, all_keys)
    assert "agreement" not in all_keys
    assert "faithfulness" not in all_keys
    assert "verdicts" not in all_keys
    assert "verdict" not in all_keys
    assert "reasoning" not in all_keys
    assert "reasonings" not in all_keys

    for row in written["tickets"]:
        assert row["semantically_correct"] is None


# ---------------------------------------------------------------------------
# rubric parity and versioning
# ---------------------------------------------------------------------------


def test_template_rubric_matches_judge_rubric_v3(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"
    out_path = tmp_path / "out.json"

    gap_tickets = [
        {"ticket_id": "T1", "category": "easy", "gold_root_cause": "a", "hypothesis": "h1"},
    ]
    gap_set_path.write_text(json.dumps({"n": 1, "tickets": gap_tickets}), encoding="utf-8")
    ragas_path.write_text(
        json.dumps({"per_ticket": [{"ticket_id": "T1", "faithfulness": 0.5}]}),
        encoding="utf-8",
    )
    tickets_path.write_text(
        json.dumps([{"id": "T1", "ticket_text": "text 1"}]), encoding="utf-8"
    )

    class Args:
        gap_set = str(gap_set_path)
        ragas = str(ragas_path)
        tickets = str(tickets_path)
        judge = None
        out = str(out_path)
        n = 1
        strategy = "faithfulness"

    run_template(Args())

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["rubric"]["version"] == RUBRIC_VERSION == 3
    assert written["rubric"]["text"] == RUBRIC_TEXT


def test_template_instructions_mention_both_clauses_and_stay_blind(tmp_path):
    gap_set_path = tmp_path / "gap_set.json"
    ragas_path = tmp_path / "ragas.json"
    tickets_path = tmp_path / "tickets.json"
    out_path = tmp_path / "out.json"

    gap_tickets = [
        {"ticket_id": "T1", "category": "easy", "gold_root_cause": "a", "hypothesis": "h1"},
    ]
    gap_set_path.write_text(json.dumps({"n": 1, "tickets": gap_tickets}), encoding="utf-8")
    ragas_path.write_text(
        json.dumps({"per_ticket": [{"ticket_id": "T1", "faithfulness": 0.5}]}),
        encoding="utf-8",
    )
    tickets_path.write_text(
        json.dumps([{"id": "T1", "ticket_text": "text 1"}]), encoding="utf-8"
    )

    class Args:
        gap_set = str(gap_set_path)
        ragas = str(ragas_path)
        tickets = str(tickets_path)
        judge = None
        out = str(out_path)
        n = 1
        strategy = "faithfulness"

    run_template(Args())

    written = json.loads(out_path.read_text(encoding="utf-8"))
    instructions = written["INSTRUCTIONS"]
    assert "clause (a)" in instructions
    assert "clause (b)" in instructions
    assert "BOTH" in instructions
    assert "observations" in instructions
    for forbidden in ["strict_lexical", "status_only", "task_success", "semantic_match\":", "evidence_supported\":"]:
        assert forbidden not in json.dumps(written)


def test_compare_rubric_version_mismatch_raises(tmp_path):
    human_path = tmp_path / "human.json"
    judge_path = tmp_path / "judge.json"
    out_path = tmp_path / "report.json"

    human_data = {
        "rubric": {"version": 1},
        "tickets": [{"ticket_id": "T1", "semantically_correct": True}],
    }
    judge_data = {
        "config": {"rubric_version": 2},
        "per_ticket": [{"ticket_id": "T1", "verdict": True}],
    }
    human_path.write_text(json.dumps(human_data), encoding="utf-8")
    judge_path.write_text(json.dumps(judge_data), encoding="utf-8")

    class Args:
        human = str(human_path)
        judge = str(judge_path)
        out = str(out_path)

    with pytest.raises(ValueError, match="rubric version mismatch"):
        run_compare(Args())


def test_compare_refuses_v2_human_against_v3_judge(tmp_path):
    human_path = tmp_path / "human.json"
    judge_path = tmp_path / "judge.json"
    out_path = tmp_path / "report.json"

    human_data = {
        "rubric": {"version": 2},
        "tickets": [{"ticket_id": "T1", "semantically_correct": True}],
    }
    judge_data = {
        "config": {"rubric_version": 3},
        "per_ticket": [{"ticket_id": "T1", "verdict": True}],
    }
    human_path.write_text(json.dumps(human_data), encoding="utf-8")
    judge_path.write_text(json.dumps(judge_data), encoding="utf-8")

    class Args:
        human = str(human_path)
        judge = str(judge_path)
        out = str(out_path)

    with pytest.raises(ValueError, match="rubric version mismatch"):
        run_compare(Args())


def test_compare_missing_rubric_version_warns_but_does_not_raise(tmp_path, capsys):
    human_path = tmp_path / "human.json"
    judge_path = tmp_path / "judge.json"
    out_path = tmp_path / "report.json"

    human_data = {
        "tickets": [{"ticket_id": "T1", "semantically_correct": True}],
    }
    judge_data = {
        "config": {"rubric_version": 2},
        "per_ticket": [{"ticket_id": "T1", "verdict": True}],
    }
    human_path.write_text(json.dumps(human_data), encoding="utf-8")
    judge_path.write_text(json.dumps(judge_data), encoding="utf-8")

    class Args:
        human = str(human_path)
        judge = str(judge_path)
        out = str(out_path)

    run_compare(Args())
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "not interpretable" in captured.out.lower() or "NOT interpretable" in captured.out

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["rubric_versions"] == {"human": None, "judge": 2}
    assert report["rubric_warning"] is not None
