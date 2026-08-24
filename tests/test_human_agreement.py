import json

import pytest

from eval.human_agreement import (
    build_template,
    cohens_kappa,
    run_compare,
    select_sample,
)


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
    human_path.write_text(json.dumps(human_data))
    judge_path.write_text(json.dumps(judge_data))

    class Args:
        human = str(human_path)
        judge = str(judge_path)
        out = str(out_path)

    run_compare(Args())

    report = json.loads(out_path.read_text())
    assert report["n_compared"] == 1
    assert report["n_skipped"] == {
        "human_unlabeled": 1,
        "judge_failed": 1,
        "not_in_both": 0,
    }
    assert report["contingency"]["both_true"] == 1


# ---------------------------------------------------------------------------
# template blindness
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
    gap_set_path.write_text(json.dumps(gap_set))
    ragas_path.write_text(json.dumps(ragas))
    tickets_path.write_text(json.dumps(tickets))

    template = build_template(gap_set_path, ragas_path, tickets_path, n=2)

    dumped = json.dumps(template)
    for forbidden in ["verdict", "reasoning", "judge"]:
        assert forbidden not in dumped

    assert len(template) == 2
    for row in template:
        assert row["semantically_correct"] is None
        assert set(row.keys()) == {
            "ticket_id",
            "category",
            "gold_root_cause",
            "hypothesis",
            "ticket_text",
            "semantically_correct",
            "notes",
        }


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

    gap_set_path.write_text(json.dumps({"n": 0, "tickets": []}))
    tickets_path.write_text(json.dumps([]))

    with pytest.raises(FileNotFoundError, match="RAGAS eval"):
        build_template(gap_set_path, ragas_path, tickets_path, n=5)
