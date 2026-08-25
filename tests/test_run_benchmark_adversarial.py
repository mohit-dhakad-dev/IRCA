"""Session 10 Step 2: default sweep excludes adversarial tickets, and
--adversarial selects exactly the adversarial category. Pure tests of
_select_tickets and main()'s flag validation -- no LLM calls, no file I/O
beyond the real data/tickets.json load."""

from __future__ import annotations

from eval import run_benchmark


def test_default_selection_returns_63_tickets_with_no_adversarial():
    tickets = run_benchmark._load_tickets()
    selected = run_benchmark._select_tickets(tickets, subset=None, ticket_ids=None)

    assert len(selected) == 63
    assert all(t.get("category") != "adversarial" for t in selected)


def test_adversarial_flag_returns_exactly_the_nine_adversarial_tickets():
    tickets = run_benchmark._load_tickets()
    selected = run_benchmark._select_tickets(tickets, subset=None, ticket_ids=None, adversarial=True)

    assert len(selected) == 9
    assert all(t.get("category") == "adversarial" for t in selected)
    assert {t["id"] for t in selected} == {f"T0{n}" for n in range(64, 73)}


def test_adversarial_tickets_remain_runnable_via_explicit_tickets_flag():
    tickets = run_benchmark._load_tickets()
    selected = run_benchmark._select_tickets(tickets, subset=None, ticket_ids=["T064", "T001"])

    assert [t["id"] for t in selected] == ["T064", "T001"]


def test_main_rejects_adversarial_combined_with_subset(capsys):
    exit_code = run_benchmark.main(["--live", "--adversarial", "--subset", "2"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


def test_main_rejects_adversarial_combined_with_tickets(capsys):
    exit_code = run_benchmark.main(["--live", "--adversarial", "--tickets", "T064"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
