import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_DIR = ROOT / "data" / "runbooks"
TICKETS_PATH = ROOT / "data" / "tickets.json"


def test_ticket_dataset_matches_schema_and_runbooks():
    assert TICKETS_PATH.exists(), "tickets dataset file is missing"

    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    assert len(tickets) == 72

    categories = {
        "easy": 15,
        "multi_step": 20,
        "tool_heavy": 10,
        "rag_heavy": 8,
        "ambiguous": 10,
        "adversarial": 9,
    }
    actual = {}
    for ticket in tickets:
        actual[ticket["category"]] = actual.get(ticket["category"], 0) + 1

    assert actual == categories

    runbook_files = {p.name for p in RUNBOOK_DIR.glob("*.md")}
    runbook_roots = set()
    for file in runbook_files:
        content = (RUNBOOK_DIR / file).read_text(encoding="utf-8")
        marker = "## Root Cause"
        idx = content.find(marker)
        if idx != -1:
            remaining = content[idx + len(marker) :]
            for line in remaining.splitlines():
                if line.strip() and not line.startswith("#"):
                    runbook_roots.add(line.strip())
                    break

    assert len(runbook_roots) == 6

    ids = [ticket["id"] for ticket in tickets]
    assert len(ids) == len(set(ids))

    main_tickets = [t for t in tickets if t["category"] != "adversarial"]
    adversarial_tickets = [t for t in tickets if t["category"] == "adversarial"]

    for ticket in main_tickets:
        assert ticket["ticket_text"].strip()
        assert ticket["category"] in {"easy", "multi_step", "tool_heavy", "rag_heavy", "ambiguous"}
        assert ticket["required_tools"]
        assert set(ticket["required_tools"]).issubset({"query_logs", "query_metrics", "search_runbooks"})
        assert ticket["min_confidence_evidence_sources"] >= 2
        assert "injection" not in ticket
        if ticket["category"] == "ambiguous":
            assert ticket["expected_behavior"] == "escalate"
            assert ticket["gold_root_cause"] is None
        else:
            assert ticket["expected_behavior"] in {"resolve_with_approval", "escalate"}
            if ticket["expected_behavior"] == "escalate":
                assert ticket["category"] == "rag_heavy"
            assert ticket["gold_root_cause"] in runbook_roots
            assert ticket["gold_runbook_id"] in runbook_files

    assert sum(1 for t in tickets if t["category"] == "ambiguous") == 10

    # --- Session 10 Step 1: adversarial fixtures (T064-T072) -------------
    LEGAL_INJECTION_POINTS = {
        "ticket_text",
        "query_logs",
        "query_metrics",
        "search_runbooks",
        "search_past_incidents",
    }

    assert len(adversarial_tickets) == 9

    adv_ids = [t["id"] for t in adversarial_tickets]
    assert len(adv_ids) == len(set(adv_ids))

    for ticket in adversarial_tickets:
        assert ticket["ticket_text"].strip()
        assert ticket["required_tools"]
        assert set(ticket["required_tools"]).issubset(
            {"query_logs", "query_metrics", "search_runbooks", "search_past_incidents"}
        )
        assert ticket["min_confidence_evidence_sources"] >= 2
        assert ticket["gold_runbook_id"] in runbook_files
        if ticket["gold_root_cause"] is not None:
            assert ticket["gold_root_cause"] in runbook_roots
        assert ticket["expected_behavior"] in {"resolve_ignoring_injection", "block_and_escalate"}
        assert ticket["notes"].strip()

        injection = ticket["injection"]
        assert injection["vector"].strip()
        assert injection["point"] in LEGAL_INJECTION_POINTS
        assert injection["payload"].strip()
        if "leak_canary" in injection:
            canary = injection["leak_canary"]
            assert isinstance(canary, list) and canary
            assert all(isinstance(c, str) and c.strip() for c in canary)
