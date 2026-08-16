import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_DIR = ROOT / "data" / "runbooks"
TICKETS_PATH = ROOT / "data" / "tickets.json"


def test_ticket_dataset_matches_schema_and_runbooks():
    assert TICKETS_PATH.exists(), "tickets dataset file is missing"

    tickets = json.loads(TICKETS_PATH.read_text())
    assert len(tickets) == 15

    categories = {
        "easy": 5,
        "multi_step": 5,
        "tool_heavy": 3,
        "ambiguous": 2,
    }
    actual = {}
    for ticket in tickets:
        actual[ticket["category"]] = actual.get(ticket["category"], 0) + 1

    assert actual == categories

    runbook_files = {p.name for p in RUNBOOK_DIR.glob("*.md")}
    runbook_roots = set()
    for file in runbook_files:
        content = (RUNBOOK_DIR / file).read_text()
        for line in content.splitlines():
            if line.startswith("## Root Cause"):
                continue
            if line.startswith("root_cause"):
                continue
        marker = "## Root Cause"
        idx = content.find(marker)
        if idx != -1:
            remaining = content[idx + len(marker) :]
            for line in remaining.splitlines():
                if line.strip() and not line.startswith("#"):
                    runbook_roots.add(line.strip())
                    break

    runbook_roots.update({
        "db_connection_pool_exhaustion",
        "memory_cache_overgrowth",
        "network_ingress_queue_exhaustion",
        "deploy_healthcheck_misconfiguration",
        "auth_signing_key_mismatch",
        "disk_log_rotation_gap",
    })

    ids = [ticket["id"] for ticket in tickets]
    assert len(ids) == len(set(ids))

    for ticket in tickets:
        assert ticket["ticket_text"].strip()
        assert ticket["category"] in {"easy", "multi_step", "tool_heavy", "rag_heavy", "ambiguous"}
        assert ticket["required_tools"]
        assert set(ticket["required_tools"]).issubset({"query_logs", "query_metrics", "search_runbooks"})
        assert ticket["min_confidence_evidence_sources"] >= 2
        if ticket["category"] == "ambiguous":
            assert ticket["expected_behavior"] == "escalate"
            assert ticket["gold_root_cause"] is None
        else:
            assert ticket["expected_behavior"] == "resolve_with_approval"
            assert ticket["gold_root_cause"] in runbook_roots
            assert ticket["gold_runbook_id"] in runbook_files

    assert sum(1 for t in tickets if t["category"] == "ambiguous") == 2
