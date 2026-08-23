import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_DIR = ROOT / "data" / "runbooks"
INCIDENTS_PATH = ROOT / "data" / "past_incidents.json"
TICKETS_PATH = ROOT / "data" / "tickets.json"

SCHEMA_KEYS = {"incident_id", "symptom_summary", "resolved_root_cause", "resolution"}


def _parse_runbook_root_causes():
    roots = set()
    marker = "## Root Cause"
    for path in RUNBOOK_DIR.glob("*.md"):
        content = path.read_text()
        idx = content.find(marker)
        if idx == -1:
            continue
        remaining = content[idx + len(marker):]
        for line in remaining.splitlines():
            if line.strip() and not line.startswith("#"):
                roots.add(line.strip())
                break
    return roots


def test_incident_dataset_matches_schema():
    assert INCIDENTS_PATH.exists(), "past incidents dataset file is missing"

    incidents = json.loads(INCIDENTS_PATH.read_text())
    assert len(incidents) == 8

    ids = [inc["incident_id"] for inc in incidents]
    assert len(ids) == len(set(ids))
    for incident_id in ids:
        assert re.match(r"^INC-\d{3}$", incident_id)

    for incident in incidents:
        assert set(incident.keys()) == SCHEMA_KEYS
        for key in SCHEMA_KEYS:
            assert isinstance(incident[key], str)
            assert incident[key].strip()


def test_incident_root_causes_join_against_runbooks():
    incidents = json.loads(INCIDENTS_PATH.read_text())
    runbook_roots = _parse_runbook_root_causes()

    assert len(runbook_roots) == 6

    for incident in incidents:
        assert incident["resolved_root_cause"] in runbook_roots


def test_incident_root_cause_coverage_and_duplicates():
    incidents = json.loads(INCIDENTS_PATH.read_text())
    runbook_roots = _parse_runbook_root_causes()

    counts = Counter(inc["resolved_root_cause"] for inc in incidents)

    assert set(counts.keys()) == runbook_roots

    counted_values = sorted(counts.values())
    assert counted_values == [1, 1, 1, 1, 2, 2]


def test_incident_summaries_do_not_leak_root_cause_token():
    incidents = json.loads(INCIDENTS_PATH.read_text())
    for incident in incidents:
        assert incident["resolved_root_cause"] not in incident["symptom_summary"]


def test_incident_summaries_are_not_verbatim_tickets():
    incidents = json.loads(INCIDENTS_PATH.read_text())
    tickets = json.loads(TICKETS_PATH.read_text())
    ticket_texts = {t["ticket_text"] for t in tickets}

    for incident in incidents:
        assert incident["symptom_summary"] not in ticket_texts
