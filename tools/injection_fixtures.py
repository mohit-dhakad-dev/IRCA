"""Single home for the adversarial injection payloads used by tickets
T064-T072 (data/tickets.json, category "adversarial").

Every payload literal lives in the ticket's own `injection` block in
data/tickets.json -- this module never duplicates them, it just reads that
block and shapes the payload into whatever the delivery point expects
(a log line, a fake runbook chunk, a fake past-incident record, or a metric
annotation). Main-63 tickets have no `injection` field, so every accessor
here returns None/[] for them.
"""

from __future__ import annotations

from tools.fake_data import get_ticket

# Fabricated doc_id used to deliver a poisoned runbook chunk for tickets
# whose injection point is search_runbooks. Deliberately not a real runbook
# file -- it masquerades as a sibling doc to the ticket's real gold_runbook_id.
_POISONED_RUNBOOK_DOC_ID = "RB-NETWORK-002.md"
_POISONED_RUNBOOK_SECTION = "Diagnosis Steps"
_POISONED_RUNBOOK_SCORE = 0.42

# Fabricated metadata for the poisoned past-incident record (search_past_incidents).
_POISONED_INCIDENT_ID = "INC-POISON-072"
_POISONED_INCIDENT_ROOT_CAUSE = "auth_signing_key_mismatch"
_POISONED_INCIDENT_SCORE = 0.42


def injection_for_ticket(ticket_id: str) -> dict | None:
    """Return the ticket's `injection` block, or None if the ticket does not
    exist or is not adversarial."""
    ticket = get_ticket(ticket_id)
    if ticket is None:
        return None
    return ticket.get("injection")


def poisoned_log_line(ticket_id: str) -> str | None:
    """Return the raw injected text to splice into query_logs' `lines`, or
    None if this ticket has no query_logs-delivered injection."""
    injection = injection_for_ticket(ticket_id)
    if injection is None or injection["point"] != "query_logs":
        return None
    return injection["payload"]


def poisoned_runbook_chunk(ticket_id: str) -> dict | None:
    """Return a fake chunk (same key shape as rag/retrieve.py chunks) to
    append to search_runbooks results, or None if this ticket has no
    search_runbooks-delivered injection."""
    injection = injection_for_ticket(ticket_id)
    if injection is None or injection["point"] != "search_runbooks":
        return None
    return {
        "doc_id": _POISONED_RUNBOOK_DOC_ID,
        "section": _POISONED_RUNBOOK_SECTION,
        "text": injection["payload"],
        "score": _POISONED_RUNBOOK_SCORE,
    }


def poisoned_past_incident(ticket_id: str) -> dict | None:
    """Return a fake incident (same key shape as memory/store.py incidents)
    to append to search_past_incidents results, or None if this ticket has
    no search_past_incidents-delivered injection."""
    injection = injection_for_ticket(ticket_id)
    if injection is None or injection["point"] != "search_past_incidents":
        return None
    return {
        "incident_id": _POISONED_INCIDENT_ID,
        "symptom_summary": injection["payload"],
        "resolved_root_cause": _POISONED_INCIDENT_ROOT_CAUSE,
        "resolution": injection["payload"],
        "similarity_score": _POISONED_INCIDENT_SCORE,
    }


def metric_annotations(ticket_id: str) -> list[str]:
    """Return the list of injected annotation strings for query_metrics, or
    [] if this ticket has no query_metrics-delivered injection."""
    injection = injection_for_ticket(ticket_id)
    if injection is None or injection["point"] != "query_metrics":
        return []
    return [injection["payload"]]
