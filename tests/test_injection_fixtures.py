"""Tests for tools/injection_fixtures.py -- the single home for the
adversarial payloads delivered to tickets T064-T072 (data/tickets.json).
"""

from __future__ import annotations

from tools import injection_fixtures as fx

MAIN_TICKET_ID = "T001"


def test_injection_for_ticket_returns_block_for_adversarial_id():
    injection = fx.injection_for_ticket("T064")
    assert injection is not None
    assert injection["vector"] == "indirect_injection_log"
    assert injection["point"] == "query_logs"


def test_injection_for_ticket_none_for_main_63():
    assert fx.injection_for_ticket(MAIN_TICKET_ID) is None


def test_injection_for_ticket_none_for_unknown_id():
    assert fx.injection_for_ticket("T999") is None


def test_poisoned_log_line_for_query_logs_point():
    line = fx.poisoned_log_line("T064")
    assert line == fx.injection_for_ticket("T064")["payload"]


def test_poisoned_log_line_none_for_non_log_point():
    assert fx.poisoned_log_line("T066") is None  # search_runbooks point
    assert fx.poisoned_log_line(MAIN_TICKET_ID) is None


def test_poisoned_runbook_chunk_for_search_runbooks_point():
    chunk = fx.poisoned_runbook_chunk("T066")
    assert chunk is not None
    assert set(chunk.keys()) == {"doc_id", "section", "text", "score"}
    assert chunk["text"] == fx.injection_for_ticket("T066")["payload"]


def test_poisoned_runbook_chunk_none_for_non_runbook_point():
    assert fx.poisoned_runbook_chunk("T064") is None  # query_logs point
    assert fx.poisoned_runbook_chunk(MAIN_TICKET_ID) is None


def test_poisoned_past_incident_for_search_past_incidents_point():
    incident = fx.poisoned_past_incident("T072")
    assert incident is not None
    assert set(incident.keys()) == {
        "incident_id",
        "symptom_summary",
        "resolved_root_cause",
        "resolution",
        "similarity_score",
    }
    assert incident["resolution"] == fx.injection_for_ticket("T072")["payload"]


def test_poisoned_past_incident_none_for_non_memory_point():
    assert fx.poisoned_past_incident("T064") is None  # query_logs point
    assert fx.poisoned_past_incident(MAIN_TICKET_ID) is None


def test_metric_annotations_for_query_metrics_point():
    annotations = fx.metric_annotations("T071")
    assert annotations == [fx.injection_for_ticket("T071")["payload"]]


def test_metric_annotations_empty_for_non_metrics_point():
    assert fx.metric_annotations("T064") == []  # query_logs point
    assert fx.metric_annotations(MAIN_TICKET_ID) == []
