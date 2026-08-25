import json

import eval.memory_warrant as mw


def test_classify_correct_and_confident_is_warranted():
    assert mw.classify("db_connection_pool_exhaustion", "ok", "db_connection_pool_exhaustion") == "warranted"


def test_classify_confident_but_wrong_root_cause_is_no_signal():
    # The important case: a non-null gold, a confident (status="ok") hit,
    # but the wrong root cause. This must NOT be "warranted" -- calling
    # memory here would hand the agent a wrong lead.
    assert mw.classify("db_connection_pool_exhaustion", "ok", "memory_cache_overgrowth") == "no_signal"


def test_classify_null_gold_and_confident_is_hazard():
    assert mw.classify(None, "ok", "memory_cache_overgrowth") == "hazard"


def test_classify_null_gold_and_no_confident_match_is_no_signal():
    assert mw.classify(None, "no_confident_match", None) == "no_signal"


def test_classify_no_confident_match_with_gold_is_no_signal():
    assert mw.classify("db_connection_pool_exhaustion", "no_confident_match", None) == "no_signal"


def _fake_search_result(status, incidents):
    return {"status": status, "data": {"incidents": incidents}, "summary": ""}


def test_classify_ticket_uses_top_incident(monkeypatch):
    def fake_search(query, **kwargs):
        return _fake_search_result(
            "ok",
            [
                {
                    "incident_id": "INC-1",
                    "resolved_root_cause": "db_connection_pool_exhaustion",
                    "similarity_score": 0.61,
                }
            ],
        )

    monkeypatch.setattr(mw, "search_past_incidents", fake_search)
    ticket = {
        "id": "T001",
        "category": "easy",
        "gold_root_cause": "db_connection_pool_exhaustion",
        "ticket_text": "some ticket text",
    }
    row = mw.classify_ticket(ticket)
    assert row == {
        "id": "T001",
        "category": "easy",
        "gold_root_cause": "db_connection_pool_exhaustion",
        "memory_status": "ok",
        "top_incident_id": "INC-1",
        "top_resolved_root_cause": "db_connection_pool_exhaustion",
        "top_score": 0.61,
        "classification": "warranted",
    }


def test_classify_ticket_handles_no_incidents(monkeypatch):
    def fake_search(query, **kwargs):
        return _fake_search_result("no_confident_match", [])

    monkeypatch.setattr(mw, "search_past_incidents", fake_search)
    ticket = {
        "id": "T002",
        "category": "medium",
        "gold_root_cause": "memory_cache_overgrowth",
        "ticket_text": "some ticket text",
    }
    row = mw.classify_ticket(ticket)
    assert row["top_incident_id"] is None
    assert row["top_resolved_root_cause"] is None
    assert row["top_score"] is None
    assert row["classification"] == "no_signal"


def test_build_warrant_report_excludes_adversarial(monkeypatch):
    def fake_search(query, **kwargs):
        return _fake_search_result("no_confident_match", [])

    monkeypatch.setattr(mw, "search_past_incidents", fake_search)
    tickets = [
        {"id": "T001", "category": "easy", "gold_root_cause": None, "ticket_text": "x"},
        {"id": "T064", "category": "adversarial", "gold_root_cause": None, "ticket_text": "y"},
    ]
    report = mw.build_warrant_report(tickets)
    ids = [row["id"] for row in report["tickets"]]
    assert ids == ["T001"]
    assert report["n_tickets"] == 1


def test_persisted_json_header_reflects_real_source(tmp_path, monkeypatch):
    """The threshold/model/corpus-size header must come from the real
    source (memory.store.SCORE_THRESHOLD, memory.ingest.EMBEDDING_MODEL,
    and the actual past_incidents.json length), not a hardcoded literal."""
    def fake_search(query, **kwargs):
        return _fake_search_result("no_confident_match", [])

    monkeypatch.setattr(mw, "search_past_incidents", fake_search)
    monkeypatch.setattr(mw, "SCORE_THRESHOLD", 0.99)
    monkeypatch.setattr(mw, "EMBEDDING_MODEL", "fake-embedding-model")

    fake_incidents_path = tmp_path / "past_incidents.json"
    fake_incidents_path.write_text(json.dumps([{"resolved_root_cause": "a"}, {"resolved_root_cause": "b"}]), encoding="utf-8")
    monkeypatch.setattr(mw, "PAST_INCIDENTS_PATH", fake_incidents_path)

    tickets = [{"id": "T001", "category": "easy", "gold_root_cause": None, "ticket_text": "x"}]
    report = mw.build_warrant_report(tickets)

    assert report["threshold"] == 0.99
    assert report["embedding_model"] == "fake-embedding-model"
    assert report["n_past_incidents"] == 2
    assert report["schema_version"] == mw.SCHEMA_VERSION


def test_main_persists_expected_shape(tmp_path, monkeypatch):
    def fake_search(query, **kwargs):
        return _fake_search_result(
            "ok",
            [
                {
                    "incident_id": "INC-1",
                    "resolved_root_cause": "db_connection_pool_exhaustion",
                    "similarity_score": 0.61,
                }
            ],
        )

    monkeypatch.setattr(mw, "search_past_incidents", fake_search)
    monkeypatch.setattr(mw, "SCORE_THRESHOLD", 0.40)
    monkeypatch.setattr(mw, "EMBEDDING_MODEL", "fake-embedding-model")

    fake_tickets_path = tmp_path / "tickets.json"
    fake_tickets_path.write_text(
        json.dumps(
            [
                {
                    "id": "T001",
                    "category": "easy",
                    "gold_root_cause": "db_connection_pool_exhaustion",
                    "ticket_text": "x",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mw, "TICKETS_PATH", fake_tickets_path)

    fake_incidents_path = tmp_path / "past_incidents.json"
    fake_incidents_path.write_text(json.dumps([{"resolved_root_cause": "a"}]), encoding="utf-8")
    monkeypatch.setattr(mw, "PAST_INCIDENTS_PATH", fake_incidents_path)

    out_path = tmp_path / "memory_warrant.json"
    mw.main(out_path)

    assert out_path.exists()
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == mw.SCHEMA_VERSION
    assert persisted["threshold"] == 0.40
    assert persisted["embedding_model"] == "fake-embedding-model"
    assert persisted["n_past_incidents"] == 1
    assert persisted["n_tickets"] == 1
    row = persisted["tickets"][0]
    assert row["id"] == "T001"
    assert row["classification"] == "warranted"
    assert set(row.keys()) == {
        "id",
        "category",
        "gold_root_cause",
        "memory_status",
        "top_incident_id",
        "top_resolved_root_cause",
        "top_score",
        "classification",
    }
