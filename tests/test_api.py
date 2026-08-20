from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_create_ticket_returns_stubbed_task_state():
    payload = {"description": "Checkout service returning 500s intermittently"}

    response = client.post("/tickets", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["description"] == payload["description"]
    assert data["status"] == "new"
    assert data["iteration"] == 0
    assert isinstance(data["ticket_id"], str)
    assert data["ticket_id"].startswith("TKT-")
    assert isinstance(data["plan"], list)
    assert isinstance(data["trajectory"], list)
    assert "called_tool_signatures" not in data
    assert "iteration_count" not in data


def test_resolve_endpoint_unknown_ticket_returns_404():
    response = client.post("/tickets/UNKNOWN-ID/resolve")
    assert response.status_code == 404


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.is_redirect
    assert response.headers["location"] == "/docs"
