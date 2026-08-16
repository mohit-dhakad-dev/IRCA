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
    assert data["iteration_count"] == 0
    assert isinstance(data["ticket_id"], str)
    assert data["ticket_id"].startswith("TKT-")
    assert isinstance(data["plan"], list)
    assert isinstance(data["trajectory"], list)
