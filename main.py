import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.single_pass import run_single_pass
from agent.state import TaskState
from tools.fake_data import get_ticket


class TicketCreateRequest(BaseModel):
    description: str


app = FastAPI(title="IRCA MVP")


@app.post("/tickets")
def create_ticket(payload: TicketCreateRequest) -> TaskState:
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    task_state = TaskState(
        ticket_id=ticket_id,
        description=payload.description,
        status="new",
        plan=[
            "Review ticket details",
            "Inspect available evidence",
            "Prepare a diagnostic response",
        ],
        iteration_count=0,
        trajectory=["ticket_received"],
    )

    return task_state


@app.post("/tickets/{ticket_id}/diagnose")
def diagnose_ticket(ticket_id: str) -> dict:
    """Run the single-pass (baseline) diagnosis for a known synthetic ticket."""
    if get_ticket(ticket_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket id '{ticket_id}'.")
    return run_single_pass(ticket_id)
