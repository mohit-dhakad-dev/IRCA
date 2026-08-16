import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from agent.state import TaskState


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
