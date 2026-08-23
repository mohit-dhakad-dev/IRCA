import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from agent.approval import get_pending_action, list_pending_actions
from agent.orchestrator import run_agent_loop
from agent.single_pass import run_single_pass
from agent.state import TaskState
from tools.fake_data import get_ticket
from tools.ticket_store import apply_write


class TicketCreateRequest(BaseModel):
    description: str


app = FastAPI(title="IRCA MVP")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


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
        iteration=0,
        trajectory=[],
    )

    return task_state


@app.post("/tickets/{ticket_id}/diagnose")
def diagnose_ticket(ticket_id: str) -> dict:
    """Run the single-pass (baseline) diagnosis for a known synthetic ticket."""
    if get_ticket(ticket_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket id '{ticket_id}'.")
    return run_single_pass(ticket_id)


@app.post("/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str) -> TaskState:
    """Run the full agent loop (plan/act/observe/replan/verify) for a known synthetic ticket."""
    if get_ticket(ticket_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket id '{ticket_id}'.")
    return run_agent_loop(ticket_id)


@app.get("/approvals")
def list_approvals() -> list:
    """List all pending (and resolved) write actions awaiting/decided on approval."""
    return list_pending_actions()


@app.get("/approvals/{action_id}")
def get_approval(action_id: str):
    """Fetch one pending write action by id."""
    action = get_pending_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail=f"Unknown action id '{action_id}'.")
    return action


@app.post("/approvals/{action_id}/approve")
def approve_action(action_id: str) -> dict:
    """Approve a pending write action and apply it.

    This is the only caller of tools.ticket_store.apply_write in the codebase.
    """
    action = get_pending_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail=f"Unknown action id '{action_id}'.")
    if action.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Action '{action_id}' is not pending (current status: '{action.status}').",
        )
    action.status = "approved"
    try:
        resolution = apply_write(action)
    except Exception as exc:
        # Revert to "pending" so a failed write doesn't strand the action in
        # "approved" with nothing written and no way to retry -- the endpoint
        # only acts on "pending", so this is what keeps the action actionable.
        action.status = "pending"
        raise HTTPException(status_code=500, detail=f"Failed to apply write: {exc}")
    return {"action_id": action.action_id, "status": "approved", "resolution": resolution}


@app.post("/approvals/{action_id}/reject")
def reject_action(action_id: str) -> dict:
    """Reject a pending write action. Mutates nothing besides the action's status."""
    action = get_pending_action(action_id)
    if action is None:
        raise HTTPException(status_code=404, detail=f"Unknown action id '{action_id}'.")
    if action.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Action '{action_id}' is not pending (current status: '{action.status}').",
        )
    action.status = "rejected"
    return {"action_id": action.action_id, "status": "rejected"}
