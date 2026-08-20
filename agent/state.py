from typing import Literal

from pydantic import BaseModel, Field

TaskStatus = Literal["new", "running", "resolved", "escalated", "error"]


class TaskState(BaseModel):
    """Shared state for one diagnosis run.

    trajectory entries are dicts of:
        {"iteration": int, "thought": str, "tool_call": dict | None,
         "observation": dict | None, "hypothesis_after": str | None}
    called_tool_signatures entries are (tool_name, canonical_json_args) —
    see docs/design.md "Loop guard".
    """

    # identity / input
    ticket_id: str
    description: str
    status: TaskStatus = "new"
    plan: list[str] = Field(default_factory=list)

    # current belief
    hypothesis: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_sources: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    # loop control
    iteration: int = 0
    max_iterations: int = 8
    called_tool_signatures: set[tuple[str, str]] = Field(
        default_factory=set, exclude=True
    )

    # history
    trajectory: list[dict] = Field(default_factory=list)

    # write gate (Session 7 spec): tracks the PendingAction, if any, queued by
    # _queue_write_action for human approval. None until a write is queued.
    pending_action_id: str | None = None
