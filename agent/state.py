from pydantic import BaseModel, Field


class TaskState(BaseModel):
    ticket_id: str
    description: str
    status: str = "new"
    plan: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    trajectory: list[str] = Field(default_factory=list)
