"""In-memory store of applied ticket resolutions.

Deliberately in-memory for this MVP: a module-level dict. It does not survive
a process restart -- a real deployment would back this with Postgres/SQLite
(see docs/design.md Memory & State table).

``apply_write`` is the ONLY mutation path in the codebase. No other module may
write into ``RESOLVED_TICKETS``. In particular, ``tools/ticket_tools.py`` (the
model-facing tool) must never import this module or call ``apply_write`` --
the write only happens after a human approves a PendingAction via the
``/approvals/{action_id}/approve`` endpoint in main.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent.approval import PendingAction

RESOLVED_TICKETS: dict[str, dict] = {}


def apply_write(action: PendingAction) -> dict:
    """Write an approved PendingAction's resolution into the store.

    Raises ValueError if `action.status != "approved"` -- this is the
    enforcement point for the whole approval gate, so the guard lives here,
    not at the call site.
    """
    if action.status != "approved":
        raise ValueError(
            f"Cannot apply write for action '{action.action_id}': "
            f"status is '{action.status}', not 'approved'."
        )

    record = {
        "ticket_id": action.ticket_id,
        "root_cause": action.proposed_root_cause,
        "fix": action.proposed_fix,
        "citation_doc_id": action.citation_doc_id,
        "action_id": action.action_id,
        "resolved_at": datetime.now(timezone.utc),
    }
    RESOLVED_TICKETS[action.ticket_id] = record
    return record


def get_resolution(ticket_id: str) -> dict | None:
    return RESOLVED_TICKETS.get(ticket_id)


def clear_store() -> None:
    """Test hook only: wipes the in-memory store. Not part of the runtime API."""
    RESOLVED_TICKETS.clear()
