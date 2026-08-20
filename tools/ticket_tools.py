"""The single WRITE-adjacent tool the model may call: ``update_ticket``.

This module deliberately does NOT import ``tools.ticket_store`` and never
references ``RESOLVED_TICKETS`` or ``apply_write``. There is no code path
from this tool to a mutation -- calling ``update_ticket`` can only ever
create a PendingAction awaiting human approval. The only place a ticket is
actually resolved is ``tools.ticket_store.apply_write``, called from the
``/approvals/{action_id}/approve`` endpoint in main.py after a human
approves.
"""

from __future__ import annotations

from agent.approval import create_pending_action, verify_against_constraints


def update_ticket(
    ticket_id: str,
    proposed_root_cause: str,
    proposed_fix: str,
    citation_doc_id: str,
) -> dict:
    """Queue a proposed ticket resolution for human approval.

    Never mutates a ticket by itself. Returns the repo's standard tool-result
    shape {"status", "data", "summary"}.
    """
    for name, value in (
        ("ticket_id", ticket_id),
        ("proposed_root_cause", proposed_root_cause),
        ("proposed_fix", proposed_fix),
        ("citation_doc_id", citation_doc_id),
    ):
        if not isinstance(value, str) or not value.strip():
            return {
                "status": "error",
                "data": {},
                "summary": f"'{name}' must be a non-empty string.",
            }

    verification = verify_against_constraints(proposed_fix, citation_doc_id)
    if not verification["passed"]:
        return {
            "status": "verification_failed",
            "data": {
                "reason": verification["reason"],
                "citation_doc_id": citation_doc_id,
            },
            "summary": (
                f"Proposed fix violates a constraint in the cited runbook "
                f"'{citation_doc_id}': {verification['reason']} Nothing was "
                f"queued for approval. Revise the fix or pick a different "
                f"citation."
            ),
        }

    action = create_pending_action(
        ticket_id=ticket_id,
        proposed_root_cause=proposed_root_cause,
        proposed_fix=proposed_fix,
        citation_doc_id=citation_doc_id,
    )

    return {
        "status": "awaiting_approval",
        "data": {
            "action_id": action.action_id,
            "ticket_id": action.ticket_id,
            "status": "pending",
            "citation_doc_id": action.citation_doc_id,
        },
        "summary": (
            f"Proposed resolution for ticket '{ticket_id}' has been queued "
            f"for human approval (action_id={action.action_id}). No ticket "
            f"has been changed."
        ),
    }
