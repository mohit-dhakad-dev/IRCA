"""Declarative registry of known, documented agent-behavior defects.

This module exists so that eval/metrics.py's classify_outcome() never has
to hardcode ticket IDs inline. Each KnownIssue documents a REAL, previously
triaged failure mode (see PROGRESS.md) along with the exact status the
failure is expected to produce. classify_outcome() only credits a ticket to
a known issue if the ticket both failed AND failed with that expected
status -- see eval/metrics.py's docstring for why that guard matters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownIssue:
    ticket_ids: tuple[str, ...]
    cause: str
    detail: str
    evidence: str
    expect_status: str


KNOWN_ISSUES: list[KnownIssue] = [
    KnownIssue(
        ticket_ids=("T025", "T029"),
        cause="loop-guard recovery",
        detail=(
            "model reissues the identical blocked call after the guard "
            "fires; the guard itself is confirmed correct"
        ),
        evidence="PROGRESS.md 2026-08-23 triage",
        expect_status="escalated",
    ),
    KnownIssue(
        ticket_ids=("T019",),
        cause="asks user instead of investigating",
        detail=(
            "returns no tool call and asks for information it could have "
            "queried; never credits an observational tool"
        ),
        evidence="PROGRESS.md 2026-08-23, class established by T044",
        expect_status="escalated",
    ),
    KnownIssue(
        ticket_ids=("T039",),
        cause="retrieval-score variance",
        detail=(
            "search_runbooks best score below the 0.50 gate on this run; "
            "the same ticket scores far higher on other runs"
        ),
        evidence="docs/design.md Reliability & Safety, retrieval-score variance",
        expect_status="escalated",
    ),
]


def find_known_issue(ticket_id: str) -> KnownIssue | None:
    """Return the KnownIssue covering ticket_id, or None."""
    for issue in KNOWN_ISSUES:
        if ticket_id in issue.ticket_ids:
            return issue
    return None
