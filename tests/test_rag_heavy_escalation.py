"""Regression tests protecting the search_runbooks-credit mechanism.

History: this module originally parametrized over RAG_HEAVY_ESCALATE_IDS =
["T049", "T050", "T055"], asserting each escalated because search_runbooks
scored below the 0.5 gate on their raw ticket_text. That premise was only
ever true of TICKET TEXT -- the agent never queries search_runbooks with
ticket text, it queries with vocabulary read from tool observations.

All three cases have now been relabelled on 2026-08-23:
  - T055 was corrected first (see data/tickets.json notes on T055): a
    reworded query hit its gold runbook RB-MEMORY-001 at 0.58.
  - T049 and T050 were corrected in this change: T049's reworded query hit
    RB-MEMORY-001 at 0.94 and RESOLVED with approval queued. T050's reworded
    query hit RB-NETWORK-001 at 0.73 and passed all six _can_resolve gates,
    but escalates via a DIFFERENT mechanism -- a unit-safety bug in
    agent/approval.py's constraint parser (see data/tickets.json notes on
    T050) that rejects its proposed fixes with false verification_failed
    errors. All three are now labelled "resolve_with_approval".

None of them are retrieval-failure cases. A genuine retrieval-failure
regression test would need an incident with NO matching runbook at all --
that is the `ambiguous` category (0/10 measured leakage), not a ticket whose
gold vocabulary is merely hidden from ticket_text while still being
reachable via observed tool vocabulary.

What still deserves protection, and is covered below with hand-built
TaskState objects instead of ticket fixtures (so the guarantee does not
depend on any ticket's label, which can drift):
  (a) agent.orchestrator._can_resolve requires a CREDITED "search_runbooks"
      evidence source specifically -- not just any evidence source.
  (b) agent.orchestrator._credit_evidence never credits a search_runbooks
      observation with status="no_confident_match" as "ok" evidence.

The ticket-parametrized and chroma-backed tests that used to exercise (a)/(b)
indirectly via T049/T050/T055 have been removed: those tickets no longer
have a no_confident_match status in the corrected sweep, so there is no
remaining ticket subject for a chroma-backed regression here.
"""

from __future__ import annotations

import agent.orchestrator as orchestrator_module
from agent.state import TaskState


# ---------------------------------------------------------------------------
# (a) Pure unit test on _can_resolve: proves the search_runbooks-credit
# requirement is load-bearing, not accidentally satisfied by something else.
# ---------------------------------------------------------------------------
def test_can_resolve_requires_credited_search_runbooks():
    """Drift this catches: _can_resolve silently dropping (or weakening) its
    "search_runbooks" in state.evidence_sources requirement."""
    doc_id = "RB-DB-001"
    trajectory_entry = {
        "iteration": 0,
        "tool_call": {"name": "search_runbooks", "arguments": {"query": "q"}},
        "observation": {
            "status": "ok",
            "data": {"chunks": [{"doc_id": doc_id, "section": "s", "text": "t", "score": 0.9}]},
            "summary": "s",
        },
    }

    state = TaskState(ticket_id="T-TEST", description="x")
    state.confidence = 0.95  # well above CONFIDENCE_THRESHOLD
    state.hypothesis = "some_root_cause"
    state.citations = [doc_id]  # a citation whose doc_id WAS observed this run
    state.trajectory = [trajectory_entry]
    # Every OTHER condition is satisfied: confidence, >=2 distinct evidence
    # sources including one from OBSERVATIONAL_TOOLS, a hypothesis, and a
    # verifiable citation -- but "search_runbooks" itself is NOT credited.
    state.evidence_sources = ["query_logs", "query_metrics"]

    assert orchestrator_module._can_resolve(state) is False

    # Flip it: credit search_runbooks too, with everything else held fixed.
    state.evidence_sources = ["query_logs", "query_metrics", "search_runbooks"]
    assert orchestrator_module._can_resolve(state) is True


# ---------------------------------------------------------------------------
# (b) Unit test on the crediting path: proves "no_confident_match" is never
# credited as "ok" evidence.
# ---------------------------------------------------------------------------
def test_no_confident_match_is_not_credited():
    """Drift this catches: _credit_evidence (or a future replacement) ever
    treating a status="no_confident_match" search_runbooks observation as
    creditable "ok" evidence."""
    state = TaskState(ticket_id="T-TEST", description="x")
    round_entries = [
        {
            "tool_name": "search_runbooks",
            "status": "no_confident_match",
            "observation": {
                "status": "no_confident_match",
                "data": {"chunks": [], "top_score": 0.3734},
                "summary": "no confident match",
            },
        }
    ]

    orchestrator_module._credit_evidence(state, round_entries)

    assert "search_runbooks" not in state.evidence_sources
