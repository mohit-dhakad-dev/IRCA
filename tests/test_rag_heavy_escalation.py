"""Regression tests for the rag_heavy escalate cases (T049, T050).

Measured fact (do not re-derive here): T049 and T050 escalate ONLY because
agent.orchestrator._can_resolve requires a CREDITED "search_runbooks" source
specifically, and search_runbooks returns status="no_confident_match" for
both (scores 0.3734 / 0.3385, below the 0.5 gate in rag/retrieve.py), so it
is never credited by _credit_evidence. The memory layer would happily credit
T050 -- its search_past_incidents score is 0.4505, above the 0.40 gate in
memory/store.py, status "ok". If the runbook-credit requirement in
_can_resolve is ever weakened (e.g. replaced with "any retrieval source" or
dropped entirely), these two tickets would silently start resolving instead
of escalating. These tests are designed to fail loudly the moment that
happens.

T055 was REMOVED from this set on 2026-08-23 because its gold label was
corrected (see data/tickets.json notes on T055): its expected_behavior is
now "resolve_with_approval", not "escalate". The scores above (0.3734 /
0.3385 / 0.2048, the third being T055's) measure search_runbooks retrieval
on TICKET TEXT -- but the agent never queries search_runbooks with ticket
text. It queries with vocabulary read from tool observations, and T055's
log fixture leaks the exact runbook wording ("OOM command not allowed when
used memory > 'maxmemory'"), so the query the agent actually issues
retrieves T055's own gold runbook (RB-MEMORY-001) at 0.58 -- well above the
gate. This guard's ticket-text-only measurement never exercised that path,
which is why it did not catch the stale label.

Note (open question, not yet acted on): T049 and T050 ALSO retrieved their
gold runbooks confidently in the 2026-08-23 sweep when queried with observed
vocabulary rather than ticket text -- T049 matched RB-MEMORY-001 at 0.75 and
T050 matched RB-NETWORK-001 at 0.73. The same staleness question that applied
to T055 is therefore open for T049/T050 too, and is awaiting a decision. Their
labels and membership in RAG_HEAVY_ESCALATE_IDS are UNCHANGED here pending
that decision; do not relabel them based on this note alone.

Follows the conventions of tests/test_orchestrator.py (unit tests on
_can_resolve / _credit_evidence, no network, TaskState built by hand) and
tests/test_retrieve.py (chroma-backed tests build a throwaway index into a
session-scoped tmp dir, no special marker -- chroma is local/offline, not the
`live` marker's concern, which is reserved for the real Groq API).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent.orchestrator as orchestrator_module
from agent.state import TaskState
from memory.ingest import build_index as build_memory_index
from memory.store import SCORE_THRESHOLD as MEMORY_SCORE_THRESHOLD
from memory.store import search_past_incidents
from rag.ingest import COLLECTION_NAME, build_index as build_runbook_index
from rag.retrieve import SCORE_THRESHOLD as RUNBOOK_SCORE_THRESHOLD
from rag.retrieve import search_runbooks
from rag import retrieve

TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.json"
RAG_HEAVY_ESCALATE_IDS = ["T049", "T050"]


def _tickets_by_id() -> dict[str, dict]:
    tickets = json.loads(TICKETS_PATH.read_text())
    return {t["id"]: t for t in tickets}


# ---------------------------------------------------------------------------
# (a) Pure unit test on _can_resolve: proves the search_runbooks-credit
# requirement is load-bearing, not accidentally satisfied by something else.
# Protects T049/T050 -- if this requirement is ever weakened, those
# two tickets would silently start resolving instead of escalating.
# ---------------------------------------------------------------------------
def test_can_resolve_requires_credited_search_runbooks():
    """Drift this catches: _can_resolve silently dropping (or weakening) its
    "search_runbooks" in state.evidence_sources requirement -- the exact
    mechanism that keeps T049/T050 escalating instead of resolving."""
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
# credited, which is the link between "retrieval was weak" and "resolve is
# blocked" for T049/T050.
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


# ---------------------------------------------------------------------------
# (c) Data/retrieval test: confirms the live data actually produces the
# no_confident_match status for search_runbooks on T049/T050, and that
# memory alone WOULD have credited T050 -- which is precisely why the
# runbooks-specific requirement in _can_resolve matters.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runbook_index_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("rag_heavy_runbook_index")
    build_runbook_index(chroma_path=path, collection_name=COLLECTION_NAME)
    yield path
    retrieve._reset_cache()


@pytest.fixture(scope="module")
def memory_index_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("rag_heavy_memory_index")
    build_memory_index(chroma_path=path, collection_name="past_incidents")
    yield path


@pytest.fixture(autouse=True)
def _clear_retrieve_cache_between_tests():
    retrieve._reset_cache()
    yield
    retrieve._reset_cache()


@pytest.mark.parametrize("ticket_id", RAG_HEAVY_ESCALATE_IDS)
def test_rag_heavy_escalate_cases_fail_runbooks_but_pass_memory(
    ticket_id, runbook_index_path, memory_index_path
):
    """Drift this catches: T049/T050's expected_behavior changing away
    from "escalate", or search_runbooks starting to confidently match one of
    them (which would remove the reason they escalate today), or -- for
    T050 specifically -- memory's independent "ok" status disappearing,
    which is the whole reason the runbooks-specific requirement in
    _can_resolve is load-bearing rather than redundant."""
    ticket = _tickets_by_id()[ticket_id]
    assert ticket["expected_behavior"] == "escalate"

    runbook_result = search_runbooks(
        ticket["ticket_text"], chroma_path=runbook_index_path, collection_name=COLLECTION_NAME
    )
    assert runbook_result["status"] == "no_confident_match"
    assert runbook_result["data"]["top_score"] < RUNBOOK_SCORE_THRESHOLD

    if ticket_id == "T050":
        memory_result = search_past_incidents(
            ticket["ticket_text"], chroma_path=memory_index_path, collection_name="past_incidents"
        )
        assert memory_result["status"] == "ok"
        assert memory_result["data"]["top_score"] >= MEMORY_SCORE_THRESHOLD
