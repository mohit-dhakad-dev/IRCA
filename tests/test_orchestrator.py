"""Tests for agent/orchestrator.py (the full plan->act->observe->replan->verify
agent loop).

Fully offline and deterministic: every test monkeypatches
agent.orchestrator.call_llm_with_tools, so no network call is ever made. No
`live` marker is used here.
"""

from __future__ import annotations

import json
import os

import pytest

import agent.orchestrator as orchestrator_module
import agent.tool_executor as tool_executor_module
from agent.orchestrator import CONFIDENCE_THRESHOLD, MAX_NO_NEW_INFO, run_agent_loop
from agent.state import TaskState

TICKET_ID = "T001"


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    # None of the existing tests care about the retry backoff delay, and a
    # couple of them (e.g. the LLM-error-path test) exercise the retry path
    # incidentally. Neutralize it globally so the suite stays fast; the new
    # backoff test below fully replaces time.sleep itself, so this fixture
    # does not interfere with it.
    monkeypatch.setattr(orchestrator_module, "RETRY_BACKOFF_SECONDS", 0)


# ---------------------------------------------------------------------------
# Stub helpers mimicking the groq ChatCompletion response shape (duplicated
# from tests/test_single_pass.py by design — that test module is left alone).
# ---------------------------------------------------------------------------
class _StubFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _StubFunction(name, arguments)


class _StubMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StubChoice:
    def __init__(self, message):
        self.message = message


class _StubCompletion:
    def __init__(self, message):
        self.choices = [_StubChoice(message)]


def _make_tool_call_response(tool_calls):
    return _StubCompletion(_StubMessage(content=None, tool_calls=tool_calls))


def _make_text_response(content):
    return _StubCompletion(_StubMessage(content=content, tool_calls=None))


def _tool_call(call_id, name, arguments):
    return _StubToolCall(call_id, name, json.dumps(arguments))


def _assessment_response(hypothesis, confidence, supports, reasoning="because", citations=None):
    return _make_text_response(
        json.dumps(
            {
                "hypothesis": hypothesis,
                "confidence": confidence,
                "supports": supports,
                "reasoning": reasoning,
                "citations": citations if citations is not None else [],
            }
        )
    )


def _stub_tool(status="ok", summary="stub"):
    def fn(**kwargs):
        return {"status": status, "data": {}, "summary": summary}

    return fn


# search_runbooks' real observation carries doc_id-bearing chunks (see
# rag/retrieve.py); a plain _stub_tool(status="ok") returns data={}, which
# has no doc_id for a citation to ever be verified against. Tests that need
# a run to actually resolve now use this stub instead, and their final
# critic reply cites RUNBOOK_DOC_ID (the citation-verification gate added
# for docs/decisions.md F7).
RUNBOOK_DOC_ID = "RB-DB-001"

# A resolved run now makes one extra LLM call after the loop's stop checks:
# _queue_write_action (the loop's terminal action for a resolved run) asks
# the model to compose a fix, then verifies it against the cited runbook's
# Constraints section before queuing a PendingAction. Every test below that
# reaches "resolved" needs one more scripted text response for that compose
# call. This fix text is written to actually PASS verify_against_constraints
# for RB-DB-001 (no numeric value it states violates that runbook's pool-size
# bounds), so these tests keep exercising a successful resolve rather than
# silently becoming escalation tests.
WRITE_GATE_FIX_TEXT = (
    "Increase the database pool size to a safe higher limit within the "
    "documented ceiling, keeping at least 25% headroom below the DB's hard "
    "limit, then monitor active connections for 45 minutes."
)


def _stub_runbook_tool(doc_id=RUNBOOK_DOC_ID, status="ok", summary="stub"):
    def fn(**kwargs):
        return {
            "status": status,
            "data": {
                "chunks": [
                    {"doc_id": doc_id, "section": "diagnosis", "text": "t", "score": 0.9}
                ]
            },
            "summary": summary,
        }

    return fn


def _install_responses(monkeypatch, responses):
    monkeypatch.setattr(
        orchestrator_module, "call_llm_with_tools", lambda *a, **kw: responses.pop(0)
    )


# ---------------------------------------------------------------------------
# 1. Loop guard
# ---------------------------------------------------------------------------
def test_loop_guard_skips_repeated_tool_call(monkeypatch):
    call_count = {"n": 0}

    def spy_query_logs(**kwargs):
        call_count["n"] += 1
        return {"status": "ok", "data": {}, "summary": "logs"}

    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", spy_query_logs)

    args = {"service": "checkout", "window": "2h", "level": "ERROR"}
    repeated_call = lambda cid: _tool_call(cid, "query_logs", args)

    responses = [
        _make_tool_call_response([repeated_call("c1")]),
        _assessment_response("db issue", 0.3, True),
        _make_tool_call_response([repeated_call("c2")]),
        _assessment_response("db issue", 0.3, True),
        _make_tool_call_response([repeated_call("c3")]),
        _assessment_response("db issue", 0.3, True),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert call_count["n"] == 1
    skipped_entries = [
        e for e in state.trajectory if e["observation"].get("status") == "skipped"
    ]
    assert len(skipped_entries) == 2
    assert "already tried" in skipped_entries[0]["observation"]["summary"].lower()
    assert state.iteration == 3
    assert state.status == "escalated"


# ---------------------------------------------------------------------------
# 2. Iteration cap
# ---------------------------------------------------------------------------
def test_iteration_cap_escalates(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )

    responses = []
    for i in range(1, 9):
        args = {"service": "checkout", "window": f"{i}h", "level": "ERROR"}
        responses.append(_make_tool_call_response([_tool_call(f"c{i}", "query_logs", args)]))
        responses.append(_assessment_response("weak hypothesis", 0.3, True))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.iteration == state.max_iterations
    assert responses == []  # exactly 8 rounds ran, no more, no fewer


# ---------------------------------------------------------------------------
# 3. Scripted 2-step resolve
# ---------------------------------------------------------------------------
def test_scripted_two_step_resolve(monkeypatch):
    # Updated for FIX 1 (_can_resolve now requires a credited search_runbooks
    # source): a third round crediting search_runbooks was added so this run
    # can resolve at all. Without it this is exactly the live-observed T001/
    # T009 case (query_logs + query_metrics only) which must now escalate,
    # not resolve.
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    logs_args = {"service": "checkout", "window": "2h", "level": "ERROR"}
    metrics_args = {
        "service": "checkout",
        "metric": "db_pool_active_connections",
        "window": "2h",
    }

    responses = [
        _make_tool_call_response([_tool_call("c1", "query_logs", logs_args)]),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response([_tool_call("c2", "query_metrics", metrics_args)]),
        _assessment_response("db_connection_pool_exhaustion", 0.8, True),
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    # The loop's terminal action for a resolved run is queueing the write
    # (_queue_write_action), which makes one more LLM call (compose) than
    # a resolved run used to -- hence this extra scripted response.
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert state.confidence == 0.9
    assert state.hypothesis == "db_connection_pool_exhaustion"
    assert state.evidence_sources == ["query_logs", "query_metrics", "search_runbooks"]
    assert state.citations == [RUNBOOK_DOC_ID]
    assert state.iteration == 3


# ---------------------------------------------------------------------------
# 4. Two consecutive no-new-info rounds escalate
# ---------------------------------------------------------------------------
def test_two_consecutive_empty_rounds_escalate(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("unclear", 0.1, False),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _assessment_response("unclear", 0.1, False),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.iteration == MAX_NO_NEW_INFO


# ---------------------------------------------------------------------------
# 5. Evidence sources do not double-count the same tool
# ---------------------------------------------------------------------------
def test_same_tool_does_not_self_confirm(monkeypatch):
    call_count = {"n": 0}

    def stub_query_logs(**kwargs):
        call_count["n"] += 1
        status = "ok" if call_count["n"] <= 2 else "empty"
        return {"status": status, "data": {}, "summary": "s"}

    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", stub_query_logs)

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.9, True),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.9, True),
        _make_tool_call_response(
            [_tool_call("c3", "query_logs", {"service": "checkout", "window": "3h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.9, True),
        _make_tool_call_response(
            [_tool_call("c4", "query_logs", {"service": "checkout", "window": "4h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.9, True),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.evidence_sources == ["query_logs"]
    assert state.confidence == 0.9
    assert state.status != "resolved"
    assert state.status == "escalated"


# ---------------------------------------------------------------------------
# 6. LLM error path
# ---------------------------------------------------------------------------
def test_llm_error_path_does_not_crash(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module, "call_llm_with_tools", lambda *a, **kw: {"error": "boom"}
    )

    state = run_agent_loop(TICKET_ID)

    assert state.status == "error"


# ---------------------------------------------------------------------------
# 7. Critic returns unparseable JSON twice
# ---------------------------------------------------------------------------
def test_critic_unparseable_json_does_not_crash(monkeypatch):
    responses = [
        _make_text_response("I think it's a db issue but not sure."),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
        _make_text_response("I still think it's a db issue."),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.confidence == 0.0
    assert state.hypothesis is None
    assessment_error_entries = [
        e for e in state.trajectory if e.get("assessment_error")
    ]
    assert len(assessment_error_entries) == 2


# ---------------------------------------------------------------------------
# 8. Unknown ticket id
# ---------------------------------------------------------------------------
def test_unknown_ticket_id_does_not_call_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(
        orchestrator_module,
        "call_llm_with_tools",
        lambda *a, **kw: calls.append(1) or _make_text_response("unused"),
    )

    state = run_agent_loop("NOPE")

    assert state.status == "error"
    assert calls == []


# ---------------------------------------------------------------------------
# 9. Contradiction is surfaced to the model (Fix 1)
# ---------------------------------------------------------------------------
def test_contradiction_is_surfaced(monkeypatch):
    # Updated for FIX 1: added a 4th round crediting search_runbooks so this
    # run can still resolve (round 3 alone only credits query_logs +
    # query_metrics, which no longer clears the evidence bar).
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        # Round 1 — confirming.
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        # Round 2 — contradicting.
        _make_tool_call_response(
            [
                _tool_call(
                    "c2",
                    "query_metrics",
                    {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"},
                )
            ]
        ),
        _assessment_response("network_partition", 0.2, False),
        # Round 3 — confirming again, and reaches the evidence bar so the
        # loop resolves without needing to script a 4th round.
        _make_tool_call_response(
            [
                _tool_call(
                    "c3",
                    "query_metrics",
                    {"service": "checkout", "metric": "db_pool_active_connections", "window": "2h"},
                )
            ]
        ),
        _assessment_response("network_partition", 0.8, True),
        # Round 4 — search_runbooks, clears the FIX 1 grounding requirement.
        _make_tool_call_response(
            [_tool_call("c4", "search_runbooks", {"query": "network partition"})]
        ),
        _assessment_response("network_partition", 0.9, True, citations=[RUNBOOK_DOC_ID]),
    ]

    captured = []

    def fake_call(messages, tools):
        # Snapshot the contents now — `messages` is the same list object
        # reused (and mutated in place) across the whole run, so recording
        # a bare reference to it would make every captured entry equal the
        # final state instead of the state at call time.
        captured.append([m["content"] for m in messages])
        return responses.pop(0)

    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"

    # Call order: r1 act(0), r1 critic(1), r2 act(2), r2 critic(3), r3 act(4), r3 critic(5).
    round2_act_messages = captured[2]  # sent right after round 1 (confirming)
    round3_act_messages = captured[4]  # sent right after round 2 (contradicting)

    def has_contradiction_warning(contents):
        return any(
            "contradict" in (c or "").lower() and "likely wrong" in (c or "").lower()
            for c in contents
        )

    assert not has_contradiction_warning(round2_act_messages)
    assert has_contradiction_warning(round3_act_messages)


# ---------------------------------------------------------------------------
# 10. Unparseable assessment leaves the belief explicitly unchanged (Fix 1)
# ---------------------------------------------------------------------------
def test_unparseable_assessment_note_says_belief_unchanged(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )

    logs_args = {"service": "checkout", "window": "1h", "level": "ERROR"}

    responses = [
        # Round 1 — critic fails twice, so the assessment is None.
        _make_tool_call_response([_tool_call("c1", "query_logs", logs_args)]),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
        # Round 2 — identical tool call, skipped by the loop guard.
        _make_tool_call_response([_tool_call("c2", "query_logs", logs_args)]),
        _assessment_response("unclear", 0.1, True),
        # Round 3 — identical tool call again, skipped again -> escalates.
        _make_tool_call_response([_tool_call("c3", "query_logs", logs_args)]),
        _assessment_response("unclear", 0.1, True),
    ]

    captured = []

    def fake_call(messages, tools):
        captured.append([m["content"] for m in messages])
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"

    # Call order: r1 act(0), r1 critic attempt(1), r1 critic reask(2), r2 act(3), ...
    round2_act_messages = captured[3]
    assert any(
        "could not be parsed" in (c or "").lower() and "unchanged" in (c or "").lower()
        for c in round2_act_messages
    )


# ---------------------------------------------------------------------------
# 11. Backoff fires on the act retry, not on a malformed-JSON critic re-ask (Fix 2)
# ---------------------------------------------------------------------------
def test_backoff_fires_on_act_retry_not_on_critic_reask(monkeypatch):
    # Updated for FIX 1: added a round 3 crediting search_runbooks so the run
    # still resolves (rounds 1-2 alone only credit query_logs/query_metrics).
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    sleep_calls = []
    monkeypatch.setattr(orchestrator_module.time, "sleep", lambda s: sleep_calls.append(s))

    responses = [
        {"error": "boom"},  # round 1 act attempt 1 -> triggers sleep + retry
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),  # round 1 act attempt 2 (retry)
        _make_text_response("not valid json"),  # round 1 critic attempt -> malformed, no sleep
        _assessment_response("db issue", 0.5, True),  # round 1 critic reask
        _make_tool_call_response(
            [
                _tool_call(
                    "c2",
                    "query_metrics",
                    {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"},
                )
            ]
        ),  # round 2 act
        _assessment_response("db issue", 0.8, True),  # round 2 critic
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db issue"})]
        ),  # round 3 act
        _assessment_response("db issue", 0.9, True, citations=[RUNBOOK_DOC_ID]),  # round 3 critic
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert sleep_calls == [orchestrator_module.RETRY_BACKOFF_SECONDS]


# ---------------------------------------------------------------------------
# 12. The critic is called with no tools at all (Fix 1)
# ---------------------------------------------------------------------------
def test_critic_called_with_no_tools(monkeypatch):
    # Updated for FIX 1: added a round 3 crediting search_runbooks so the run
    # still resolves (rounds 1-2 alone only credit query_logs/query_metrics).
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.5, True),
        _make_tool_call_response(
            [
                _tool_call(
                    "c2",
                    "query_metrics",
                    {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"},
                )
            ]
        ),
        _assessment_response("db issue", 0.8, True),
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db issue"})]
        ),
        _assessment_response("db issue", 0.9, True, citations=[RUNBOOK_DOC_ID]),
    ]

    captured_tools = []

    def fake_call(messages, tools):
        captured_tools.append(tools)
        return responses.pop(0)

    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    # Call order: r1 act(0), r1 critic(1), r2 act(2), r2 critic(3).
    assert captured_tools[0]  # act call gets TOOL_SCHEMAS (non-empty)
    assert not captured_tools[1]  # critic call gets no tools
    assert captured_tools[2]
    assert not captured_tools[3]


# ---------------------------------------------------------------------------
# 13. The critic context is isolated from the main tool-calling conversation
# ---------------------------------------------------------------------------
def test_critic_context_is_isolated(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("unclear", 0.1, False),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _assessment_response("unclear", 0.1, False),
    ]

    critic_message_lists = []

    def fake_call(messages, tools):
        if not tools:
            critic_message_lists.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert len(critic_message_lists) == 2
    for msgs in critic_message_lists:
        assert len(msgs) == 2
        assert all(m.get("role") != "tool" for m in msgs)
        assert all("tool_calls" not in m for m in msgs)


# ---------------------------------------------------------------------------
# 14. The digest carries prior observations forward into later rounds
# ---------------------------------------------------------------------------
def test_digest_carries_prior_observation_summary(monkeypatch):
    # Updated for FIX 1: added a round 3 crediting search_runbooks so the run
    # still resolves (rounds 1-2 alone only credit query_logs/query_metrics).
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY,
        "query_logs",
        _stub_tool(status="ok", summary="checkout-db connection errors spiking"),
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response(
            [
                _tool_call(
                    "c2",
                    "query_metrics",
                    {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"},
                )
            ]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.8, True),
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]

    critic_user_contents = []
    compose_user_contents = []

    def fake_call(messages, tools):
        if not tools:
            # The write-gate's compose call also passes tools=[], so it lands
            # in this same "no tools" branch as a critic pass would -- tell
            # them apart by content rather than merging them into one count,
            # which would hide a real critic-loop regression (an extra
            # critic round firing) behind "the write gate must have fired".
            content = messages[-1]["content"] or ""
            if orchestrator_module.WRITE_COMPOSE_INSTRUCTION in content:
                compose_user_contents.append(content)
            else:
                critic_user_contents.append(content)
        return responses.pop(0)

    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert len(critic_user_contents) == 3
    assert len(compose_user_contents) == 1
    # Round 2's critic digest still carries round 1's summary forward,
    # sourced from state.trajectory rather than the (now unused) main
    # tool-calling conversation.
    assert "checkout-db connection errors spiking" in critic_user_contents[1]


# ---------------------------------------------------------------------------
# 15. Regression: a failed critic round no longer permanently loses that
# round's evidence (the observed T001 failure this fix addresses) (Fix 2)
# ---------------------------------------------------------------------------
def test_evidence_credited_across_a_failed_critic_round(monkeypatch):
    # Updated for FIX 1: added a round 3 crediting search_runbooks so the run
    # still resolves (rounds 1-2 alone only credit query_logs/query_metrics).
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        # Round 1 — query_logs ok, critic fails both the ask and the re-ask.
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
        # Round 2 — query_metrics ok, critic supports at high confidence.
        _make_tool_call_response(
            [
                _tool_call(
                    "c2",
                    "query_metrics",
                    {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"},
                )
            ]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.8, True),
        # Round 3 — search_runbooks ok, clears the FIX 1 grounding requirement.
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    # Under the old behavior, evidence was only credited for tools in the
    # CURRENT round, so query_logs (round 1, assessment_error) was never
    # credited and T001 escalated at the iteration cap despite a correct
    # hypothesis and 0.94 confidence. It must now be credited retroactively
    # once a later round's critic supports the hypothesis.
    assert state.evidence_sources == ["query_logs", "query_metrics", "search_runbooks"]
    assert state.status == "resolved"


# ---------------------------------------------------------------------------
# 16. supports: false credits nothing, even for an "ok" tool result (Fix 2)
# ---------------------------------------------------------------------------
def test_supports_false_credits_nothing_even_when_tool_ok(monkeypatch):
    call_count = {"n": 0}

    def spy_query_logs(**kwargs):
        call_count["n"] += 1
        return {"status": "ok", "data": {}, "summary": "logs"}

    monkeypatch.setitem(tool_executor_module.TOOL_REGISTRY, "query_logs", spy_query_logs)

    args = {"service": "checkout", "window": "2h", "level": "ERROR"}
    repeated_call = lambda cid: _tool_call(cid, "query_logs", args)

    responses = [
        _make_tool_call_response([repeated_call("c1")]),
        _assessment_response("unclear", 0.2, False),
        _make_tool_call_response([repeated_call("c2")]),
        _assessment_response("unclear", 0.2, False),
        _make_tool_call_response([repeated_call("c3")]),
        _assessment_response("unclear", 0.2, False),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert call_count["n"] == 1
    assert state.evidence_sources == []


# ---------------------------------------------------------------------------
# 17. A failed assessment credits nothing, even for an "ok" tool result. If
# no round's critic ever supports the hypothesis, the "ok" tool result is
# never credited, retroactively or otherwise (Fix 2)
# ---------------------------------------------------------------------------
def test_failed_assessment_credits_nothing(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )

    args = {"service": "checkout", "window": "2h", "level": "ERROR"}
    responses = [
        # Round 1 — ok tool, critic fails both ask and re-ask.
        _make_tool_call_response([_tool_call("c1", "query_logs", args)]),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
        # Round 2 — identical call, skipped by the loop guard; critic fails
        # again.
        _make_tool_call_response([_tool_call("c2", "query_logs", args)]),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
        # Round 3 — identical call, skipped again; critic fails again ->
        # escalates via the no-new-info guard.
        _make_tool_call_response([_tool_call("c3", "query_logs", args)]),
        _make_text_response("not valid json"),
        _make_text_response("still not valid json"),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.evidence_sources == []


# ---------------------------------------------------------------------------
# 18. _credit_evidence helper: dedupe, first-seen order, "ok"-only, skips
# entries with no tool_call (Fix 2)
# ---------------------------------------------------------------------------
def test_credit_evidence_helper_dedupes_and_preserves_order():
    state = TaskState(ticket_id=TICKET_ID, description="d")
    state.trajectory = [
        {
            "iteration": 0,
            "tool_call": {"name": "query_logs", "arguments": {}},
            "observation": {"status": "ok"},
        },
        {"iteration": 0, "tool_call": None, "observation": {"text": "..."}},
        {
            "iteration": 1,
            "tool_call": {"name": "query_metrics", "arguments": {}},
            "observation": {"status": "empty"},
        },
    ]
    round_entries = [
        {"tool_name": "query_metrics", "status": "ok", "observation": {}},
        {"tool_name": "query_logs", "status": "ok", "observation": {}},
    ]

    orchestrator_module._credit_evidence(state, round_entries)

    assert state.evidence_sources == ["query_logs", "query_metrics"]


# ---------------------------------------------------------------------------
# 19. The critic deadlock fix: in a tool-calling round, the act model's text
# is passed to the critic as answer_text so it has something to assess.
# ---------------------------------------------------------------------------
def test_critic_receives_act_text_in_tool_calling_round(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )

    def _tool_call_response_with_text(text, tool_calls):
        return _StubCompletion(_StubMessage(content=text, tool_calls=tool_calls))

    args = {"service": "checkout", "window": "1h", "level": "ERROR"}
    repeated_call = lambda cid: _tool_call(cid, "query_logs", args)

    responses = [
        _tool_call_response_with_text(
            "I suspect a db connection pool exhaustion; checking logs.",
            [repeated_call("c1")],
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.3, True),
        # Repeat the same call twice more; the loop guard skips both without
        # consuming a real tool result, so no_new_info climbs to the cap and
        # the loop escalates without needing to script confidence to 0.75.
        _tool_call_response_with_text("still checking", [repeated_call("c2")]),
        _assessment_response("db_connection_pool_exhaustion", 0.3, True),
        _tool_call_response_with_text("still checking", [repeated_call("c3")]),
        _assessment_response("db_connection_pool_exhaustion", 0.3, True),
    ]

    critic_digests = []

    def fake_call(messages, tools):
        if not tools:
            critic_digests.append(messages[-1]["content"])
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    orchestrator_module.run_agent_loop(TICKET_ID)

    assert len(critic_digests) >= 1
    assert "db connection pool exhaustion" in critic_digests[0].lower()
    assert "current hypothesis: none yet" in critic_digests[0].lower()


# ---------------------------------------------------------------------------
# 20. The critic deadlock fix: a critic that returns supports=false,
# confidence=0.0 with a self-referential "no hypothesis" string does not
# make the loop reuse that same string as if it were fresh evidence in the
# next round's digest -- each round's digest is rebuilt from the current
# act text, not frozen on the poisoned hypothesis.
# ---------------------------------------------------------------------------
def test_deadlock_hypothesis_does_not_persist_as_stale_evidence(monkeypatch):
    # status="empty" (not "ok") so each round counts toward no_new_info and
    # the loop escalates after exactly 2 rounds -- keeping this test's
    # scripted response list short and independent of the iteration cap.
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    def _tool_call_response_with_text(text, tool_calls):
        return _StubCompletion(_StubMessage(content=text, tool_calls=tool_calls))

    responses = [
        _tool_call_response_with_text(
            "round one reasoning about db pool",
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})],
        ),
        _assessment_response("No hypothesis defined yet", 0.0, False),
        _tool_call_response_with_text(
            "round two reasoning about disk io",
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})],
        ),
        _assessment_response("No hypothesis defined yet", 0.0, False),
    ]

    critic_digests = []

    def fake_call(messages, tools):
        if not tools:
            critic_digests.append(messages[-1]["content"])
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = orchestrator_module.run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert len(critic_digests) == 2
    # Each round's digest carries THAT round's own act text as the stated
    # answer, not a copy of the previous round's poisoned hypothesis string
    # being fed back in as if it were new evidence.
    assert "round one reasoning about db pool" in critic_digests[0]
    assert "round two reasoning about disk io" in critic_digests[1]
    assert "round one reasoning" not in critic_digests[1]


# ---------------------------------------------------------------------------
# 21. Evidence-credit guard (Fix 2): retrieval-only evidence (search_runbooks
# + search_past_incidents) meets confidence/count but must NOT resolve.
# ---------------------------------------------------------------------------
def test_retrieval_only_evidence_does_not_resolve(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_past_incidents", _stub_tool(status="ok")
    )

    responses = []
    for i in range(1, 9):
        if i % 2 == 1:
            tc = _tool_call(f"c{i}", "search_runbooks", {"query": f"q{i}"})
        else:
            tc = _tool_call(f"c{i}", "search_past_incidents", {"query": f"q{i}"})
        responses.append(_make_tool_call_response([tc]))
        responses.append(_assessment_response("db_connection_pool_exhaustion", 0.9, True))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.iteration == state.max_iterations
    assert set(state.evidence_sources) == {"search_runbooks", "search_past_incidents"}


# ---------------------------------------------------------------------------
# 22. Evidence-credit guard (Fix 2): one observational source plus one
# retrieval source is enough -- the guard requires only ONE observational
# source, not two.
# ---------------------------------------------------------------------------
def test_one_observational_plus_one_retrieval_source_resolves(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response(
            [_tool_call("c2", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert set(state.evidence_sources) == {"query_logs", "search_runbooks"}


# ---------------------------------------------------------------------------
# 23. Critic-inference fix: CRITIC_INSTRUCTION must actually tell the model
# to infer a hypothesis rather than refuse when none is stated yet.
# ---------------------------------------------------------------------------
def test_critic_instruction_requires_inference_when_no_hypothesis():
    text = orchestrator_module.CRITIC_INSTRUCTION
    assert "infer the most likely root cause" in text
    assert "No hypothesis defined yet" in text
    assert "do not answer that there is no hypothesis" in text


# ---------------------------------------------------------------------------
# 24. Critic-inference fix: a real inferred hypothesis on the first
# tool-calling round (no prior hypothesis) is accepted, credited, and the
# run can resolve given one observational source and sufficient confidence.
# ---------------------------------------------------------------------------
def test_critic_infers_hypothesis_on_first_tool_calling_round(monkeypatch):
    # Updated for FIX 1: added a round 3 crediting search_runbooks so the run
    # still resolves (rounds 1-2 alone only credit query_logs/query_metrics).
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        # First tool-calling round: no prior hypothesis exists yet, so the
        # critic must infer one from the observation rather than refuse.
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response(
            [_tool_call("c2", "query_metrics", {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.8, True),
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.hypothesis == "db_connection_pool_exhaustion"
    assert state.evidence_sources == ["query_logs", "query_metrics", "search_runbooks"]
    assert state.status == "resolved"
    assert state.confidence >= CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# 25. Critic-inference fix: a stale placeholder hypothesis already sitting in
# state must not block a later round's critic from replacing it with a real,
# inferred one.
# ---------------------------------------------------------------------------
def test_placeholder_hypothesis_is_replaced_by_later_inference(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("No hypothesis defined yet", 0.0, False),
        _make_tool_call_response(
            [_tool_call("c2", "query_metrics", {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.5, True),
        _make_tool_call_response(
            [_tool_call("c3", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.hypothesis == "db_connection_pool_exhaustion"
    assert state.status == "resolved"
    assert "query_metrics" in state.evidence_sources


# ---------------------------------------------------------------------------
# 26. _can_resolve now requires a credited search_runbooks source (Fix 1)
# ---------------------------------------------------------------------------
def _searched_runbook_entry(doc_id=RUNBOOK_DOC_ID):
    # A state.trajectory entry recording a credited (status="ok")
    # search_runbooks call that actually returned doc_id -- what
    # _observed_runbook_doc_ids scans for.
    return {
        "iteration": 0,
        "tool_call": {"name": "search_runbooks", "arguments": {"query": "q"}},
        "observation": {
            "status": "ok",
            "data": {"chunks": [{"doc_id": doc_id, "section": "s", "text": "t", "score": 0.9}]},
            "summary": "s",
        },
    }


def test_can_resolve_requires_credited_search_runbooks():
    # The exact live-observed T001/T009 case: confidence 0.95, two credited
    # observational sources, but search_runbooks never called. Must NOT
    # resolve under the new grounding requirement.
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.confidence = 0.95
    state.evidence_sources = ["query_logs", "query_metrics"]
    assert orchestrator_module._can_resolve(state) is False

    # Adding search_runbooks to the credited sources, a matching observed
    # doc_id in the trajectory, and a citation for it flips it to resolvable.
    state.evidence_sources = ["query_logs", "query_metrics", "search_runbooks"]
    state.trajectory = [_searched_runbook_entry()]
    state.citations = [RUNBOOK_DOC_ID]
    assert orchestrator_module._can_resolve(state) is True


def test_can_resolve_still_requires_confidence_and_source_count(monkeypatch):
    # Sanity: the new requirement is additive, not a replacement for the
    # existing bars.
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.confidence = 0.95
    state.evidence_sources = ["search_runbooks"]
    state.trajectory = [_searched_runbook_entry()]
    state.citations = [RUNBOOK_DOC_ID]
    assert orchestrator_module._can_resolve(state) is False  # no observational source

    state.confidence = 0.5
    state.evidence_sources = ["query_logs", "search_runbooks"]
    assert orchestrator_module._can_resolve(state) is False  # confidence too low


# ---------------------------------------------------------------------------
# 30. Citation verification gate (F7): _can_resolve additionally requires at
# least one citation, and every citation must be a doc_id the run actually
# observed via a status="ok" search_runbooks result -- a hallucinated or
# misremembered doc_id must block the resolve, not be ignored.
# ---------------------------------------------------------------------------
def _fully_evidenced_state(**overrides):
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.confidence = 0.95
    state.evidence_sources = ["query_logs", "search_runbooks"]
    state.trajectory = [_searched_runbook_entry()]
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_can_resolve_requires_at_least_one_citation():
    state = _fully_evidenced_state(citations=[])
    assert orchestrator_module._can_resolve(state) is False


def test_can_resolve_blocks_a_hallucinated_citation():
    state = _fully_evidenced_state(citations=["RB-FAKE-999"])
    assert orchestrator_module._can_resolve(state) is False


def test_can_resolve_true_when_citation_was_actually_observed():
    state = _fully_evidenced_state(citations=[RUNBOOK_DOC_ID])
    assert orchestrator_module._can_resolve(state) is True


def test_observed_runbook_doc_ids_helper():
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.trajectory = [_searched_runbook_entry("RB-NET-002")]
    round_entries = [
        {
            "tool_name": "search_runbooks",
            "status": "ok",
            "observation": {
                "status": "ok",
                "data": {"chunks": [{"doc_id": "RB-DB-003"}]},
            },
        }
    ]
    observed = orchestrator_module._observed_runbook_doc_ids(state, round_entries)
    assert observed == {"RB-NET-002", "RB-DB-003"}


# ---------------------------------------------------------------------------
# 27. A no_confident_match runbook result can never be credited, so a run
# with strong logs/metrics and high confidence still escalates rather than
# fabricate a fix from a bad match (Fix 1, T015 regression pin).
# ---------------------------------------------------------------------------
def test_no_confident_runbook_match_escalates_despite_strong_evidence(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY,
        "search_runbooks",
        _stub_tool(status="no_confident_match"),
    )

    def make_round(call_id, name, args):
        return _make_tool_call_response([_tool_call(call_id, name, args)])

    responses = [
        make_round("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"}),
        _assessment_response("db_connection_pool_exhaustion", 0.9, True),
        make_round(
            "c2",
            "query_metrics",
            {"service": "checkout", "metric": "db_pool_active_connections", "window": "1h"},
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.95, True),
        make_round("c3", "search_runbooks", {"query": "db pool exhaustion"}),
        _assessment_response("db_connection_pool_exhaustion", 0.97, True),
    ]
    # Pad enough rounds to exhaust the iteration cap: the model just keeps
    # trying (skipped/repeated calls collapse via the loop guard, so vary
    # the window arg each round to avoid tripping it).
    while len(responses) < 2 * TaskState(ticket_id=TICKET_ID, description="x").max_iterations:
        i = len(responses) // 2 + 1
        responses.append(
            make_round(
                f"c{i}",
                "query_metrics",
                {"service": "checkout", "metric": "db_pool_active_connections", "window": f"{i}h"},
            )
        )
        responses.append(_assessment_response("db_connection_pool_exhaustion", 0.97, True))

    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert "search_runbooks" not in state.evidence_sources
    assert state.status == "escalated"
    assert state.confidence >= CONFIDENCE_THRESHOLD  # strong evidence, still escalates


# ---------------------------------------------------------------------------
# 28. query_logs + search_runbooks at threshold DOES resolve — the new
# requirement doesn't accidentally demand all four tools (Fix 1).
# ---------------------------------------------------------------------------
def test_logs_and_runbook_evidence_resolves(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool()
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.6, True),
        _make_tool_call_response(
            [_tool_call("c2", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert set(state.evidence_sources) == {"query_logs", "search_runbooks"}


# ---------------------------------------------------------------------------
# 29. CRITIC_INSTRUCTION text (Fix 2): absence of evidence must not raise
# confidence, and mechanism-free hypotheses must receive low confidence.
# ---------------------------------------------------------------------------
def test_critic_instruction_treats_absence_of_evidence_as_not_support():
    instruction = orchestrator_module.CRITIC_INSTRUCTION
    assert "absence of evidence" in instruction
    assert "empty" in instruction
    assert "must lower confidence" in instruction


def test_critic_instruction_requires_concrete_mechanism_for_high_confidence():
    instruction = orchestrator_module.CRITIC_INSTRUCTION
    assert "concrete, checkable mechanism" in instruction
    assert "LOW confidence" in instruction
    assert "positively identify a specific failure mode" in instruction


# ---------------------------------------------------------------------------
# 31. CRITIC_INSTRUCTION now also asks for citations, matching the Assessment
# schema (F7).
# ---------------------------------------------------------------------------
def test_critic_instruction_requires_citations_matching_schema():
    instruction = orchestrator_module.CRITIC_INSTRUCTION
    assert '"citations": list of runbook doc_id strings' in instruction
    assert "never invented" in instruction


# ---------------------------------------------------------------------------
# 32. Assessment schema (F7): citations defaults to [] when the critic's
# reply omits it (older/looser stubbed replies must still parse), and is
# carried through when present.
# ---------------------------------------------------------------------------
def test_assessment_parses_with_citations_absent():
    assessment = orchestrator_module.Assessment.model_validate(
        {"hypothesis": "h", "confidence": 0.5, "supports": True, "reasoning": "r"}
    )
    assert assessment.citations == []


def test_assessment_parses_with_citations_present():
    assessment = orchestrator_module.Assessment.model_validate(
        {
            "hypothesis": "h",
            "confidence": 0.5,
            "supports": True,
            "reasoning": "r",
            "citations": ["RB-DB-001"],
        }
    )
    assert assessment.citations == ["RB-DB-001"]


# ---------------------------------------------------------------------------
# 33. Digest truncation (F7): a realistic search_runbooks observation, whose
# chunk "text" fields alone exceed the 500-char truncation, still surfaces
# doc_ids to the critic. Pins the _format_observation fix -- regressing it
# (e.g. reverting to a bare truncated data blob) makes this fail.
# ---------------------------------------------------------------------------
def test_digest_surfaces_doc_ids_despite_500_char_truncation():
    state = TaskState(ticket_id=TICKET_ID, description="d")
    long_text = "This runbook section describes the diagnosis procedure. " * 20
    round_entries = [
        {
            "tool_name": "search_runbooks",
            "tool_args": {"query": "db pool exhaustion"},
            "status": "ok",
            "observation": {
                "status": "ok",
                "summary": "top match",
                "data": {
                    "query": "db pool exhaustion",
                    "top_score": 0.91,
                    "chunks": [
                        {
                            "doc_id": "RB-DB-001",
                            "section": "diagnosis",
                            "text": long_text,
                            "score": 0.91,
                        },
                        {
                            "doc_id": "RB-DB-002",
                            "section": "mitigation",
                            "text": long_text,
                            "score": 0.6,
                        },
                    ],
                },
            },
        }
    ]

    digest = orchestrator_module._build_critic_digest(state, round_entries)

    # Confirm this test is actually exercising the truncation boundary: the
    # untruncated data blob for this observation must exceed 500 chars.
    raw_data_str = json.dumps(
        round_entries[0]["observation"]["data"], sort_keys=True, separators=(",", ":")
    )
    assert len(raw_data_str) > 500

    assert "RB-DB-001" in digest
    assert "RB-DB-002" in digest
    assert "doc_ids=" in digest


# ---------------------------------------------------------------------------
# 34. Loop-level (F7): a run whose critic cites a doc_id the run's
# search_runbooks observation actually returned resolves, and state.citations
# reflects it.
# ---------------------------------------------------------------------------
def test_loop_resolves_when_citation_matches_observed_doc_id(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool("RB-DB-001")
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db_connection_pool_exhaustion", 0.6, True),
        _make_tool_call_response(
            [_tool_call("c2", "search_runbooks", {"query": "db pool exhaustion"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=["RB-DB-001"]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert state.citations == ["RB-DB-001"]


# ---------------------------------------------------------------------------
# 35. Loop-level (F7): a run whose critic cites a doc_id that was never
# returned this run escalates instead of resolving -- the fabrication guard.
# ---------------------------------------------------------------------------
def test_loop_escalates_when_citation_is_fabricated(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _stub_runbook_tool("RB-DB-001")
    )

    responses = []
    for i in range(1, 9):
        if i % 2 == 1:
            tc = _tool_call(f"c{i}", "query_logs", {"service": "checkout", "window": f"{i}h", "level": "ERROR"})
        else:
            tc = _tool_call(f"c{i}", "search_runbooks", {"query": f"q{i}"})
        responses.append(_make_tool_call_response([tc]))
        responses.append(
            _assessment_response(
                "db_connection_pool_exhaustion", 0.9, True, citations=["RB-FAKE-999"]
            )
        )
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.citations == ["RB-FAKE-999"]


# ---------------------------------------------------------------------------
# 36. F7 citation-accumulation fix: `state.citations = assessment.citations`
# used to be an unconditional overwrite, so a critic reply that simply
# omitted "citations" (Assessment.citations defaults to []) silently wiped
# out a citation a previous round had correctly established. These tests pin
# the accumulate-then-retract replacement (_merge_citations). See
# tests/fixtures/citation_regression/README.md for the live replay evidence.
# ---------------------------------------------------------------------------
def _raw_json_response(payload):
    return _make_text_response(json.dumps(payload))


def test_loop_omitted_citations_key_does_not_wipe_prior_citation(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    round1_payload = {
        "hypothesis": "disk full",
        "confidence": 0.6,
        "supports": True,
        "reasoning": "because",
        "citations": ["RB-DISK-001"],
    }
    round2_payload = {
        # "citations" key entirely absent -- exercises the Pydantic default
        # path, not an explicit empty list.
        "hypothesis": "disk full",
        "confidence": 0.6,
        "supports": True,
        "reasoning": "still because",
    }
    assert "citations" not in round2_payload

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _raw_json_response(round1_payload),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _raw_json_response(round2_payload),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.citations == ["RB-DISK-001"]


def test_loop_explicit_empty_citations_does_not_wipe_prior_citation(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("disk full", 0.6, True, citations=["RB-DISK-001"]),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _assessment_response("disk full", 0.6, True, citations=[]),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.citations == ["RB-DISK-001"]


def test_loop_retracted_citation_is_removed(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    round2_payload = {
        "hypothesis": "actually something else",
        "confidence": 0.4,
        "supports": False,
        "reasoning": "the db runbook no longer fits",
        "citations": [],
        "retracted_citations": ["RB-DB-001"],
    }

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.6, True, citations=["RB-DB-001"]),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _raw_json_response(round2_payload),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.citations == []


def test_loop_retraction_wins_over_citation_in_same_round(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="empty")
    )

    round1_payload = {
        "hypothesis": "db issue",
        "confidence": 0.6,
        "supports": True,
        "reasoning": "because",
        "citations": ["RB-DB-001"],
        "retracted_citations": ["RB-DB-001"],
    }

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _raw_json_response(round1_payload),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "2h", "level": "ERROR"})]
        ),
        _assessment_response("db issue", 0.6, True),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "escalated"
    assert state.citations == []


# ---------------------------------------------------------------------------
# 37. _merge_citations unit tests -- the helper both call sites share.
# ---------------------------------------------------------------------------
def test_merge_citations_omitted_new_citations_keeps_existing():
    merged = orchestrator_module._merge_citations(["RB-DISK-001"], [], [])
    assert merged == ["RB-DISK-001"]


def test_merge_citations_adds_new_without_dropping_existing():
    merged = orchestrator_module._merge_citations(["RB-A"], ["RB-B"], [])
    assert merged == ["RB-A", "RB-B"]


def test_merge_citations_reaffirmed_doc_id_moves_to_end_without_duplicating():
    # Reaffirming RB-A must not create a second copy, AND must move it to
    # the end -- the last element is what the critic most recently stood
    # behind.
    merged = orchestrator_module._merge_citations(["RB-A", "RB-B"], ["RB-A"], [])
    assert merged == ["RB-B", "RB-A"]


def test_merge_citations_retracts_existing():
    merged = orchestrator_module._merge_citations(["RB-A", "RB-B"], [], ["RB-A"])
    assert merged == ["RB-B"]


def test_merge_citations_retraction_wins_over_citation_same_round():
    merged = orchestrator_module._merge_citations([], ["RB-A"], ["RB-A"])
    assert merged == []


def test_merge_citations_orders_by_most_recent_affirmation():
    # Round 1 cites RB-A, round 2 cites RB-B, round 3 reaffirms RB-A -- the
    # LAST element must always be whatever was most recently affirmed,
    # since _queue_write_action grounds the write in citations[-1]: the
    # write must be tied to the critic's latest judgement about what
    # supports the CURRENT hypothesis, not to whichever doc_id happened to
    # be cited first (which may belong to an abandoned hypothesis).
    state_citations = []
    state_citations = orchestrator_module._merge_citations(state_citations, ["RB-A"], [])
    assert state_citations == ["RB-A"]
    assert state_citations[-1] == "RB-A"

    state_citations = orchestrator_module._merge_citations(state_citations, ["RB-B"], [])
    assert state_citations == ["RB-A", "RB-B"]
    assert state_citations[-1] == "RB-B"

    state_citations = orchestrator_module._merge_citations(state_citations, ["RB-A"], [])
    assert state_citations == ["RB-B", "RB-A"]
    assert state_citations[-1] == "RB-A"


# ---------------------------------------------------------------------------
# 38. Assessment schema (F7): retracted_citations defaults to [] and is
# carried through when present.
# ---------------------------------------------------------------------------
def test_assessment_parses_with_retracted_citations_absent():
    assessment = orchestrator_module.Assessment.model_validate(
        {"hypothesis": "h", "confidence": 0.5, "supports": True, "reasoning": "r"}
    )
    assert assessment.retracted_citations == []


def test_assessment_parses_with_retracted_citations_present():
    assessment = orchestrator_module.Assessment.model_validate(
        {
            "hypothesis": "h",
            "confidence": 0.5,
            "supports": True,
            "reasoning": "r",
            "retracted_citations": ["RB-DB-001"],
        }
    )
    assert assessment.retracted_citations == ["RB-DB-001"]


def test_critic_instruction_documents_retracted_citations():
    instruction = orchestrator_module.CRITIC_INSTRUCTION
    assert "retracted_citations" in instruction
    assert "does NOT retract" in instruction


# ---------------------------------------------------------------------------
# 39. Superseded-hypothesis citation fix: _queue_write_action must ground the
# write in the MOST RECENTLY AFFIRMED citation (citations[-1]), not
# citations[0] -- an older citation left over from an abandoned hypothesis
# must never end up backing the write.
# ---------------------------------------------------------------------------
def test_queue_write_action_uses_most_recently_affirmed_citation(monkeypatch):
    # RB-DISK-001 was cited first (and would belong to an earlier, now
    # abandoned hypothesis); RB-DB-001 (RUNBOOK_DOC_ID) was affirmed most
    # recently and is last in state.citations. The write must be grounded in
    # RB-DB-001, not RB-DISK-001.
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.hypothesis = "db_connection_pool_exhaustion"
    state.citations = ["RB-DISK-001", RUNBOOK_DOC_ID]

    captured_update_ticket_kwargs = {}

    def _fake_update_ticket(**kwargs):
        captured_update_ticket_kwargs.update(kwargs)
        return {
            "status": "awaiting_approval",
            "data": {"action_id": "a1"},
            "summary": "queued",
        }

    monkeypatch.setattr(orchestrator_module, "update_ticket", _fake_update_ticket)

    captured_compose_messages = []

    def _fake_call_llm_with_tools(messages, tools):
        captured_compose_messages.append(messages)
        return _make_text_response(WRITE_GATE_FIX_TEXT)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", _fake_call_llm_with_tools)

    orchestrator_module._queue_write_action(state, [])

    assert captured_update_ticket_kwargs["citation_doc_id"] == RUNBOOK_DOC_ID
    injected_content = captured_compose_messages[0][-1]["content"]
    assert RUNBOOK_DOC_ID in injected_content
    assert "RB-DISK-001" not in injected_content


def test_replan_grounds_write_in_new_hypothesis_citation(monkeypatch):
    # Round 1 cites RB-DISK-001 under one hypothesis; round 2 replans to a
    # different hypothesis and cites RUNBOOK_DOC_ID (RB-DB-001). Even though
    # RB-DISK-001 is never explicitly retracted, the write must still be
    # grounded in RB-DB-001 -- the citation supporting the CURRENT
    # hypothesis -- not the stale one from the abandoned hypothesis.
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )

    # Returns chunks for BOTH doc_ids so each is a valid _can_resolve
    # citation regardless of which round cites it -- this test is about
    # write-grounding order, not about the observed-doc_id gate.
    def _search_runbooks_stub(**kwargs):
        return {
            "status": "ok",
            "data": {
                "chunks": [
                    {"doc_id": "RB-DISK-001", "section": "diagnosis", "text": "t", "score": 0.9},
                    {"doc_id": RUNBOOK_DOC_ID, "section": "diagnosis", "text": "t", "score": 0.9},
                ]
            },
            "summary": "stub",
        }

    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "search_runbooks", _search_runbooks_stub
    )

    responses = [
        _make_tool_call_response(
            [_tool_call("c1", "search_runbooks", {"query": "disk full"})]
        ),
        _assessment_response("disk full", 0.6, True, citations=["RB-DISK-001"]),
        _make_tool_call_response(
            [_tool_call("c2", "query_logs", {"service": "checkout", "window": "1h", "level": "ERROR"})]
        ),
        _assessment_response(
            "db_connection_pool_exhaustion", 0.9, True, citations=[RUNBOOK_DOC_ID]
        ),
    ]
    responses.append(_make_text_response(WRITE_GATE_FIX_TEXT))
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert state.citations[-1] == RUNBOOK_DOC_ID
    write_entry = next(
        entry
        for entry in state.trajectory
        if entry.get("tool_call") and entry["tool_call"].get("name") == "update_ticket"
    )
    assert write_entry["tool_call"]["arguments"]["citation_doc_id"] == RUNBOOK_DOC_ID


# ---------------------------------------------------------------------------
# 40. _build_critic_digest surfaces state.citations so the critic can
# actually retract a stale one.
# ---------------------------------------------------------------------------
def test_critic_digest_includes_previously_cited_section():
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.citations = ["RB-DISK-001"]

    digest = orchestrator_module._build_critic_digest(state, [])

    assert "Previously cited" in digest
    assert "RB-DISK-001" in digest


def test_critic_digest_previously_cited_section_empty_case():
    state = TaskState(ticket_id=TICKET_ID, description="x")
    state.citations = []

    digest = orchestrator_module._build_critic_digest(state, [])

    assert "Previously cited" in digest
    assert "(none yet)" in digest


# ---------------------------------------------------------------------------
# 39. Regression fixtures (tests/fixtures/citation_regression/): the digest
# built from each trimmed real-sweep capture still surfaces the runbook
# doc_id the critic should cite, guarding the F7 hoisting that makes doc_id
# visible in the digest at all.
# ---------------------------------------------------------------------------
_FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "citation_regression"
)


@pytest.mark.parametrize("ticket_fixture", ["T009", "T024", "T038"])
def test_citation_regression_fixture_digest_surfaces_expected_doc_id(ticket_fixture):
    with open(os.path.join(_FIXTURES_DIR, f"{ticket_fixture}.json")) as f:
        fixture = json.load(f)

    state = TaskState(
        ticket_id=fixture["ticket_id"],
        description=fixture["description"],
        hypothesis=fixture["hypothesis"],
        trajectory=fixture["trajectory"],
    )

    digest = orchestrator_module._build_critic_digest(state, [])

    assert f'doc_ids=[\'{fixture["expected_doc_id"]}\'' in digest or fixture[
        "expected_doc_id"
    ] in digest


# ---------------------------------------------------------------------------
# 40. Live replay (F7): rebuilds each citation_regression fixture's digest
# and calls the REAL critic, asserting it cites the expected runbook doc_id.
# Costs real tokens -- deselected by default (see pytest.ini's `live`
# marker). This is the replay evidence documented in
# tests/fixtures/citation_regression/README.md: replaying the exact final
# digests of T009/T024/T038 against the live critic returns the correct
# doc_id 9/9 times, confirming these tickets falsely escalated purely from
# the state-management bug (the unconditional citations overwrite) and not
# from any weakness in the critic's judgment or retrieval.
# ---------------------------------------------------------------------------
@pytest.mark.live
@pytest.mark.parametrize("ticket_fixture", ["T009", "T024", "T038"])
def test_citation_regression_fixture_live_critic_cites_expected_doc_id(ticket_fixture):
    """Hits the real Groq API -- costs real API calls, requires GROQ_API_KEY.

    Deselected by default via pytest.ini; run explicitly with `pytest -m live`.
    """
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("LLM_API_KEY"):
        pytest.skip("GROQ_API_KEY/LLM_API_KEY not set")

    with open(os.path.join(_FIXTURES_DIR, f"{ticket_fixture}.json")) as f:
        fixture = json.load(f)

    state = TaskState(
        ticket_id=fixture["ticket_id"],
        description=fixture["description"],
        hypothesis=fixture["hypothesis"],
        trajectory=fixture["trajectory"],
    )

    assessment, errored = orchestrator_module._run_critic(state, [])

    assert not errored
    assert assessment is not None
    assert fixture["expected_doc_id"] in assessment.citations
