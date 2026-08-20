"""Tests for agent/orchestrator.py (the full plan->act->observe->replan->verify
agent loop).

Fully offline and deterministic: every test monkeypatches
agent.orchestrator.call_llm_with_tools, so no network call is ever made. No
`live` marker is used here.
"""

from __future__ import annotations

import json

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


def _assessment_response(hypothesis, confidence, supports, reasoning="because"):
    return _make_text_response(
        json.dumps(
            {
                "hypothesis": hypothesis,
                "confidence": confidence,
                "supports": supports,
                "reasoning": reasoning,
            }
        )
    )


def _stub_tool(status="ok", summary="stub"):
    def fn(**kwargs):
        return {"status": status, "data": {}, "summary": summary}

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
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
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
        _assessment_response("db_connection_pool_exhaustion", 0.9, True),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert state.confidence == 0.9
    assert state.hypothesis == "db_connection_pool_exhaustion"
    assert state.evidence_sources == ["query_logs", "query_metrics"]
    assert state.iteration == 2


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
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
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
        _assessment_response("network_partition", 0.9, True),
    ]

    captured = []

    def fake_call(messages, tools):
        # Snapshot the contents now — `messages` is the same list object
        # reused (and mutated in place) across the whole run, so recording
        # a bare reference to it would make every captured entry equal the
        # final state instead of the state at call time.
        captured.append([m["content"] for m in messages])
        return responses.pop(0)

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
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
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
        _assessment_response("db issue", 0.9, True),  # round 2 critic
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert sleep_calls == [orchestrator_module.RETRY_BACKOFF_SECONDS]


# ---------------------------------------------------------------------------
# 12. The critic is called with no tools at all (Fix 1)
# ---------------------------------------------------------------------------
def test_critic_called_with_no_tools(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
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
        _assessment_response("db issue", 0.9, True),
    ]

    captured_tools = []

    def fake_call(messages, tools):
        captured_tools.append(tools)
        return responses.pop(0)

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
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY,
        "query_logs",
        _stub_tool(status="ok", summary="checkout-db connection errors spiking"),
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
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
        _assessment_response("db_connection_pool_exhaustion", 0.9, True),
    ]

    critic_user_contents = []

    def fake_call(messages, tools):
        if not tools:
            critic_user_contents.append(messages[-1]["content"])
        return responses.pop(0)

    monkeypatch.setattr(orchestrator_module, "call_llm_with_tools", fake_call)

    state = run_agent_loop(TICKET_ID)

    assert state.status == "resolved"
    assert len(critic_user_contents) == 2
    # Round 2's critic digest still carries round 1's summary forward,
    # sourced from state.trajectory rather than the (now unused) main
    # tool-calling conversation.
    assert "checkout-db connection errors spiking" in critic_user_contents[1]


# ---------------------------------------------------------------------------
# 15. Regression: a failed critic round no longer permanently loses that
# round's evidence (the observed T001 failure this fix addresses) (Fix 2)
# ---------------------------------------------------------------------------
def test_evidence_credited_across_a_failed_critic_round(monkeypatch):
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_logs", _stub_tool(status="ok")
    )
    monkeypatch.setitem(
        tool_executor_module.TOOL_REGISTRY, "query_metrics", _stub_tool(status="ok")
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
        _assessment_response("db_connection_pool_exhaustion", 0.9, True),
    ]
    _install_responses(monkeypatch, responses)

    state = run_agent_loop(TICKET_ID)

    # Under the old behavior, evidence was only credited for tools in the
    # CURRENT round, so query_logs (round 1, assessment_error) was never
    # credited and T001 escalated at the iteration cap despite a correct
    # hypothesis and 0.94 confidence. It must now be credited retroactively
    # once a later round's critic supports the hypothesis.
    assert state.evidence_sources == ["query_logs", "query_metrics"]
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
