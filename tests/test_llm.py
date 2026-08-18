"""Tests for agent/llm.py. No network access, no real API key — the Groq
client is always stubbed out via monkeypatch.
"""

import httpx
import groq
import pytest

import agent.llm as llm_module
from agent.llm import MODEL, call_llm_with_tools


class _StubCompletions:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._result


class _StubChat:
    def __init__(self, completions):
        self.completions = completions


class _StubClient:
    def __init__(self, result=None, exc=None):
        self.completions = _StubCompletions(result=result, exc=exc)
        self.chat = _StubChat(self.completions)


class _Sentinel:
    """Stands in for a ChatCompletion without depending on groq internals."""


def test_call_llm_with_tools_success_returns_sentinel(monkeypatch):
    sentinel = _Sentinel()
    stub = _StubClient(result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "query_logs"}}]

    result = call_llm_with_tools(messages, tools)

    assert result is sentinel
    assert len(stub.completions.calls) == 1
    recorded = stub.completions.calls[0]
    assert recorded["model"] == MODEL
    assert recorded["tools"] == tools
    assert recorded["tool_choice"] == "auto"


def test_call_llm_with_tools_empty_tools_omits_kwargs(monkeypatch):
    sentinel = _Sentinel()
    stub = _StubClient(result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert result is sentinel
    recorded = stub.completions.calls[0]
    assert "tools" not in recorded
    assert "tool_choice" not in recorded


def test_call_llm_with_tools_missing_key_returns_error_dict(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "_client", None)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert isinstance(result, dict)
    assert "error" in result
    assert "GROQ_API_KEY" in result["error"]


def _make_api_connection_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIConnectionError(message="conn fail", request=req)


def _make_rate_limit_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return groq.RateLimitError("rate limited", response=resp, body=None)


def _make_api_status_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(500, request=req)
    return groq.APIStatusError("server error", response=resp, body=None)


def _make_api_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIError("generic api error", req, body=None)


@pytest.mark.parametrize(
    "make_exc",
    [
        _make_api_connection_error,
        _make_rate_limit_error,
        _make_api_status_error,
        _make_api_error,
    ],
)
def test_call_llm_with_tools_api_errors_return_error_dict(monkeypatch, make_exc):
    exc = make_exc()
    stub = _StubClient(exc=exc)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert isinstance(result, dict)
    assert "error" in result
    assert result["error"]


def test_error_path_is_dict_and_success_sentinel_is_not(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "_client", None)
    error_result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])
    assert isinstance(error_result, dict) is True

    sentinel = _Sentinel()
    stub = _StubClient(result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)
    success_result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])
    assert isinstance(success_result, dict) is False
