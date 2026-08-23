"""Tests for agent/llm.py. No network access, no real API key — the OpenAI
client is always stubbed out via monkeypatch.
"""

import httpx
import openai
import pytest

import agent.llm as llm_module
from agent.llm import MAX_RETRIES, MAX_BACKOFF_SECONDS, MODEL, call_llm_with_tools


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """No test in this module may block on a real sleep. Records requested
    delays instead so retry/backoff behaviour can be asserted.
    """
    delays = []
    monkeypatch.setattr(llm_module, "_sleep", lambda seconds: delays.append(seconds))
    return delays


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
    """Stands in for a ChatCompletion without depending on openai internals."""


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
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "_client", None)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert isinstance(result, dict)
    assert "error" in result
    assert "LLM_API_KEY" in result["error"]


def _make_api_connection_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.APIConnectionError(message="conn fail", request=req)


def _make_rate_limit_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return openai.RateLimitError("rate limited", response=resp, body=None)


def _make_api_status_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(500, request=req)
    return openai.APIStatusError("server error", response=resp, body=None)


def _make_api_error():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.APIError("generic api error", req, body=None)


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
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "_client", None)
    error_result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])
    assert isinstance(error_result, dict) is True

    sentinel = _Sentinel()
    stub = _StubClient(result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)
    success_result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])
    assert isinstance(success_result, dict) is False


class TestClientConfigResolution:
    """Offline tests for env-driven base_url/api_key/model resolution."""

    def setup_method(self):
        llm_module._client = None

    def teardown_method(self):
        llm_module._client = None

    def test_defaults_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        captured = {}

        class _FakeOpenAI:
            def __init__(self, api_key, base_url):
                captured["api_key"] = api_key
                captured["base_url"] = base_url

        monkeypatch.setattr(llm_module.openai, "OpenAI", _FakeOpenAI)

        client = llm_module._get_client()

        assert captured["base_url"] == llm_module.DEFAULT_BASE_URL
        assert captured["api_key"] == "groq-key"
        assert isinstance(client, _FakeOpenAI)

    def test_llm_api_key_takes_precedence_over_groq_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("LLM_API_KEY", "llm-key")

        captured = {}

        class _FakeOpenAI:
            def __init__(self, api_key, base_url):
                captured["api_key"] = api_key

        monkeypatch.setattr(llm_module.openai, "OpenAI", _FakeOpenAI)

        llm_module._get_client()

        assert captured["api_key"] == "llm-key"

    def test_groq_api_key_alone_still_works(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "groq-only-key")

        captured = {}

        class _FakeOpenAI:
            def __init__(self, api_key, base_url):
                captured["api_key"] = api_key

        monkeypatch.setattr(llm_module.openai, "OpenAI", _FakeOpenAI)

        llm_module._get_client()

        assert captured["api_key"] == "groq-only-key"

    def test_custom_base_url_used(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://inference.baseten.co/v1")

        captured = {}

        class _FakeOpenAI:
            def __init__(self, api_key, base_url):
                captured["base_url"] = base_url

        monkeypatch.setattr(llm_module.openai, "OpenAI", _FakeOpenAI)

        llm_module._get_client()

        assert captured["base_url"] == "https://inference.baseten.co/v1"

    def test_missing_key_raises_runtime_error_mapped_to_error_dict(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        with pytest.raises(RuntimeError):
            llm_module._get_client()


class _FlakyCompletions:
    """Raises the given exceptions in order, then returns `result` (or keeps
    raising the last exception forever if `result` is None and exhausted).
    """

    def __init__(self, excs, result=None):
        self._excs = list(excs)
        self._result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._excs:
            raise self._excs.pop(0)
        return self._result


class _FlakyClient:
    def __init__(self, excs, result=None):
        self.completions = _FlakyCompletions(excs, result=result)
        self.chat = _StubChat(self.completions)


def _make_rate_limit_error_with_retry_after(value):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": value} if value is not None else {}
    resp = httpx.Response(429, request=req, headers=headers)
    return openai.RateLimitError("rate limited", response=resp, body=None)


def _make_status_error(status_code):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(status_code, request=req)
    return openai.APIStatusError(f"status {status_code}", response=resp, body=None)


def test_429_then_success_retries_once(monkeypatch, _no_real_sleep=None):
    sentinel = _Sentinel()
    exc = _make_rate_limit_error_with_retry_after(None)
    stub = _FlakyClient([exc], result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert result is sentinel
    assert len(stub.completions.calls) == 2


def test_repeated_429_exhausts_retries(monkeypatch):
    excs = [_make_rate_limit_error_with_retry_after(None) for _ in range(MAX_RETRIES + 1)]
    stub = _FlakyClient(excs, result=None)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert isinstance(result, dict)
    assert "error" in result
    assert str(MAX_RETRIES + 1) in result["error"]
    assert len(stub.completions.calls) == MAX_RETRIES + 1


def test_delays_grow_exponentially_and_cap(monkeypatch, _no_real_sleep):
    excs = [_make_rate_limit_error_with_retry_after(None) for _ in range(MAX_RETRIES + 1)]
    stub = _FlakyClient(excs, result=None)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    delays = _no_real_sleep
    assert len(delays) == MAX_RETRIES
    assert all(d <= MAX_BACKOFF_SECONDS for d in delays)
    # non-decreasing (jitter can only add, cap prevents strict decrease)
    assert all(delays[i] <= delays[i + 1] for i in range(len(delays) - 1))


def test_retry_after_header_honoured(monkeypatch, _no_real_sleep):
    sentinel = _Sentinel()
    exc = _make_rate_limit_error_with_retry_after("7")
    stub = _FlakyClient([exc], result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert result is sentinel
    assert _no_real_sleep == [7.0]


def test_malformed_retry_after_falls_back_to_computed_backoff(monkeypatch, _no_real_sleep):
    sentinel = _Sentinel()
    exc = _make_rate_limit_error_with_retry_after("soon")
    stub = _FlakyClient([exc], result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert result is sentinel
    assert len(_no_real_sleep) == 1
    assert _no_real_sleep[0] != 7.0


def test_401_returns_immediately_no_retry(monkeypatch, _no_real_sleep):
    exc = _make_status_error(401)
    stub = _FlakyClient([exc], result=_Sentinel())
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert isinstance(result, dict)
    assert "error" in result
    assert len(stub.completions.calls) == 1
    assert _no_real_sleep == []


def test_404_returns_immediately_no_retry(monkeypatch, _no_real_sleep):
    exc = _make_status_error(404)
    stub = _FlakyClient([exc], result=_Sentinel())
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert isinstance(result, dict)
    assert "error" in result
    assert len(stub.completions.calls) == 1
    assert _no_real_sleep == []


def test_api_connection_error_retries(monkeypatch, _no_real_sleep):
    sentinel = _Sentinel()
    exc = _make_api_connection_error()
    stub = _FlakyClient([exc], result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert result is sentinel
    assert len(stub.completions.calls) == 2
    assert len(_no_real_sleep) == 1


def test_500_status_error_retries(monkeypatch, _no_real_sleep):
    sentinel = _Sentinel()
    exc = _make_status_error(500)
    stub = _FlakyClient([exc], result=sentinel)
    monkeypatch.setattr(llm_module, "_get_client", lambda: stub)

    result = call_llm_with_tools([{"role": "user", "content": "hi"}], [])

    assert result is sentinel
    assert len(stub.completions.calls) == 2
    assert len(_no_real_sleep) == 1
