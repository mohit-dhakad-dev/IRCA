"""Thin wrapper over an OpenAI-compatible chat-completions API.

Returns the raw ``ChatCompletion`` on success and a structured ``LLMError``
dict on failure, so the caller can degrade gracefully (skip a turn, retry,
surface a message) rather than crash the whole agent run — see docs/design.md
Reliability & Safety.

The provider is configurable via env vars so this can point at any
OpenAI-compatible endpoint serving the same open-weights model (Groq,
Baseten, etc.) without code changes.

Retry policy: ``call_llm_with_tools`` retries only genuinely transient
failures — ``openai.RateLimitError`` (429), ``openai.APIConnectionError``,
and ``openai.APIStatusError`` with ``status_code >= 500`` — using exponential
backoff with jitter, up to ``MAX_RETRIES`` additional attempts. A `Retry-After`
header from the provider (Groq sends one on 429s) is honoured verbatim when
present and parseable, capped at ``MAX_BACKOFF_SECONDS``. All other errors
(400, 401, 403, 404, and any other permanent 4xx, plus bad local config) are
NOT retried and return the same ``{"error": ...}`` shape immediately — a
retry loop cannot fix a bad API key or a retired/unknown model (see the
llama-3.3-70b-versatile note above), so retrying those would only waste wall
clock. This function never raises, regardless of which path it takes.
"""

from __future__ import annotations

import os
import random
import time
from typing import TypedDict

from dotenv import load_dotenv

import openai
from openai.types.chat import ChatCompletion

load_dotenv()

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0

# Indirection so tests can monkeypatch sleeping without ever actually
# blocking the test suite.
_sleep = time.sleep

# llama-3.3-70b-versatile was retired on Groq (404 model_not_found);
# gpt-oss-120b is the largest tool-calling-capable model on the current API.
# Open-weights model, so it's servable from multiple OpenAI-compatible hosts
# (Groq, Baseten, ...) — see LLM_BASE_URL / LLM_API_KEY / LLM_MODEL below.
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)


class LLMError(TypedDict):
    error: str


_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is not None:
        return _client
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip()
    key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GROQ_API_KEY", "").strip()
    )
    if not key:
        raise RuntimeError(
            "LLM_API_KEY (or GROQ_API_KEY) is not set; add it to .env "
            "(see .env.example)."
        )
    _client = openai.OpenAI(api_key=key, base_url=base_url)
    return _client


def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of a ``Retry-After`` header (seconds) from an
    API exception's underlying HTTP response. Returns ``None`` if absent or
    unparseable — never raises.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _computed_backoff(attempt: int) -> float:
    delay = min(INITIAL_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, delay * 0.25)
    return min(delay + jitter, MAX_BACKOFF_SECONDS)


def _backoff_delay(attempt: int, exc: Exception) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, MAX_BACKOFF_SECONDS)
    return _computed_backoff(attempt)


def call_llm_with_tools(
    messages: list,
    tools: list,
    *,
    temperature: float = 0.0,
    tool_choice: str = "auto",
) -> ChatCompletion | LLMError:
    """Call the configured chat-completions endpoint with an optional
    tool-calling schema.

    ``messages`` are OpenAI-style role/content dicts. ``tools`` are
    OpenAI-style function schemas (typically ``agent.tool_schemas.TOOL_SCHEMAS``).
    On success, returns the raw ``ChatCompletion`` so the caller can read
    ``.choices[0].message.tool_calls``. On any API/config failure, returns
    ``{"error": "..."}`` instead of raising. Callers should branch with
    ``isinstance(resp, dict)``. Transient failures (429, connection errors,
    5xx) are retried internally with backoff — see module docstring.
    """
    try:
        client = _get_client()
    except RuntimeError as exc:
        return {"error": str(exc)}

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    attempts = 0
    while True:
        attempts += 1
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            transient = True
            error_text = f"LLM API rate limit hit: {exc}"
            transient_exc = exc
        except openai.APIConnectionError as exc:
            transient = True
            error_text = f"Could not reach the LLM API: {exc}"
            transient_exc = exc
        except openai.APIStatusError as exc:
            transient = exc.status_code >= 500
            error_text = f"LLM API returned {exc.status_code}: {exc}"
            transient_exc = exc
        except openai.APIError as exc:
            return {"error": f"LLM API error: {exc}"}
        except openai.OpenAIError as exc:
            # Backstop: preserves this module's "never raises" contract if a
            # future SDK version raises an OpenAIError that isn't an APIError
            # subclass.
            return {"error": f"LLM client error: {exc}"}

        if not transient or attempts > MAX_RETRIES:
            total_attempts = attempts
            return {"error": f"{error_text} (after {total_attempts} attempts)"}

        delay = _backoff_delay(attempts - 1, transient_exc)
        _sleep(delay)
