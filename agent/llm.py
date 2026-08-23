"""Thin wrapper over an OpenAI-compatible chat-completions API.

Returns the raw ``ChatCompletion`` on success and a structured ``LLMError``
dict on failure, so the caller can degrade gracefully (skip a turn, retry,
surface a message) rather than crash the whole agent run — see docs/design.md
Reliability & Safety.

The provider is configurable via env vars so this can point at any
OpenAI-compatible endpoint serving the same open-weights model (Groq,
Baseten, etc.) without code changes.
"""

from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv

import openai
from openai.types.chat import ChatCompletion

load_dotenv()

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
    ``isinstance(resp, dict)``.
    """
    try:
        client = _get_client()
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return client.chat.completions.create(**kwargs)
    except RuntimeError as exc:
        return {"error": str(exc)}
    except openai.APIConnectionError as exc:
        return {"error": f"Could not reach the LLM API: {exc}"}
    except openai.RateLimitError as exc:
        return {"error": f"LLM API rate limit hit: {exc}"}
    except openai.APIStatusError as exc:
        return {"error": f"LLM API returned {exc.status_code}: {exc}"}
    except openai.APIError as exc:
        return {"error": f"LLM API error: {exc}"}
    except openai.OpenAIError as exc:
        # Backstop: preserves this module's "never raises" contract if a
        # future SDK version raises an OpenAIError that isn't an APIError
        # subclass.
        return {"error": f"LLM client error: {exc}"}
