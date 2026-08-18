"""Thin wrapper over the Groq chat-completions API.

Returns the raw ``ChatCompletion`` on success and a structured ``LLMError``
dict on failure, so the caller can degrade gracefully (skip a turn, retry,
surface a message) rather than crash the whole agent run — see docs/design.md
Reliability & Safety.
"""

from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv

import groq
from groq.types.chat import ChatCompletion

# llama-3.3-70b-versatile was retired on Groq (404 model_not_found);
# gpt-oss-120b is the largest tool-calling-capable model on the current API.
MODEL = "openai/gpt-oss-120b"


class LLMError(TypedDict):
    error: str


load_dotenv()

_client: groq.Groq | None = None


def _get_client() -> groq.Groq:
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set; add it to .env (see .env.example)."
        )
    _client = groq.Groq(api_key=key)
    return _client


def call_llm_with_tools(
    messages: list,
    tools: list,
    *,
    temperature: float = 0.0,
    tool_choice: str = "auto",
) -> ChatCompletion | LLMError:
    """Call Groq chat completions with an optional tool-calling schema.

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
    except groq.APIConnectionError as exc:
        return {"error": f"Could not reach the Groq API: {exc}"}
    except groq.RateLimitError as exc:
        return {"error": f"Groq rate limit hit: {exc}"}
    except groq.APIStatusError as exc:
        return {"error": f"Groq API returned {exc.status_code}: {exc}"}
    except groq.APIError as exc:
        return {"error": f"Groq API error: {exc}"}
    except groq.GroqError as exc:
        # Backstop: preserves this module's "never raises" contract if a
        # future SDK version raises a GroqError that isn't an APIError
        # subclass.
        return {"error": f"Groq client error: {exc}"}
