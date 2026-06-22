"""LLM client abstraction supporting both Anthropic and OpenAI providers.

Exposes a unified `chat(messages)` function and provider-agnostic exception
types so callers do not need to know which provider is in use.
"""

from src.configuration.config import (
    API_KEY,
    MAX_TOKENS,
    MODEL_NAME,
    PROVIDER,
    TEMPERATURE,
)


class LLMError(Exception):
    """Generic LLM API error."""


class LLMRateLimitError(LLMError):
    """LLM rate-limit error."""


class LLMEmptyResponseError(LLMError):
    """LLM returned no content blocks (usually a content-based refusal).

    Distinguished from generic LLMError so callers can skip the
    timeout-style exponential backoff: an empty response is almost
    always deterministic for the same prompt, so retries with backoff
    just waste wall-clock time.
    """


_client = None


def _get_client():
    """Lazy-init the provider client on first call."""
    global _client
    if _client is not None:
        return _client

    if PROVIDER == "anthropic":
        import anthropic
        _client = anthropic.Anthropic(api_key=API_KEY)
    elif PROVIDER == "openai":
        import openai
        _client = openai.OpenAI(api_key=API_KEY)
    else:
        raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")
    return _client


def chat(messages: list[dict]) -> str:
    """Send a multi-turn chat and return the assistant's reply text.

    Args:
        messages: list of {"role": "user"|"assistant", "content": str} dicts.

    Raises:
        LLMRateLimitError: if the provider returns a rate-limit error.
        LLMError: for other API errors.
    """
    client = _get_client()

    if PROVIDER == "anthropic":
        import anthropic
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=messages,
            )
            if not response.content:
                stop_reason = getattr(response, "stop_reason", "unknown")
                raise LLMEmptyResponseError(
                    f"Empty response content (stop_reason={stop_reason})"
                )
            return response.content[0].text
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.APIError as e:
            raise LLMError(str(e)) from e

    elif PROVIDER == "openai":
        import openai
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=messages,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIError as e:
            raise LLMError(str(e)) from e

    raise ValueError(f"Unknown PROVIDER: {PROVIDER!r}")
