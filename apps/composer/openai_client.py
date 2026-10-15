"""Shared OpenAI plumbing for the composer's AI features (chat, ideas, images).

Optional: every feature is off unless ``OPENAI_API_KEY`` is set.
"""

from __future__ import annotations

import logging

import openai
from django.conf import settings

logger = logging.getLogger(__name__)

NOT_CONFIGURED = "AI features are not configured. Add OPENAI_API_KEY to your environment."


class AIError(Exception):
    """An OpenAI failure with a message that is safe to show the user."""


def is_enabled() -> bool:
    return bool(settings.OPENAI_API_KEY)


def get_client(timeout: float = 90.0) -> openai.OpenAI:
    if not is_enabled():
        raise AIError(NOT_CONFIGURED)
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY, timeout=timeout, max_retries=1)


def reasoning_kwargs(model: str) -> dict:
    """Low reasoning effort keeps chat-style calls snappy; only reasoning models accept it."""
    return {"reasoning_effort": "low"} if model.startswith(("gpt-5", "o")) else {}


def friendly_error(exc: openai.OpenAIError, fallback: str) -> AIError:
    """Map an SDK exception to a user-safe ``AIError`` (logging the details)."""
    if isinstance(exc, openai.AuthenticationError):
        return AIError("OpenAI rejected the API key. Check OPENAI_API_KEY.")
    if isinstance(exc, openai.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota":
            return AIError("The OpenAI account is out of credits. Add billing at platform.openai.com.")
        return AIError("OpenAI rate limit reached. Try again in a minute.")
    if isinstance(exc, openai.APIConnectionError):
        logger.warning("OpenAI connection failed: %s", exc)
        return AIError("Could not reach OpenAI. Try again.")
    if isinstance(exc, openai.BadRequestError) and getattr(exc, "code", None) == "moderation_blocked":
        return AIError("OpenAI's safety system blocked this request. Try rephrasing it.")
    if isinstance(exc, openai.APIStatusError):
        logger.warning(
            "OpenAI API error %s (request %s): %s", exc.status_code, exc.response.headers.get("x-request-id"), exc
        )
    else:
        logger.warning("OpenAI error: %s", exc)
    return AIError(fallback)
