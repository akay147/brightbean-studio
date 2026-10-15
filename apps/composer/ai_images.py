"""Prompt-to-image generation for the composer (OpenAI image models).

Optional: off unless ``OPENAI_API_KEY`` is set. Returns JPEG bytes; the view
stores them as a ``MediaAsset`` and attaches it like any upload. JPEG because
Instagram's publishing API only accepts JPEG images.
"""

from __future__ import annotations

import base64
import logging

import openai
from django.conf import settings

from apps.composer.openai_client import AIError, friendly_error, get_client
from apps.composer.openai_client import is_enabled as is_enabled

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 4000
# Square and landscape fit Instagram's feed (4:5 to 1.91:1); portrait (2:3) suits Stories.
SIZES = {"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"}

AIImageError = AIError


def generate_image(prompt: str, size: str = "square") -> bytes:
    """JPEG bytes for ``prompt``. Raises ``AIImageError``."""
    prompt = prompt.strip()[:MAX_PROMPT_LENGTH]
    if not prompt:
        raise AIImageError("Describe the image you want.")
    # Image generation regularly takes 30-60 seconds.
    client = get_client(timeout=180.0)
    try:
        result = client.images.generate(
            model=settings.OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=SIZES.get(size, SIZES["square"]),
            n=1,
            output_format="jpeg",
        )
    except openai.OpenAIError as exc:
        raise friendly_error(exc, "Image generation failed. Try again.") from exc

    encoded = result.data[0].b64_json if result.data else None
    if not encoded:
        logger.warning("Image response had no data (request %s)", result._request_id)
        raise AIImageError("Image generation failed. Try again.")
    return base64.b64decode(encoded)
