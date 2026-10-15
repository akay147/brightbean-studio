"""Chat-driven post drafting for the composer.

Optional: the feature is off unless ``OPENAI_API_KEY`` is set. The browser
keeps the conversation and sends it in full each turn; the model answers with
a short chat reply, the complete post text (dropped into the editor) and,
when the user asks for a visual, a prompt the browser hands to the image
generator (``ai_images``). Images attached to the post are sent along so the
model can write about what they show. Nothing here writes to the database.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from typing import Literal

import openai
from django.conf import settings
from pydantic import BaseModel

from apps.composer.openai_client import AIError, friendly_error, get_client, reasoning_kwargs
from apps.composer.openai_client import is_enabled as is_enabled

logger = logging.getLogger(__name__)

MAX_MESSAGES = 30
MAX_MESSAGE_LENGTH = 4000
MAX_CAPTION_LENGTH = 10000
MAX_IMAGES = 4
# Keep each image small enough to send every turn; these are the formats the API reads.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

SYSTEM_PROMPT = (
    "You are the writing assistant inside a social media scheduling tool. The user "
    "chats with you about a content idea and you turn it into a post that is ready "
    "to publish on the channels they selected.\n\n"
    "Each turn, return two fields:\n"
    "- caption: the complete post text exactly as it should be published, including "
    "any hashtags and emoji. Fit the tightest character limit among the selected "
    "channels and match each platform's conventions (e.g. a hook in the first line "
    "for Instagram, a professional tone for LinkedIn). The editor contents you are "
    "given are the latest version, possibly edited by the user, so revise from them "
    "rather than from your earlier drafts. Return an empty string when the message "
    "doesn't call for a new or revised draft, such as a question about timing.\n"
    "- image_prompt: you can create images. When the user asks for an image, photo, "
    "graphic or visual (or asks you to make one for the post), write a detailed prompt "
    "for an image generator: subject, setting, style, composition, lighting and mood. "
    "Don't ask for text in the image unless the user wants it. Otherwise return an "
    "empty string; don't create a new image unless the user asks for one.\n"
    "- image_size: the shape of that image. Use square for Instagram feed posts and "
    "when unsure, landscape for LinkedIn, X and Facebook, and portrait only for "
    "Stories, Pinterest or TikTok.\n"
    "- reply: a short, friendly chat message (1-3 sentences) about what you drafted "
    "or answering the user. When it helps, recommend one of the suggested posting "
    "times you are given and mention relevant engagement figures. Only cite numbers "
    "that appear in the provided context; never invent statistics.\n\n"
    "If the idea is too vague to write a good post, write your best attempt and ask "
    "one focused question in the reply. Write in the language the user writes in.\n\n"
    "Images attached to the post may be included with the newest message. Write about "
    "what they actually show and don't describe details you can't see. You can't edit "
    "attached images, only create new ones."
)


AIComposerError = AIError


class DraftTurn(BaseModel):
    reply: str
    caption: str
    image_prompt: str
    image_size: Literal["square", "portrait", "landscape"]


def _read_image(field) -> tuple[str, bytes] | None:
    """(media type, bytes) for a stored image file, or None if the API can't read it."""
    if not field:
        return None
    media_type = mimetypes.guess_type(field.name)[0]
    if media_type not in SUPPORTED_IMAGE_TYPES:
        return None
    try:
        if field.size > MAX_IMAGE_BYTES:
            return None
        with field.open("rb") as fh:
            return media_type, fh.read()
    except OSError as exc:
        logger.warning("Could not read image %s for the composer assistant: %s", field.name, exc)
        return None


def image_blocks(assets) -> list[dict]:
    """Inline (data URL) image parts for up to ``MAX_IMAGES`` attached images.

    Falls back to the asset's thumbnail when the original is too large or in a
    format the API doesn't accept (e.g. HEIC). Unreadable assets are skipped.
    """
    blocks = []
    for asset in assets:
        image = _read_image(asset.file) or _read_image(asset.thumbnail)
        if image is None:
            continue
        media_type, data = image
        encoded = base64.b64encode(data).decode()
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}})
        if len(blocks) == MAX_IMAGES:
            break
    return blocks


def _format_context(workspace, accounts, caption: str, best_times: dict, engagement: dict) -> str:
    lines = [f"Workspace: {workspace.name}"]
    if workspace.description:
        lines.append(f"About this workspace: {workspace.description}")

    if accounts:
        lines.append("Selected channels:")
        lines.extend(
            f"- {acc.get_platform_display()} ({acc.display_label}), {acc.char_limit} character limit"
            for acc in accounts
        )
    else:
        lines.append("No channels selected yet; write a post that works across mainstream platforms.")

    from apps.analytics.ai_review import playbook_lessons

    lessons = playbook_lessons(workspace)
    if lessons:
        lines.append("Lessons learned from reviewing this workspace's past posts (apply them):")
        lines.extend(f"- {lesson}" for lesson in lessons)

    lines.append(f"Current editor contents:\n{caption}" if caption.strip() else "The editor is empty.")

    if best_times.get("slots"):
        basis = (
            f"from the workspace's own post history ({best_times['sample_size']} posts)"
            if best_times["source"] == "history"
            else "from general platform benchmarks, not this workspace's data"
        )
        lines.append(f"Suggested posting times ({best_times['timezone']}, {basis}):")
        lines.extend(f"- {slot['label']}" for slot in best_times["slots"])

    if engagement.get("measured_posts"):
        lines.append(
            f"Engagement over the last {engagement['days']} days: {engagement['measured_posts']} posts with metrics, "
            f"{engagement['total_engagements']} total engagements, {engagement['avg_engagements']} per post"
            + (
                f", {engagement['avg_engagement_rate']}% average engagement rate"
                if engagement["avg_engagement_rate"] is not None
                else ""
            )
            + "."
        )
        if engagement["top_post"]:
            top = engagement["top_post"]
            lines.append(f"Top post ({top['platform']}, {top['engagements']} engagements): {top['caption']}")
    else:
        lines.append("No engagement data yet for these channels.")

    return "\n".join(lines)


def _to_api_messages(history: list[dict], context: str, images: list[dict] | None = None) -> list[dict]:
    """Map browser history to API messages, attaching context (and the post's
    images) to the newest user turn.

    Assistant turns are replayed as the JSON they were generated as, so the
    model sees its own earlier drafts in the shape it produced them.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, msg in enumerate(history):
        if msg["role"] == "assistant":
            content = DraftTurn(
                reply=msg["content"],
                caption=msg.get("caption", ""),
                image_prompt=msg.get("image_prompt", ""),
                image_size="square",
            ).model_dump_json(exclude={"image_size"})
        elif i == len(history) - 1:
            content = f"<composer_context>\n{context}\n</composer_context>\n\n{msg['content']}"
            if images:
                content = [{"type": "text", "text": content}, *images]
        else:
            content = msg["content"]
        messages.append({"role": msg["role"], "content": content})
    return messages


def clean_history(raw) -> list[dict]:
    """Validate the browser-supplied conversation. Raises ``AIComposerError``."""
    if not isinstance(raw, list) or not raw:
        raise AIComposerError("Send a message to get started.")
    history = []
    for item in raw[-MAX_MESSAGES:]:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            raise AIComposerError("The conversation could not be read. Refresh and try again.")
        content = str(item.get("content", "")).strip()[:MAX_MESSAGE_LENGTH]
        caption = str(item.get("caption", ""))[:MAX_CAPTION_LENGTH]
        image_prompt = str(item.get("image_prompt", ""))[:MAX_MESSAGE_LENGTH]
        if not content and not caption:
            continue
        history.append({"role": item["role"], "content": content, "caption": caption, "image_prompt": image_prompt})
    # Trimming to MAX_MESSAGES can leave an assistant turn first; the API needs a user turn.
    while history and history[0]["role"] != "user":
        history.pop(0)
    if not history or history[-1]["role"] != "user":
        raise AIComposerError("Send a message to get started.")
    return history


def draft_post(
    workspace,
    accounts,
    history: list[dict],
    caption: str = "",
    best_times: dict | None = None,
    engagement: dict | None = None,
    images: list[dict] | None = None,
) -> DraftTurn:
    """Ask the model for the next chat reply and draft. Raises ``AIComposerError``."""
    context = _format_context(workspace, accounts, caption[:MAX_CAPTION_LENGTH], best_times or {}, engagement or {})
    if images:
        context += f"\n{len(images)} image(s) attached to the post are included below."
    client = get_client()
    model = settings.OPENAI_COMPOSER_MODEL
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=_to_api_messages(history, context, images),
            response_format=DraftTurn,
            **reasoning_kwargs(model),
        )
    except openai.LengthFinishReasonError as exc:
        logger.warning("Composer response hit the token limit: %s", exc)
        raise AIComposerError("The assistant couldn't respond. Try again.") from exc
    except openai.OpenAIError as exc:
        raise friendly_error(exc, "The assistant couldn't respond. Try again.") from exc

    message = completion.choices[0].message
    if message.refusal:
        raise AIComposerError("The assistant declined this request. Try rephrasing it.")
    if message.parsed is None:
        logger.warning("Unusable composer response (request %s)", completion._request_id)
        raise AIComposerError("The assistant couldn't respond. Try again.")
    return message.parsed
