"""AI-generated content ideas for the Ideas Kanban board.

Optional: the feature is off unless ``OPENAI_API_KEY`` is set. The view
calls ``generate_ideas`` and turns each result into an ``Idea`` row; nothing
here touches the database.
"""

from __future__ import annotations

import logging

import openai
from django.conf import settings
from pydantic import BaseModel

from apps.composer.openai_client import AIError, friendly_error, get_client, reasoning_kwargs
from apps.composer.openai_client import is_enabled as is_enabled

logger = logging.getLogger(__name__)

MAX_TOPIC_LENGTH = 500
DEFAULT_IDEA_COUNT = 5

SYSTEM_PROMPT = (
    "You are a social media strategist helping a team plan content. "
    "Suggest distinct, concrete post ideas the team could act on this week: "
    "each needs a short, specific title (under 80 characters), a 1-3 sentence "
    "description of the angle and format (e.g. carousel, short video, thread), "
    "and 1-3 lowercase single-word tags. Avoid generic advice and avoid "
    "repeating ideas the team already has."
)


AIIdeasError = AIError


class GeneratedIdea(BaseModel):
    title: str
    description: str
    tags: list[str]


class GeneratedIdeas(BaseModel):
    ideas: list[GeneratedIdea]


def _build_prompt(workspace, topic: str, existing_titles: list[str], count: int) -> str:
    lines = [f"Workspace: {workspace.name}"]
    if workspace.description:
        lines.append(f"About this workspace: {workspace.description}")
    lines.append(f"Topic or focus: {topic}" if topic else "Topic or focus: anything relevant to this workspace")
    if existing_titles:
        lines.append("Ideas already on the board:")
        lines.extend(f"- {title}" for title in existing_titles)
    from apps.analytics.ai_review import playbook_lessons

    lessons = playbook_lessons(workspace)
    if lessons:
        lines.append("Lessons learned from reviewing this workspace's past posts (shape ideas around them):")
        lines.extend(f"- {lesson}" for lesson in lessons)
    lines.append(f"Suggest {count} new content ideas.")
    return "\n".join(lines)


def generate_ideas(
    workspace,
    topic: str = "",
    existing_titles: list[str] | None = None,
    count: int = DEFAULT_IDEA_COUNT,
) -> list[GeneratedIdea]:
    """Ask the model for ``count`` content ideas. Raises ``AIIdeasError``."""
    client = get_client()
    model = settings.OPENAI_IDEAS_MODEL
    prompt = _build_prompt(workspace, topic[:MAX_TOPIC_LENGTH], existing_titles or [], count)
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            response_format=GeneratedIdeas,
            **reasoning_kwargs(model),
        )
    except openai.LengthFinishReasonError as exc:
        logger.warning("Idea response hit the token limit: %s", exc)
        raise AIIdeasError("Idea generation failed. Try again.") from exc
    except openai.OpenAIError as exc:
        raise friendly_error(exc, "Idea generation failed. Try again.") from exc

    message = completion.choices[0].message
    if message.refusal:
        raise AIIdeasError("The AI declined this topic. Try rephrasing it.")
    if message.parsed is None:
        logger.warning("Unusable idea response (request %s)", completion._request_id)
        raise AIIdeasError("Idea generation failed. Try again.")

    return [idea for idea in message.parsed.ideas if idea.title.strip()][:count]
