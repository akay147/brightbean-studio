"""AI review of a published post: why it performed as it did, and how to do better.

Optional: off unless ``OPENAI_API_KEY`` is set. One review compares the post
with the account's own recent posts, looks at its image, and uses OpenAI web
search to find similar content that performed well publicly. Instagram's API
offers no search of other accounts' posts for Instagram-Login connections, so
those comparisons come from the public web, and only examples whose URL the
search actually visited are kept.

Each review's lessons are distilled into the workspace's ``ContentPlaybook``,
which the composer's chat assistant and the Ideas generator read, so later
posts on any topic build on what was learned.
"""

from __future__ import annotations

import logging
import re
import zoneinfo
from typing import Literal
from urllib.parse import urlsplit

import openai
from django.conf import settings
from django.utils import timezone
from pydantic import BaseModel

from apps.composer.openai_client import AIError, friendly_error, get_client, reasoning_kwargs
from apps.composer.openai_client import is_enabled as is_enabled

from .metrics import post_metrics_for

logger = logging.getLogger(__name__)

AIReviewError = AIError

MAX_LESSONS = 12
BASELINE_POSTS = 30

SYSTEM_PROMPT = (
    "You are a social media performance analyst inside a scheduling tool. You review "
    "one published post: explain why it performed the way it did and how the next post "
    "could do much better.\n\n"
    "Judge performance against the account's own baseline (its other recent posts) and "
    "its follower count, not against celebrity accounts. A post younger than about 48 "
    "hours is still collecting engagement: say so and use verdict too_early unless the "
    "numbers are already clear.\n\n"
    "Use web search to find similar or same-topic content that performed well, ideally "
    "public Instagram posts, reels or creators, otherwise strong examples from other "
    "platforms or articles about what works for this topic. Only list examples you "
    "actually found, with the exact URL you visited, and only mention engagement figures "
    "you saw. Instagram engagement is often not public: then explain what makes the "
    "example strong (hook, format, visual, caption) instead of inventing numbers.\n\n"
    "Be specific and practical. Reasons should point at concrete things: the hook in the "
    "first line, caption length, hashtag choice, format (single image vs carousel vs "
    "reel), visual, posting time, call to action. Lessons must be general rules that "
    "help future posts on any topic, not facts about this one post. Never invent metrics."
)


class SimilarContent(BaseModel):
    title: str
    url: str
    platform: str
    why_it_works: str


class PostReview(BaseModel):
    verdict: Literal["underperformed", "on_par", "outperformed", "too_early"]
    headline: str
    reasons: list[str]
    similar_content: list[SimilarContent]
    improvements: list[str]
    improved_caption: str
    lessons: list[str]


class Playbook(BaseModel):
    lessons: list[str]


def _latest_stats(post, metrics) -> dict[str, float]:
    from .services import _latest_post_stats

    return _latest_post_stats([post.id], metrics).get(post.id, {})


def _baseline(post, metrics) -> tuple[int, dict[str, float]]:
    """(number of posts, average of each metric) over the account's other recent posts."""
    from apps.composer.models import PlatformPost

    from .services import _latest_post_stats

    others = list(
        PlatformPost.objects.filter(social_account=post.social_account, status=PlatformPost.Status.PUBLISHED)
        .exclude(pk=post.pk)
        .order_by("-published_at")
        .values_list("pk", flat=True)[:BASELINE_POSTS]
    )
    stats = _latest_post_stats(others, metrics)
    if not stats:
        return 0, {}
    return len(stats), {m: round(sum(s.get(m, 0) for s in stats.values()) / len(stats), 1) for m in metrics}


def _image_inputs(post) -> list[dict]:
    from apps.composer.ai_composer import image_blocks

    assets = [pm.media_asset for pm in post.post.media_attachments.select_related("media_asset").all()]
    images = [a for a in assets if not a.is_video]
    return [{"type": "input_image", "image_url": b["image_url"]["url"]} for b in image_blocks(images[:2])]


def _build_context(post, lessons: list[str]) -> str:
    account = post.social_account
    workspace = account.workspace
    metrics = post_metrics_for(account.platform)
    caption = (post.platform_specific_caption or post.post.caption or "").strip()
    media = list(post.post.media_attachments.select_related("media_asset").all())
    kinds = {"video" if pm.media_asset.is_video else "image" for pm in media}
    if len(media) > 1:
        media_desc = f"carousel of {len(media)} ({', '.join(sorted(kinds))})"
    elif media:
        media_desc = f"single {next(iter(kinds))}"
    else:
        media_desc = "no media"

    lines = [
        f"Workspace: {workspace.name}",
        f"Platform: {account.get_platform_display()} (@{account.account_handle or account.account_name})",
        f"Followers: {account.follower_count}",
    ]
    if workspace.description:
        lines.append(f"About this workspace: {workspace.description}")
    if post.published_at:
        tz = zoneinfo.ZoneInfo(workspace.effective_timezone or "UTC")
        local = post.published_at.astimezone(tz)
        age_hours = int((timezone.now() - post.published_at).total_seconds() // 3600)
        lines.append(f"Published: {local:%A %Y-%m-%d %H:%M} ({tz.key}), {age_hours} hours ago")
    lines.append(f"Media: {media_desc}")
    hashtags = re.findall(r"#\w+", caption)
    lines.append(f"Caption ({len(caption)} characters, {len(hashtags)} hashtags):\n{caption or '(no caption)'}")

    stats = _latest_stats(post, metrics)
    lines.append("This post's metrics: " + ", ".join(f"{m} {stats.get(m, 0):g}" for m in metrics))
    count, averages = _baseline(post, metrics)
    if count:
        lines.append(
            f"Account baseline (average of its {count} other recent posts): "
            + ", ".join(f"{m} {averages[m]:g}" for m in metrics)
        )
    else:
        lines.append("No other posts with metrics on this account yet, so there is no baseline.")
    if lessons:
        lines.append("Lessons already learned from earlier reviews (build on these, don't repeat them):")
        lines.extend(f"- {lesson}" for lesson in lessons)
    return "\n".join(lines)


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}"


def _visited_urls(response) -> set[str]:
    """URLs the web search actually returned or opened, plus cited ones."""
    urls = set()
    for item in response.output or []:
        if item.type == "web_search_call":
            action = getattr(item, "action", None)
            for source in getattr(action, "sources", None) or []:
                if getattr(source, "url", None):
                    urls.add(_normalize_url(source.url))
            if getattr(action, "url", None):
                urls.add(_normalize_url(action.url))
        elif item.type == "message":
            for content in item.content or []:
                for ann in getattr(content, "annotations", None) or []:
                    if getattr(ann, "url", None):
                        urls.add(_normalize_url(ann.url))
    return urls


def review_post(post, lessons: list[str] | None = None) -> PostReview:
    """Review one published PlatformPost. Raises ``AIReviewError``."""
    client = get_client(timeout=240.0)
    model = settings.OPENAI_ANALYTICS_MODEL
    content = [{"type": "input_text", "text": _build_context(post, lessons or [])}, *_image_inputs(post)]
    try:
        response = client.responses.parse(
            model=model,
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
            text_format=PostReview,
            **({"reasoning": {"effort": "medium"}} if reasoning_kwargs(model) else {}),
        )
    except openai.OpenAIError as exc:
        raise friendly_error(exc, "The review couldn't be completed. Try again.") from exc

    review = response.output_parsed
    if review is None:
        logger.warning("Unusable post review (status %s, request %s)", response.status, response._request_id)
        raise AIReviewError("The review couldn't be completed. Try again.")

    visited = _visited_urls(response)
    kept = [ex for ex in review.similar_content if _normalize_url(ex.url) in visited]
    if len(kept) < len(review.similar_content):
        logger.info("Dropped %s unverified example URL(s) from a post review", len(review.similar_content) - len(kept))
    review.similar_content = kept
    return review


def distill_playbook(existing: list[str], new_lessons: list[str]) -> list[str]:
    """Merge new lessons into the playbook, keeping it short. Raises ``AIReviewError``."""
    if not new_lessons:
        return existing
    client = get_client()
    model = settings.OPENAI_IDEAS_MODEL
    prompt = (
        f"Merge these lessons into one playbook of at most {MAX_LESSONS} short, specific rules for "
        "writing better social media posts for this account. Combine duplicates, keep the most "
        "useful and well-supported rules, prefer newer evidence when rules conflict, and drop "
        "anything about a single post's topic.\n\nCurrent playbook:\n"
        + ("\n".join(f"- {lesson}" for lesson in existing) or "(empty)")
        + "\n\nNew lessons from the latest post review:\n"
        + "\n".join(f"- {lesson}" for lesson in new_lessons)
    )
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=Playbook,
            **reasoning_kwargs(model),
        )
    except openai.OpenAIError as exc:
        raise friendly_error(exc, "The playbook couldn't be updated.") from exc
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise AIReviewError("The playbook couldn't be updated.")
    return [lesson.strip() for lesson in parsed.lessons if lesson.strip()][:MAX_LESSONS]


def playbook_lessons(workspace) -> list[str]:
    """Lessons the AI has learned for ``workspace`` (empty when none yet)."""
    from .models import ContentPlaybook

    playbook = ContentPlaybook.objects.filter(workspace=workspace).first()
    return list(playbook.lessons) if playbook else []


def run_review(post):
    """Review ``post``, store it, and fold its lessons into the playbook.

    Returns ``(PostAIReview, playbook_error)``; ``playbook_error`` is a message
    when only the playbook update failed (the review itself is still saved).
    """
    from .models import ContentPlaybook, PostAIReview

    workspace = post.social_account.workspace
    lessons = playbook_lessons(workspace)
    review = review_post(post, lessons)
    stored, _ = PostAIReview.objects.update_or_create(
        platform_post=post,
        defaults={"result": review.model_dump(), "model": settings.OPENAI_ANALYTICS_MODEL},
    )

    playbook_error = ""
    try:
        merged = distill_playbook(lessons, review.lessons)
    except AIReviewError as exc:
        playbook_error = str(exc)
    else:
        playbook, _ = ContentPlaybook.objects.get_or_create(workspace=workspace)
        playbook.lessons = merged
        playbook.review_count += 1
        playbook.save()
    return stored, playbook_error
