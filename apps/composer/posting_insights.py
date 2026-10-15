"""Best-time-to-post suggestions and an engagement summary for the composer.

Both read the analytics snapshot tables; nothing here calls a platform API.
Best times come from the workspace's own published posts when there are
enough of them with metrics, and fall back to general per-platform
benchmarks otherwise. The result says which, so the UI never presents a
benchmark as if it were the user's own data.
"""

from __future__ import annotations

import zoneinfo
from collections import defaultdict
from datetime import datetime, timedelta

from django.utils import timezone

from apps.analytics.metrics import ENGAGEMENT_DENOMINATORS, ENGAGEMENT_PARTS
from apps.analytics.services import _latest_post_stats

from .models import PlatformPost

# How far back published posts count towards best-time scoring.
HISTORY_DAYS = 180
# Below this many posts with metrics, history is too thin to rank hours by.
MIN_HISTORY_POSTS = 6
# Weight of the "average post" prior per hour bucket. Shrinks one-off spikes
# towards average so a single viral post doesn't own the recommendation.
PRIOR_WEIGHT = 1.0
# A suggested slot must be at least this far in the future.
MIN_LEAD = timedelta(minutes=30)
ENGAGEMENT_DAYS = 30

# General industry guidance, as (weekday, hour) in the workspace's local time,
# Monday = 0. Only used until the workspace has its own history.
BENCHMARK_SLOTS: dict[str, list[tuple[int, int]]] = {
    "instagram": [(1, 11), (2, 11), (3, 13), (4, 10)],
    "instagram_login": [(1, 11), (2, 11), (3, 13), (4, 10)],
    "facebook": [(1, 9), (2, 9), (3, 13), (4, 9)],
    "linkedin_company": [(1, 9), (2, 10), (3, 9), (1, 12)],
    "linkedin_personal": [(1, 9), (2, 10), (3, 9), (1, 12)],
    "tiktok": [(1, 18), (3, 19), (4, 17), (5, 11)],
    "youtube": [(3, 15), (4, 15), (5, 12), (6, 10)],
    "threads": [(1, 9), (2, 12), (3, 10)],
    "bluesky": [(1, 10), (2, 11), (3, 9)],
    "mastodon": [(1, 10), (2, 11), (3, 9)],
    "pinterest": [(4, 20), (5, 20), (6, 15)],
    "google_business": [(1, 10), (2, 10), (3, 10)],
}
DEFAULT_BENCHMARK_SLOTS = [(1, 10), (2, 11), (3, 12)]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _engagements(stats: dict[str, float]) -> float:
    return sum(stats.get(part, 0.0) for part in ENGAGEMENT_PARTS)


def _engagement_rate(stats: dict[str, float]) -> float | None:
    denom = next((stats[d] for d in ENGAGEMENT_DENOMINATORS if stats.get(d, 0) > 0), None)
    return _engagements(stats) / denom * 100 if denom else None


def _published_posts(accounts, since):
    return list(
        PlatformPost.objects.filter(
            social_account__in=accounts,
            status=PlatformPost.Status.PUBLISHED,
            published_at__gte=since,
        ).select_related("post", "social_account")
    )


def _next_occurrence(weekday: int, hour: int, now_local: datetime) -> datetime:
    days_ahead = (weekday - now_local.weekday()) % 7
    candidate = (now_local + timedelta(days=days_ahead)).replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate < now_local + MIN_LEAD:
        candidate += timedelta(days=7)
    return candidate


def _slot(weekday: int, hour: int, now_local: datetime, reason: str) -> dict:
    when = _next_occurrence(weekday, hour, now_local)
    return {
        "weekday": WEEKDAYS[weekday],
        "hour": hour,
        "date": when.strftime("%Y-%m-%d"),
        "time": when.strftime("%H:%M"),
        # Built by hand: the %-d / %-I no-padding flags are glibc-only.
        "label": f"{when:%a, %b} {when.day} · {when.hour % 12 or 12}:{when:%M %p}",
        "reason": reason,
    }


def _history_buckets(posts, stats_by_post, tz) -> tuple[list[tuple[int, int]], int]:
    """Rank (weekday, hour) buckets by engagement relative to each account's average.

    Normalising per account lets a small Bluesky account and a large Instagram
    account share one ranking without the bigger one drowning out the other.
    """
    scored = [(pp, _engagements(stats_by_post[pp.id])) for pp in posts if stats_by_post.get(pp.id)]
    if len(scored) < MIN_HISTORY_POSTS:
        return [], len(scored)

    by_account: dict = defaultdict(list)
    for pp, eng in scored:
        by_account[pp.social_account_id].append(eng)
    account_mean = {acc: sum(vals) / len(vals) for acc, vals in by_account.items()}

    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for pp, eng in scored:
        mean = account_mean[pp.social_account_id]
        local = pp.published_at.astimezone(tz)
        buckets[(local.weekday(), local.hour)].append(eng / mean if mean else 1.0)

    def score(item):
        ratios = item[1]
        return (sum(ratios) + PRIOR_WEIGHT) / (len(ratios) + PRIOR_WEIGHT)

    ranked = sorted(buckets.items(), key=lambda item: (score(item), len(item[1])), reverse=True)
    return [key for key, ratios in ranked if score((key, ratios)) > 1.0], len(scored)


def _benchmark_buckets(accounts) -> list[tuple[int, int]]:
    platforms = sorted({acc.platform for acc in accounts})
    counts: dict[tuple[int, int], int] = defaultdict(int)
    order: list[tuple[int, int]] = []
    for platform in platforms or [""]:
        for slot in BENCHMARK_SLOTS.get(platform, DEFAULT_BENCHMARK_SLOTS):
            if slot not in counts:
                order.append(slot)
            counts[slot] += 1
    # Slots shared by more of the selected platforms first; ties keep table order.
    return sorted(order, key=lambda slot: -counts[slot])


def best_times(workspace, accounts, count: int = 3, now: datetime | None = None) -> dict:
    tz = zoneinfo.ZoneInfo(workspace.effective_timezone or "UTC")
    now = now or timezone.now()
    now_local = now.astimezone(tz)
    accounts = list(accounts)

    posts = _published_posts(accounts, now - timedelta(days=HISTORY_DAYS)) if accounts else []
    stats = _latest_post_stats([pp.id for pp in posts], list(ENGAGEMENT_PARTS))
    history, sample_size = _history_buckets(posts, stats, tz)

    slots = [
        _slot(wd, hr, now_local, f"Your posts at this time get above-average engagement ({sample_size} posts analysed)")
        for wd, hr in history[:count]
    ]
    if len(slots) < count:
        platform_names = ", ".join(sorted({acc.get_platform_display() for acc in accounts})) or "social media"
        taken = set(history[:count])
        for wd, hr in _benchmark_buckets(accounts):
            if len(slots) >= count:
                break
            if (wd, hr) not in taken:
                slots.append(_slot(wd, hr, now_local, f"General benchmark for {platform_names}"))

    slots.sort(key=lambda s: (s["date"], s["time"]))
    return {
        "source": "history" if history else "benchmark",
        "sample_size": sample_size,
        "timezone": str(tz),
        "slots": slots,
    }


def engagement_summary(accounts, days: int = ENGAGEMENT_DAYS, now: datetime | None = None) -> dict:
    now = now or timezone.now()
    accounts = list(accounts)
    posts = _published_posts(accounts, now - timedelta(days=days)) if accounts else []
    metrics = list(ENGAGEMENT_PARTS) + list(ENGAGEMENT_DENOMINATORS)
    stats = _latest_post_stats([pp.id for pp in posts], metrics)

    measured = [(pp, stats[pp.id]) for pp in posts if stats.get(pp.id)]
    total = sum(_engagements(s) for _, s in measured)
    rates = [r for r in (_engagement_rate(s) for _, s in measured) if r is not None]
    top = max(measured, key=lambda item: _engagements(item[1]), default=None)

    return {
        "days": days,
        "published_posts": len(posts),
        "measured_posts": len(measured),
        "total_engagements": round(total),
        "avg_engagements": round(total / len(measured), 1) if measured else None,
        "avg_engagement_rate": round(sum(rates) / len(rates), 2) if rates else None,
        "top_post": (
            {
                "caption": top[0].effective_caption[:140],
                "platform": top[0].social_account.get_platform_display(),
                "engagements": round(_engagements(top[1])),
            }
            if top
            else None
        ),
    }
