"""AI post reviews and the workspace playbook they feed.

OpenAI traffic is mocked at apps.composer.openai_client.openai.OpenAI so no
network is involved.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.accounts.models import User
from apps.analytics.ai_review import Playbook, PostReview, SimilarContent
from apps.analytics.models import ContentPlaybook, PostAIReview, PostInsightsSnapshot
from apps.composer.ai_composer import _format_context
from apps.composer.models import PlatformPost, Post
from apps.members.models import OrgMembership, WorkspaceMembership
from apps.organizations.models import Organization
from apps.social_accounts.models import SocialAccount
from apps.workspaces.models import Workspace

OPENAI_CLIENT = "apps.composer.openai_client.openai.OpenAI"


def _review(**overrides):
    fields = {
        "verdict": "underperformed",
        "headline": "Weak hook and a static format for a topic that thrives as Reels.",
        "reasons": ["The first line is a generic question."],
        "similar_content": [
            SimilarContent(
                title="Pyramid reel", url="https://www.example.com/reel/1/", platform="Instagram", why_it_works="Hook"
            ),
            SimilarContent(title="Made up", url="https://invented.example/x", platform="Instagram", why_it_works="?"),
        ],
        "improvements": ["Post it as a carousel."],
        "improved_caption": "Not aliens. Here's how. 🏜️",
        "lessons": ["Lead with a curiosity gap."],
    }
    fields.update(overrides)
    return PostReview(**fields)


def _responses_result(review, visited=("https://example.com/reel/1",)):
    search = SimpleNamespace(
        type="web_search_call",
        action=SimpleNamespace(sources=[SimpleNamespace(url=u) for u in visited], url=None),
    )
    return SimpleNamespace(output_parsed=review, output=[search], status="completed", _request_id="req_1")


def _playbook_result(lessons):
    completion = MagicMock()
    completion.choices[0].message.parsed = Playbook(lessons=lessons)
    return completion


@override_settings(OPENAI_API_KEY="sk-test", OPENAI_ANALYTICS_MODEL="gpt-5.5")
class PostAIReviewViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="owner@example.com", password="testpass123", tos_accepted_at=timezone.now()
        )
        self.org = Organization.objects.create(name="Org")
        self.workspace = Workspace.objects.create(organization=self.org, name="History Facts", timezone="UTC")
        OrgMembership.objects.create(user=self.user, organization=self.org, org_role=OrgMembership.OrgRole.OWNER)
        WorkspaceMembership.objects.create(
            user=self.user, workspace=self.workspace, workspace_role=WorkspaceMembership.WorkspaceRole.OWNER
        )
        self.account = SocialAccount.objects.create(
            workspace=self.workspace,
            platform="instagram_login",
            account_platform_id="ig-1",
            account_name="techad52",
            connection_status=SocialAccount.ConnectionStatus.CONNECTED,
        )
        post = Post.objects.create(workspace=self.workspace, caption="How were the pyramids built? #Pyramids")
        self.pp = PlatformPost.objects.create(
            post=post,
            social_account=self.account,
            status=PlatformPost.Status.PUBLISHED,
            platform_post_id="ig-media-1",
            published_at=timezone.now() - timezone.timedelta(days=3),
        )
        PostInsightsSnapshot.objects.create(
            platform_post=self.pp, metric_key="likes", date=timezone.now().date(), value=1
        )
        self.client.force_login(self.user)
        self.url = reverse(
            "analytics:post_ai_review", kwargs={"workspace_id": self.workspace.id, "post_id": self.pp.id}
        )

    @patch(OPENAI_CLIENT)
    def test_review_is_stored_unverified_examples_dropped_and_playbook_updated(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.responses.parse.return_value = _responses_result(_review())
        client.chat.completions.parse.return_value = _playbook_result(["Lead with a curiosity gap."])

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Underperformed")
        self.assertContains(response, "Pyramid reel")
        self.assertNotContains(response, "invented.example")
        stored = PostAIReview.objects.get(platform_post=self.pp)
        self.assertEqual([ex["title"] for ex in stored.result["similar_content"]], ["Pyramid reel"])
        playbook = ContentPlaybook.objects.get(workspace=self.workspace)
        self.assertEqual(playbook.lessons, ["Lead with a curiosity gap."])
        self.assertEqual(playbook.review_count, 1)

        kwargs = client.responses.parse.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5.5")
        self.assertEqual(kwargs["tools"], [{"type": "web_search"}])
        prompt = kwargs["input"][0]["content"][0]["text"]
        self.assertIn("How were the pyramids built?", prompt)
        self.assertIn("likes 1", prompt)
        self.assertIn("no baseline", prompt)

    @patch(OPENAI_CLIENT)
    def test_review_is_kept_when_only_the_playbook_update_fails(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.responses.parse.return_value = _responses_result(_review())
        client.chat.completions.parse.return_value = _playbook_result([])
        client.chat.completions.parse.return_value.choices[0].message.parsed = None

        response = self.client.post(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(PostAIReview.objects.filter(platform_post=self.pp).exists())
        self.assertContains(response, "The review was saved")
        self.assertFalse(ContentPlaybook.objects.exists())

    @override_settings(OPENAI_API_KEY="")
    def test_returns_503_without_api_key(self):
        self.assertEqual(self.client.post(self.url).status_code, 503)

    def test_other_workspaces_posts_are_not_found(self):
        other = Workspace.objects.create(organization=self.org, name="Other")
        url = reverse("analytics:post_ai_review", kwargs={"workspace_id": other.id, "post_id": self.pp.id})
        # Not a member of ``other`` either, so this is refused before the lookup.
        self.assertIn(self.client.post(url).status_code, (403, 404))

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_composer_context_includes_learned_lessons(self):
        ContentPlaybook.objects.create(workspace=self.workspace, lessons=["Post history topics as carousels."])

        context = _format_context(self.workspace, [], "", {}, {})

        self.assertIn("Post history topics as carousels.", context)
