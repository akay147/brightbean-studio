"""Tests for AI-generated ideas on the Ideas board.

OpenAI traffic is mocked at apps.composer.openai_client.openai.OpenAI so no
network is involved.
"""

from unittest.mock import MagicMock, patch

import httpx
import openai
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.accounts.models import User
from apps.composer.ai_ideas import GeneratedIdea, GeneratedIdeas
from apps.composer.models import Idea, IdeaGroup, Tag
from apps.members.models import OrgMembership, WorkspaceMembership
from apps.organizations.models import Organization
from apps.workspaces.models import Workspace

OPENAI_CLIENT = "apps.composer.openai_client.openai.OpenAI"


def _parsed_response(ideas, refusal=None):
    completion = MagicMock()
    message = completion.choices[0].message
    message.parsed = GeneratedIdeas(ideas=ideas) if ideas is not None else None
    message.refusal = refusal
    return completion


class AIIdeasTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="owner@example.com",
            password="testpass123",
            tos_accepted_at=timezone.now(),
        )
        self.org = Organization.objects.create(name="Test Org")
        self.workspace = Workspace.objects.create(organization=self.org, name="Test Workspace")
        OrgMembership.objects.create(
            user=self.user,
            organization=self.org,
            org_role=OrgMembership.OrgRole.OWNER,
        )
        WorkspaceMembership.objects.create(
            user=self.user,
            workspace=self.workspace,
            workspace_role=WorkspaceMembership.WorkspaceRole.OWNER,
        )
        self.client.force_login(self.user)
        self.generate_url = reverse("composer:idea_generate", kwargs={"workspace_id": self.workspace.id})


class GenerateButtonVisibilityTests(AIIdeasTestCase):
    def _landing_html(self):
        url = reverse("composer:create_landing", kwargs={"workspace_id": self.workspace.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        return response.content.decode()

    @override_settings(OPENAI_API_KEY="sk-test")
    def test_button_rendered_with_api_key(self):
        html = self._landing_html()
        self.assertIn('title="Generate ideas with AI"', html)
        self.assertIn('x-show="showGenerateModal"', html)

    @override_settings(OPENAI_API_KEY="")
    def test_button_hidden_without_api_key(self):
        html = self._landing_html()
        self.assertNotIn('title="Generate ideas with AI"', html)
        self.assertNotIn('x-show="showGenerateModal"', html)


@override_settings(OPENAI_API_KEY="sk-test", OPENAI_IDEAS_MODEL="gpt-5.4-mini")
class GenerateIdeasViewTests(AIIdeasTestCase):
    @override_settings(OPENAI_API_KEY="")
    def test_returns_503_without_api_key(self):
        response = self.client.post(self.generate_url, {"topic": "coffee"})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(Idea.objects.exists())

    def test_get_not_allowed(self):
        response = self.client.get(self.generate_url)
        self.assertEqual(response.status_code, 405)

    @patch(OPENAI_CLIENT)
    def test_creates_ideas_in_requested_group(self, mock_client_cls):
        first = IdeaGroup.objects.create(workspace=self.workspace, name="Backlog", position=0)
        target = IdeaGroup.objects.create(workspace=self.workspace, name="This week", position=1)
        Idea.objects.create(workspace=self.workspace, title="Existing idea", group=first)
        parse = mock_client_cls.return_value.chat.completions.parse
        parse.return_value = _parsed_response(
            [
                GeneratedIdea(title="Latte art reel", description="Short video.", tags=["video", "coffee"]),
                GeneratedIdea(title="Bean origin carousel", description="Five slides.", tags=["carousel"]),
                GeneratedIdea(title="   ", description="Blank titles are dropped.", tags=[]),
            ]
        )

        response = self.client.post(self.generate_url, {"topic": "coffee shop", "group": str(target.id)})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "count": 2})
        created = Idea.objects.filter(group=target).order_by("title")
        self.assertEqual([i.title for i in created], ["Bean origin carousel", "Latte art reel"])
        latte = created.get(title="Latte art reel")
        self.assertEqual(latte.author, self.user)
        self.assertEqual(latte.tags, ["video", "coffee"])
        self.assertEqual(
            set(Tag.objects.for_workspace(self.workspace.id).values_list("name", flat=True)),
            {"video", "coffee", "carousel"},
        )

        kwargs = parse.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5.4-mini")
        self.assertEqual(kwargs["response_format"], GeneratedIdeas)
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        prompt = kwargs["messages"][1]["content"]
        self.assertIn("coffee shop", prompt)
        self.assertIn("Existing idea", prompt)
        self.assertIn("Test Workspace", prompt)

    @patch(OPENAI_CLIENT)
    def test_defaults_to_first_group(self, mock_client_cls):
        first = IdeaGroup.objects.create(workspace=self.workspace, name="Backlog", position=0)
        IdeaGroup.objects.create(workspace=self.workspace, name="Later", position=1)
        mock_client_cls.return_value.chat.completions.parse.return_value = _parsed_response(
            [GeneratedIdea(title="One", description="", tags=[])]
        )

        response = self.client.post(self.generate_url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Idea.objects.get().group, first)

    @patch(OPENAI_CLIENT)
    def test_refusal_returns_error_and_creates_nothing(self, mock_client_cls):
        mock_client_cls.return_value.chat.completions.parse.return_value = _parsed_response(
            None, refusal="I can't help with that."
        )

        response = self.client.post(self.generate_url, {"topic": "something"})

        self.assertEqual(response.status_code, 502)
        self.assertIn("declined", response.json()["error"])
        self.assertFalse(Idea.objects.exists())

    @patch(OPENAI_CLIENT)
    def test_bad_api_key_returns_readable_error(self, mock_client_cls):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_client_cls.return_value.chat.completions.parse.side_effect = openai.AuthenticationError(
            "Incorrect API key provided",
            response=httpx.Response(401, request=request),
            body=None,
        )

        response = self.client.post(self.generate_url)

        self.assertEqual(response.status_code, 502)
        self.assertIn("OPENAI_API_KEY", response.json()["error"])
        self.assertFalse(Idea.objects.exists())

    @patch(OPENAI_CLIENT)
    def test_cannot_target_another_workspaces_group(self, mock_client_cls):
        other_ws = Workspace.objects.create(organization=self.org, name="Other")
        foreign_group = IdeaGroup.objects.create(workspace=other_ws, name="Theirs", position=0)
        mock_client_cls.return_value.chat.completions.parse.return_value = _parsed_response(
            [GeneratedIdea(title="Mine", description="", tags=[])]
        )

        response = self.client.post(self.generate_url, {"group": str(foreign_group.id)})

        self.assertEqual(response.status_code, 200)
        idea = Idea.objects.get()
        self.assertEqual(idea.workspace, self.workspace)
        self.assertIsNone(idea.group)
