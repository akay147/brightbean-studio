"""URLs for the analytics page."""

from django.urls import path

from . import views

app_name = "analytics"

urlpatterns = [
    path("<uuid:account_id>/status/", views.analytics_status, name="status"),
    path("", views.analytics_index, name="index"),
    path("post/<uuid:post_id>/", views.post_detail, name="post_detail"),
    path("post/<uuid:post_id>/ai-review/", views.post_ai_review, name="post_ai_review"),
    path("<uuid:account_id>/", views.analytics_account, name="account"),
]
