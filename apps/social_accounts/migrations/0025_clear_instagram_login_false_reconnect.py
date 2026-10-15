from django.db import migrations


def clear_flags(apps, schema_editor):
    # Deleted Instagram posts used to flag their account for a reconnect (Meta's
    # "does not exist ... missing permissions" wording). The next sync re-flags
    # any account whose grant really lacks a scope.
    SocialAccount = apps.get_model("social_accounts", "SocialAccount")
    SocialAccount.objects.filter(platform="instagram_login", analytics_needs_reconnect=True).update(
        analytics_needs_reconnect=False
    )


class Migration(migrations.Migration):
    dependencies = [
        ("social_accounts", "0024_socialaccount_instagram_dm_poll_state"),
    ]

    operations = [
        migrations.RunPython(clear_flags, migrations.RunPython.noop),
    ]
