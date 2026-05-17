"""
Tests for db.py and key_manager.py.

These exercise the new _connection() context manager (auto-commit,
auto-rollback, auto-close) and the CRUD operations on top of it.
All tests use the temp_db fixture so the real quorumai.db is never touched.
"""
import pytest


class TestContextManager:
    """The new _connection() context manager is the heart of the v15 refactor."""

    def test_commits_on_success(self, temp_db, test_user):
        import db
        with db._connection() as conn:
            conn.execute(
                "INSERT INTO user_settings (user_id, meta_prompt) VALUES (?, ?)",
                (test_user["id"], "committed"),
            )
        # The value must be visible to a new connection — meaning the commit happened.
        assert db.load_meta_prompt(test_user["id"]) == "committed"

    def test_rolls_back_on_exception(self, temp_db, test_user):
        """If the with-block raises, the transaction must not commit."""
        import db
        with pytest.raises(RuntimeError):
            with db._connection() as conn:
                conn.execute(
                    "INSERT INTO user_settings (user_id, meta_prompt) VALUES (?, ?)",
                    (test_user["id"], "should-rollback"),
                )
                raise RuntimeError("force a rollback")
        # The insert must have been rolled back
        assert db.load_meta_prompt(test_user["id"]) == ""

    def test_row_factory_returns_dict_like_rows(self, temp_db, test_user):
        import db
        with db._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (test_user["id"],)).fetchone()
        # Row should be dict-accessible thanks to row_factory = sqlite3.Row
        assert row["email"] == "pytest@example.com"

    def test_foreign_keys_are_enforced(self, temp_db):
        """The PRAGMA foreign_keys = ON must be active inside the context."""
        import db
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            with db._connection() as conn:
                # user_id 99999 doesn't exist — FK should fire
                conn.execute(
                    "INSERT INTO api_keys (user_id, provider_name, encrypted_key, model_id, endpoint_url) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (99999, "Bad", "x", "y", "z"),
                )


class TestUserCrud:

    def test_get_or_create_user_creates_new(self, temp_db):
        import db
        user = db.get_or_create_user(github_id=1, email="a@b.com", display_name="A", avatar_url="")
        assert user["github_id"] == 1
        assert user["email"] == "a@b.com"

    def test_get_or_create_user_is_idempotent(self, temp_db):
        """Calling twice with the same github_id must return the same row, not create a duplicate."""
        import db
        u1 = db.get_or_create_user(github_id=1, email="a@b.com", display_name="A", avatar_url="")
        u2 = db.get_or_create_user(github_id=1, email="a@b.com", display_name="A", avatar_url="")
        assert u1["id"] == u2["id"]

    def test_get_or_create_user_updates_profile(self, temp_db):
        """Subsequent calls update email/display_name to the latest value from GitHub."""
        import db
        db.get_or_create_user(github_id=1, email="old@b.com", display_name="Old", avatar_url="")
        u = db.get_or_create_user(github_id=1, email="new@b.com", display_name="New", avatar_url="")
        assert u["email"] == "new@b.com"
        assert u["display_name"] == "New"


class TestHistory:

    def _entry(self, summary="test"):
        return {
            "summary": summary, "full_prompt": "x", "prompt": "x",
            "mode": "Arena", "quality": 50, "complexity": "Simple",
            "synthesis": None, "responses": {"M1": "hello"},
        }

    def test_save_and_load_round_trip(self, temp_db, test_user):
        import db
        db.save_history_entry(test_user["id"], self._entry("first"))
        history = db.load_history(test_user["id"])
        assert len(history) == 1
        assert history[0]["summary"] == "first"
        assert history[0]["responses"] == {"M1": "hello"}

    def test_history_is_isolated_per_user(self, temp_db, test_user):
        """User A must not see User B's history."""
        import db
        other = db.get_or_create_user(github_id=2, email="other@x.com", display_name="Other", avatar_url="")
        db.save_history_entry(test_user["id"], self._entry("mine"))
        db.save_history_entry(other["id"], self._entry("theirs"))
        assert [h["summary"] for h in db.load_history(test_user["id"])] == ["mine"]
        assert [h["summary"] for h in db.load_history(other["id"])] == ["theirs"]

    def test_clear_history_wipes_only_target_user(self, temp_db, test_user):
        import db
        other = db.get_or_create_user(github_id=2, email="other@x.com", display_name="Other", avatar_url="")
        db.save_history_entry(test_user["id"], self._entry())
        db.save_history_entry(other["id"], self._entry())
        db.clear_history(test_user["id"])
        assert db.load_history(test_user["id"]) == []
        assert len(db.load_history(other["id"])) == 1


class TestMetaPrompt:

    def test_round_trip(self, temp_db, test_user):
        import db
        db.save_meta_prompt(test_user["id"], "Always reply in haiku.")
        assert db.load_meta_prompt(test_user["id"]) == "Always reply in haiku."

    def test_upsert_overwrites_existing(self, temp_db, test_user):
        """save_meta_prompt is an UPSERT — the second call must overwrite, not duplicate."""
        import db
        db.save_meta_prompt(test_user["id"], "first")
        db.save_meta_prompt(test_user["id"], "second")
        assert db.load_meta_prompt(test_user["id"]) == "second"

    def test_load_when_unset_returns_empty_string(self, temp_db, test_user):
        import db
        assert db.load_meta_prompt(test_user["id"]) == ""


class TestKeyManager:

    def test_save_and_retrieve_encrypts_at_rest(self, temp_db, test_user):
        """The plaintext key must never appear in the on-disk encrypted_key column."""
        import db
        import key_manager
        key_id, _ = key_manager.save_key(
            test_user["id"], "Gemini", "Test", "AIza-supersecret-12345",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        # Pull the raw row and confirm plaintext is NOT stored
        with db._connection() as conn:
            row = conn.execute(
                "SELECT encrypted_key FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
        assert "AIza-supersecret-12345" not in row["encrypted_key"]
        # But get_keys must decrypt it correctly
        keys = key_manager.get_keys(test_user["id"])
        assert keys[0]["key"] == "AIza-supersecret-12345"

    def test_masked_key_hides_middle(self, temp_db, test_user):
        import key_manager
        key_manager.save_key(
            test_user["id"], "Gemini", "Test", "AIza-supersecret-12345",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        masked = key_manager.get_keys(test_user["id"])[0]["masked_key"]
        # Should show prefix + last 4 only — never the middle
        assert "supersecret" not in masked
        assert masked.endswith("2345")

    def test_duplicate_alias_rejected(self, temp_db, test_user):
        import key_manager
        key_manager.save_key(
            test_user["id"], "Gemini", "MyKey", "first-key-abc",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        with pytest.raises(ValueError, match="already exists"):
            key_manager.save_key(
                test_user["id"], "Gemini", "MyKey", "different-key-xyz",
                "gemini-2.5-flash", "https://x", "gemini",
            )

    def test_duplicate_plaintext_key_rejected(self, temp_db, test_user):
        """Same key, different alias — should still be rejected (hash-based)."""
        import key_manager
        key_manager.save_key(
            test_user["id"], "Gemini", "AliasA", "same-key-abc",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        with pytest.raises(ValueError, match="already been added"):
            key_manager.save_key(
                test_user["id"], "Gemini", "AliasB", "same-key-abc",
                "gemini-2.5-flash", "https://x", "gemini",
            )

    def test_toggle_flips_active_state(self, temp_db, test_user):
        import key_manager
        key_id, _ = key_manager.save_key(
            test_user["id"], "Gemini", "Test", "first-key-abc",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        assert key_manager.count_active_keys(test_user["id"]) == 1
        key_manager.toggle_key(test_user["id"], key_id)
        assert key_manager.count_active_keys(test_user["id"]) == 0
        key_manager.toggle_key(test_user["id"], key_id)
        assert key_manager.count_active_keys(test_user["id"]) == 1

    def test_delete_key_removes_row(self, temp_db, test_user):
        import key_manager
        key_id, _ = key_manager.save_key(
            test_user["id"], "Gemini", "Test", "first-key-abc",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        assert len(key_manager.get_keys(test_user["id"])) == 1
        key_manager.delete_key(test_user["id"], key_id)
        assert key_manager.get_keys(test_user["id"]) == []

    def test_user_cannot_delete_other_users_keys(self, temp_db, test_user):
        """delete_key must enforce user ownership (user_id in WHERE clause)."""
        import db
        import key_manager
        other = db.get_or_create_user(github_id=2, email="other@x.com", display_name="Other", avatar_url="")
        key_id, _ = key_manager.save_key(
            other["id"], "Gemini", "Test", "other-key-abc",
            "gemini-2.5-flash", "https://x", "gemini",
        )
        # test_user tries to delete other_user's key — should silently fail
        key_manager.delete_key(test_user["id"], key_id)
        # other still has their key
        assert len(key_manager.get_keys(other["id"])) == 1


class TestReasoningModelKeyTest:
    """
    Covers the merged fix where _test_openai_compat sends `max_completion_tokens`
    for OpenAI reasoning models (o-series, gpt-5-*) and the legacy `max_tokens`
    for older chat-completion models. Without this branch, key validation for
    reasoning models would fail with an HTTP 400 from OpenAI.
    """

    def _captured_payload(self, requests_post_call):
        """Pull the JSON payload that key_manager actually sent."""
        # requests.post is called as: post(url, headers=..., json=PAYLOAD, timeout=...)
        return requests_post_call.kwargs.get("json")

    def test_o_series_model_uses_max_completion_tokens(self, monkeypatch):
        """o-series reasoning models must send max_completion_tokens, not max_tokens."""
        from unittest.mock import MagicMock
        import key_manager
        fake_resp = MagicMock(); fake_resp.status_code = 200
        post_mock = MagicMock(return_value=fake_resp)
        monkeypatch.setattr(key_manager.requests, "post", post_mock)

        key_manager._test_openai_compat("sk-fake", "o3-mini", "https://api.openai.com/v1/chat/completions")

        payload = post_mock.call_args.kwargs["json"]
        assert "max_completion_tokens" in payload, "o-series should use max_completion_tokens"
        assert "max_tokens" not in payload, "o-series should NOT use legacy max_tokens"

    def test_o4_mini_also_uses_max_completion_tokens(self, monkeypatch):
        """The regex must catch o4-mini too, not just o3-mini."""
        from unittest.mock import MagicMock
        import key_manager
        fake_resp = MagicMock(); fake_resp.status_code = 200
        post_mock = MagicMock(return_value=fake_resp)
        monkeypatch.setattr(key_manager.requests, "post", post_mock)

        key_manager._test_openai_compat("sk-fake", "o4-mini", "https://api.openai.com/v1/chat/completions")

        payload = post_mock.call_args.kwargs["json"]
        assert "max_completion_tokens" in payload
        assert "max_tokens" not in payload

    def test_gpt5_family_uses_max_completion_tokens(self, monkeypatch):
        """gpt-5* must also use the new parameter name (matches the ^gpt-5 part of the regex)."""
        from unittest.mock import MagicMock
        import key_manager
        fake_resp = MagicMock(); fake_resp.status_code = 200
        post_mock = MagicMock(return_value=fake_resp)
        monkeypatch.setattr(key_manager.requests, "post", post_mock)

        key_manager._test_openai_compat("sk-fake", "gpt-5-turbo", "https://api.openai.com/v1/chat/completions")

        payload = post_mock.call_args.kwargs["json"]
        assert "max_completion_tokens" in payload
        assert "max_tokens" not in payload

    def test_legacy_chat_models_still_use_max_tokens(self, monkeypatch):
        """gpt-4o and friends must keep using the legacy max_tokens parameter — regression guard."""
        from unittest.mock import MagicMock
        import key_manager
        fake_resp = MagicMock(); fake_resp.status_code = 200
        post_mock = MagicMock(return_value=fake_resp)
        monkeypatch.setattr(key_manager.requests, "post", post_mock)

        key_manager._test_openai_compat("sk-fake", "gpt-4o", "https://api.openai.com/v1/chat/completions")

        payload = post_mock.call_args.kwargs["json"]
        assert "max_tokens" in payload, "Legacy models should use max_tokens"
        assert "max_completion_tokens" not in payload

    def test_non_openai_models_through_compat_endpoint_use_max_tokens(self, monkeypatch):
        """A Groq llama or Mistral model going through the openai_compat path should still get max_tokens."""
        from unittest.mock import MagicMock
        import key_manager
        fake_resp = MagicMock(); fake_resp.status_code = 200
        post_mock = MagicMock(return_value=fake_resp)
        monkeypatch.setattr(key_manager.requests, "post", post_mock)

        key_manager._test_openai_compat("gsk-fake", "llama-3.3-70b-versatile",
                                         "https://api.groq.com/openai/v1/chat/completions")

        payload = post_mock.call_args.kwargs["json"]
        assert "max_tokens" in payload
        assert "max_completion_tokens" not in payload


class TestProviderPresets:
    """The merged PROVIDER_PRESETS must contain the corrected OpenAI model list."""

    def test_openai_preset_lists_o3_mini(self):
        """Teammate fix: o3-mini was added to the OpenAI preset (replacing the non-existent o5-mini)."""
        import key_manager
        assert "o3-mini" in key_manager.PROVIDER_PRESETS["OpenAI"]["models"]

    def test_openai_preset_no_longer_lists_o5_mini(self):
        """Regression guard against the original o5-mini typo."""
        import key_manager
        assert "o5-mini" not in key_manager.PROVIDER_PRESETS["OpenAI"]["models"]

    def test_o4_mini_display_name_preserved(self):
        """Merge integrity: o4-mini's display name should NOT have been lost during the merge."""
        import key_manager
        assert key_manager.MODEL_DISPLAY_NAMES.get("o4-mini") == "o4 Mini"

    def test_o3_mini_has_display_name(self):
        """Merge integrity: o3-mini was added with a display name by the teammates."""
        import key_manager
        assert "o3-mini" in key_manager.MODEL_DISPLAY_NAMES


class TestTimestampHelper:
    """The _format_timestamp helper is now shared between load_history and load_project_entries."""

    def test_formats_valid_iso_timestamp(self):
        import db
        assert db._format_timestamp("2025-05-14 13:45:00.123") == "14.05.2025 · 13:45:00"

    def test_falls_back_gracefully_on_garbage(self):
        import db
        assert db._format_timestamp("not-a-timestamp") == "not-a-timestamp"

    def test_handles_empty_string(self):
        import db
        assert db._format_timestamp("") == ""

    def test_handles_none_safely(self):
        """Defensive: callers might pass None in degenerate cases."""
        import db
        # The helper should not crash even if called with None
        assert db._format_timestamp(None) == ""
