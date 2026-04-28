"""
test_link_author_api_key.py -- API key validation exit paths.

Covers every failure mode of ``link_author._validate_api_key``:
present / format / works. Each branch must exit with a specific,
actionable code so cron/systemd logs distinguish failure modes:

  exit 0 -- success
  exit 2 -- bad/missing/malformed/rejected key  (fix .env)
  exit 3 -- Anthropic API unreachable           (fix network)

All SDK calls are mocked -- tests never hit the live API. See
docs/link_author_plan.md §2.8 + docs/link_author_scheduling.md.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import anthropic  # import for exception classes; network isn't touched

from mempalace import link_author


# ─────────────────────────────────────────────────────────────────────
# Default config used by every test (isolated per-test via copy)
# ─────────────────────────────────────────────────────────────────────


def _cfg():
    return {
        "api_key_env": "ANTHROPIC_API_KEY",
        "jury_execution_model": "claude-haiku-4-5",
    }


# ─────────────────────────────────────────────────────────────────────
# 1. Present -- missing env var → exit 2
# ─────────────────────────────────────────────────────────────────────


class TestKeyMissing:
    def test_empty_env_exits_two_with_not_set_message(self, monkeypatch, capsys):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_BAD_KEY
        err = capsys.readouterr().err
        assert "ANTHROPIC_API_KEY not set" in err
        # Docs pointer is actionable.
        assert "console.anthropic.com" in err

    def test_empty_string_env_exits_two(self, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_BAD_KEY
        assert "not set" in capsys.readouterr().err

    def test_whitespace_only_env_exits_two(self, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   \t\n ")
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_BAD_KEY
        assert "not set" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────
# 2. Format -- wrong prefix → exit 2
# ─────────────────────────────────────────────────────────────────────


class TestKeyMalformed:
    def test_wrong_prefix_exits_two(self, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "foo-bar-baz")
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_BAD_KEY
        err = capsys.readouterr().err
        assert "does not look like a valid Anthropic key" in err
        assert "sk-ant-" in err

    def test_similar_looking_key_still_rejected(self, monkeypatch, capsys):
        """An API key from a different provider happens to start with
        'sk-' but NOT 'sk-ant-'. The prefix check must be strict."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_BAD_KEY
        assert "does not look like" in capsys.readouterr().err


# ─────────────────────────────────────────────────────────────────────
# 3. Works -- ping failures
# ─────────────────────────────────────────────────────────────────────


def _mock_client_factory(monkeypatch, behavior):
    """Install a fake AsyncAnthropic that runs ``behavior(client)`` on ping.

    ``behavior`` is an async callable returning the ping result (or
    raising). This is the core of all 'key present + format OK but
    network/auth weird' tests.
    """
    mock_instance = MagicMock()
    mock_messages = MagicMock()
    mock_messages.create = behavior
    mock_instance.messages = mock_messages

    def fake_factory(*args, **kwargs):
        return mock_instance

    monkeypatch.setattr(anthropic, "AsyncAnthropic", fake_factory)


class TestKeyRejected:
    def test_authentication_error_exits_two(self, monkeypatch, capsys):
        """Key is well-formed but rejected by the API (401)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-revoked123456789")

        async def _raise(**kwargs):
            raise anthropic.AuthenticationError(
                "Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )

        _mock_client_factory(monkeypatch, AsyncMock(side_effect=_raise))
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_BAD_KEY
        err = capsys.readouterr().err
        assert "rejected by API" in err


class TestKeyWorks:
    def test_happy_path_returns_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefghijklmno12345")

        async def _ok(**kwargs):
            return MagicMock()

        _mock_client_factory(monkeypatch, AsyncMock(side_effect=_ok))
        client = link_author._validate_api_key(_cfg())
        assert client is not None


class TestApiUnreachable:
    def test_connection_error_exits_three(self, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefghijklmno12345")

        async def _raise(**kwargs):
            raise anthropic.APIConnectionError(request=MagicMock())

        _mock_client_factory(monkeypatch, AsyncMock(side_effect=_raise))
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_API_DOWN
        err = capsys.readouterr().err
        assert "unreachable" in err.lower()

    def test_transient_5xx_exits_three(self, monkeypatch, capsys):
        """5xx after SDK-level retries becomes exit 3 too -- different
        from a flat network error but same 'API side is broken'
        category, so a single fix-the-API exit code covers both."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefghijklmno12345")

        async def _raise(**kwargs):
            raise anthropic.APIError(
                "internal server error",
                request=MagicMock(),
                body=None,
            )

        _mock_client_factory(monkeypatch, AsyncMock(side_effect=_raise))
        with pytest.raises(SystemExit) as exc:
            link_author._validate_api_key(_cfg())
        assert exc.value.code == link_author.EXIT_API_DOWN


# ─────────────────────────────────────────────────────────────────────
# 4. Logging -- keys are never logged verbatim
# ─────────────────────────────────────────────────────────────────────


class TestKeyNeverLoggedVerbatim:
    def test_mask_helper_hides_prefix_and_body(self):
        key = "sk-ant-supersecretabc123"
        masked = link_author._mask_key(key)
        # The actual secret body must NOT appear in the masked form.
        assert "supersecret" not in masked
        assert "sk-ant-" not in masked
        # Last 4 chars are OK (helps operators visually correlate).
        assert masked.endswith("...c123")
        # Length marker is present so a human can spot truncation.
        assert "len=" in masked

    def test_empty_key_masks_safely(self):
        assert link_author._mask_key("") == "<empty>"

    def test_happy_path_log_line_uses_mask(self, monkeypatch, caplog):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-totallynotreal12345")

        async def _ok(**kwargs):
            return MagicMock()

        _mock_client_factory(monkeypatch, AsyncMock(side_effect=_ok))
        with caplog.at_level("INFO", logger="mempalace.link_author"):
            link_author._validate_api_key(_cfg())
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        # The full secret body must never appear in captured logs.
        assert "totallynotreal" not in log_text


# ─────────────────────────────────────────────────────────────────────
# 5. .env loading -- file contents become environment
# ─────────────────────────────────────────────────────────────────────


class TestDotenvLoading:
    def test_env_file_loaded_before_validation(self, tmp_path, monkeypatch):
        """Drop a .env at <palace>/.env; _load_env should pick it up."""
        # Clear any pre-existing env var so we can verify the file
        # actually provided the value.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        palace = tmp_path / "palace"
        palace.mkdir()
        (palace / ".env").write_text(
            "ANTHROPIC_API_KEY=sk-ant-from_dotenv_file_abcdefg\n",
            encoding="utf-8",
        )
        link_author._load_env(palace_path=str(palace))
        import os

        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-from_dotenv_file_abcdefg"

    def test_missing_env_file_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        link_author._load_env(palace_path=str(tmp_path / "nonexistent"))
        import os

        assert os.environ.get("ANTHROPIC_API_KEY") is None

    def test_file_wins_over_shell_env(self, tmp_path, monkeypatch):
        """The palace .env file is authoritative for ANTHROPIC_API_KEY.
        If both the shell env var AND the .env define the key, the
        file value wins (override=True). This prevents the
        silent-shadow failure mode where an empty or stale shell
        value blocks the CLI from seeing a correctly-formed .env
        (documented as record_ga_agent_env_key_shell_shadowing_diagnostic
        after a live-panic 2026-04-22).
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from_shell_should_lose")
        palace = tmp_path / "palace"
        palace.mkdir()
        (palace / ".env").write_text(
            "ANTHROPIC_API_KEY=sk-ant-from_dotenv_file_wins\n",
            encoding="utf-8",
        )
        link_author._load_env(palace_path=str(palace))
        import os

        # File value overrides the pre-existing shell value.
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-from_dotenv_file_wins"
