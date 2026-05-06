"""Regression guard for the 2026-04-20 palace-path mismatch fix.

When the MCP server is launched with --palace, it sets the env var AND
writes ~/.mempalace/hook_state/active_palace.txt. Hook subprocesses
don't inherit the server's environ, so they read the hint file via
MempalaceConfig.palace_path fallback to resolve the SAME palace.

Without this fallback, retrieval in the PreToolUse hook silently hit
the empty default palace (~/.mempalace/palace) and returned zero hits.
"""

import pytest

from mempalace.config import DEFAULT_PALACE_PATH, MempalaceConfig


def test_palace_path_prefers_env_over_hint(tmp_path, monkeypatch):
    """env var still wins when both are set."""
    # Set both; env should take precedence
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", "/env/palace")
    hint_dir = tmp_path / "hook_state"
    hint_dir.mkdir()
    (hint_dir / "active_palace.txt").write_text("/hint/palace", encoding="utf-8")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == "/env/palace"


def test_palace_path_uses_hint_when_no_env(tmp_path, monkeypatch):
    """Hook subprocess: no env, but hint file exists \u2014 use it."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    hint_dir = tmp_path / "hook_state"
    hint_dir.mkdir()
    (hint_dir / "active_palace.txt").write_text("/hint/from/server", encoding="utf-8")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == "/hint/from/server"


def test_palace_path_ignores_empty_hint(tmp_path, monkeypatch):
    """Empty hint file: fall through to config/default."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    hint_dir = tmp_path / "hook_state"
    hint_dir.mkdir()
    (hint_dir / "active_palace.txt").write_text("   \n", encoding="utf-8")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    # Should fall through to default since file_config is empty
    assert cfg.palace_path == DEFAULT_PALACE_PATH


def test_palace_path_falls_back_to_default_no_hint(tmp_path, monkeypatch):
    """No env, no hint, no config.json \u2014 returns DEFAULT_PALACE_PATH."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == DEFAULT_PALACE_PATH


def test_palace_path_hint_strips_whitespace(tmp_path, monkeypatch):
    """Hint file content is stripped before use."""
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)
    monkeypatch.delenv("MEMPAL_PALACE_PATH", raising=False)
    hint_dir = tmp_path / "hook_state"
    hint_dir.mkdir()
    (hint_dir / "active_palace.txt").write_text("  /p/with/whitespace  \n", encoding="utf-8")

    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == "/p/with/whitespace"


pytestmark = pytest.mark.unit
