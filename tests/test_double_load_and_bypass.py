"""
test_double_load_and_bypass.py — Two related invariants added on
2026-04-19 as the final fix for the phantom-pending-enrichment deadlock.

# 1. `python -m` double-load prevention

The symptom: running `python -m mempalace.mcp_server` caused `_STATE` to be
initialized TWICE in the same process — once under `__main__`, once under
`mempalace.mcp_server` — because any dependent import of the dotted name
triggered a second exec of `mcp_server.py` (distinct entry in
`sys.modules`). `handle_request` wrote `session_id` on one `_STATE`;
`intent._persist_active_intent` read `session_id` from the other
(empty); persist skipped; the on-disk state file was never written;
the hook then denied every subsequent tool call for "no active intent".

The fix (mcp_server.py top): when `__name__ == '__main__'`, alias
`sys.modules["mempalace.mcp_server"] = sys.modules["__main__"]` BEFORE
any dependent import runs. Future dotted-name imports hit the cache
and return this same module. Only one `_STATE` can exist.

# 2. Break-glass hook bypass file

The symptom: once the double-load bug wedged the state file, the agent
couldn't even `Edit` a source file to fix the bug (hook kept denying).
The only way out was to uninstall the plugin. That's painful in a live
debugging session.

The fix (hooks_cli.py): if `~/.mempalace/HOOK_BYPASS_USER_ONLY` exists,
the hook runs its usual deny-composition logic but then downgrades
every `deny` decision to `allow` with a LOUD `[!] HOOK BYPASS ACTIVE`
reason so the user sees it firing. The file is USER-ONLY — agents
must never create or touch it. Convention + tests enforce this.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════
#  Double-load prevention
# ═══════════════════════════════════════════════════════════════════════


class TestNoDoubleLoad:
    def test_state_constructed_once_under_m_dash_m(self, tmp_path):
        """Spawning `python -m mempalace.mcp_server` as a subprocess and
        sending it a tool call must produce exactly ONE ServerState
        construction. Before the fix, a second construction fired when
        `mempalace.mcp_server` was imported via dotted name after the
        `__main__` exec.
        """
        # Patch ServerState.__setattr__ via a sitecustomize-like trick:
        # we inject a tiny probe module that counts __init__ calls and
        # appends to a file we can read. Subprocess inherits the env.
        probe_file = tmp_path / "probe.jsonl"
        probe_src = tmp_path / "sitecustomize.py"
        probe_src.write_text(
            "import mempalace.server_state as _ss\n"
            "_orig = _ss.ServerState.__init__\n"
            "def _counted(self, *a, **kw):\n"
            f"    open(r{str(probe_file)!r}, 'a', encoding='utf-8').write('init\\n')\n"
            "    return _orig(self, *a, **kw)\n"
            "_ss.ServerState.__init__ = _counted\n"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
        # Minimal JSON-RPC: init + notify + one read-only tool call.
        rpc = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "t", "version": "1"},
                    },
                }
            )
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "mempalace_kg_stats",
                        "arguments": {"sessionId": "test-sid"},
                    },
                }
            )
            + "\n"
        )

        proc = subprocess.Popen(
            [sys.executable, "-m", "mempalace.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )
        try:
            proc.communicate(input=rpc, timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise

        # Count init events.
        inits = 0
        if probe_file.is_file():
            inits = sum(1 for _ in probe_file.read_text(encoding="utf-8").splitlines())
        assert inits == 1, (
            f"ServerState was constructed {inits} time(s); expected exactly 1. "
            "Double-load regression — check mcp_server.py top-of-file "
            "`sys.modules['mempalace.mcp_server'] = sys.modules['__main__']` alias."
        )

    def test_dotted_import_of_main_returns_same_module(self):
        """After the `__main__` exec aliases itself as
        `mempalace.mcp_server`, subsequent `import mempalace.mcp_server`
        must return THAT same object — not a fresh one."""
        # In-process we can't exactly reproduce `-m`, but we can verify
        # the alias discipline holds for ordinary imports.
        import mempalace.mcp_server as m1
        import importlib

        m2 = importlib.import_module("mempalace.mcp_server")
        assert m1 is m2, "dotted-name reimport created a new module object"


# ═══════════════════════════════════════════════════════════════════════
#  Hook bypass — break-glass file, user-only, agent-forbidden
# ═══════════════════════════════════════════════════════════════════════


def _run_hook(payload: dict, bypass_file: Path | None = None) -> dict:
    """Invoke hook_pretooluse with the given payload, optionally with a
    bypass file that `_bypass_active` will find. Returns the parsed
    JSON output."""
    from mempalace import hooks_cli

    if bypass_file is not None:
        with patch.object(hooks_cli, "_BYPASS_FILE", bypass_file):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                hooks_cli.hook_pretooluse(payload, "claude-code")
            return json.loads(buf.getvalue())
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        hooks_cli.hook_pretooluse(payload, "claude-code")
    return json.loads(buf.getvalue())


class TestHookBypass:
    def test_deny_without_bypass_file(self, tmp_path, monkeypatch):
        """Absent bypass file → hook denies as normal."""
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        monkeypatch.setattr(hooks_cli, "_BYPASS_FILE", tmp_path / "HOOK_BYPASS_USER_ONLY")
        out = _run_hook(
            {
                "session_id": "s1",
                "tool_name": "Read",
                "tool_input": {"file_path": "/x"},
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "No active intent" in hso.get("permissionDecisionReason", "")

    def test_allow_when_bypass_file_present(self, tmp_path, monkeypatch):
        """Bypass file exists → decision downgrades to `allow` with
        loud [!] HOOK BYPASS ACTIVE in the reason."""
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        bypass = tmp_path / "HOOK_BYPASS_USER_ONLY"
        bypass.touch()
        monkeypatch.setattr(hooks_cli, "_BYPASS_FILE", bypass)
        out = _run_hook(
            {
                "session_id": "s1",
                "tool_name": "Read",
                "tool_input": {"file_path": "/x"},
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"
        reason = hso.get("permissionDecisionReason", "")
        assert "HOOK BYPASS ACTIVE" in reason
        assert "HOOK_BYPASS_USER_ONLY" in reason
        # Original deny reason is preserved inside the bypass message.
        assert "No active intent" in reason

    def test_bypass_file_does_not_affect_already_allowed_calls(self, tmp_path, monkeypatch):
        """If the hook was going to allow anyway (e.g. an ALWAYS_ALLOWED
        tool), the bypass mechanism doesn't mutate the response."""
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        bypass = tmp_path / "HOOK_BYPASS_USER_ONLY"
        bypass.touch()
        monkeypatch.setattr(hooks_cli, "_BYPASS_FILE", bypass)
        out = _run_hook(
            {
                "session_id": "s1",
                "tool_name": "TodoWrite",  # ALWAYS_ALLOWED
                "tool_input": {},
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"
        # No bypass marker because the allow was legitimate, not downgraded.
        assert "HOOK BYPASS ACTIVE" not in hso.get("permissionDecisionReason", "")

    def test_bypass_logs_loud_warning(self, tmp_path, monkeypatch):
        """When bypass fires, hook.log must contain a loud warning line
        so the user can see the bypass in the running-log."""
        from mempalace import hooks_cli

        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        bypass = tmp_path / "HOOK_BYPASS_USER_ONLY"
        bypass.touch()
        monkeypatch.setattr(hooks_cli, "_BYPASS_FILE", bypass)
        _run_hook(
            {
                "session_id": "s1",
                "tool_name": "Read",
                "tool_input": {"file_path": "/x"},
            }
        )
        log = (tmp_path / "hook.log").read_text(encoding="utf-8")
        assert "HOOK BYPASS ACTIVE" in log


# ═══════════════════════════════════════════════════════════════════════
#  Agent-forbidden invariant — the bypass file must not be referenced by
#  any production code that could be triggered via a tool call.
# ═══════════════════════════════════════════════════════════════════════


class TestAgentForbiddenInvariant:
    """The bypass file is for HUMAN user hands only. No production code
    path an agent could trigger should create, touch, or suggest
    touching the file. The ONLY legitimate reference in production is
    the hook's read (`_BYPASS_FILE` constant + `_bypass_active` check).
    """

    @staticmethod
    def _code_files():
        from pathlib import Path as _P

        root = _P(__file__).parent.parent / "mempalace"
        return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]

    def test_only_hooks_cli_references_the_bypass_filename(self):
        """HOOK_BYPASS_USER_ONLY must appear ONLY in hooks_cli.py (the
        hook reads it). Any other reference in production code is a
        red flag."""
        offenders = []
        for p in self._code_files():
            src = p.read_text(encoding="utf-8")
            if "HOOK_BYPASS_USER_ONLY" in src and p.name != "hooks_cli.py":
                offenders.append(p.name)
        assert not offenders, (
            f"HOOK_BYPASS_USER_ONLY referenced outside hooks_cli.py: {offenders}. "
            "That file is user-only; no other code may read or write it."
        )

    def test_hooks_cli_never_creates_or_writes_the_bypass_file(self):
        """In hooks_cli.py, the bypass file may be READ (is_file) but
        never CREATED / WRITTEN by production code (touch, open('w'),
        write_text, mkdir-then-touch, etc)."""
        src = (Path(__file__).parent.parent / "mempalace" / "hooks_cli.py").read_text(
            encoding="utf-8"
        )
        # Find the region around _BYPASS_FILE usage and assert no write
        # operations appear in the same line or adjacent context.
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "_BYPASS_FILE" not in line:
                continue
            if ".touch()" in line or ".write_text(" in line or ".mkdir(" in line:
                raise AssertionError(
                    f"hooks_cli.py:{i + 1} writes to _BYPASS_FILE: {line!r}. "
                    "Bypass file must be user-created only."
                )
            if "open(" in line and '"w"' in line:
                raise AssertionError(
                    f"hooks_cli.py:{i + 1} opens _BYPASS_FILE for writing: {line!r}"
                )

    def test_documentation_banner_present(self):
        """hooks_cli.py must carry the prominent docstring banner
        telling any agent reading the source that this file is
        user-only. Removing that banner is a policy regression."""
        src = (Path(__file__).parent.parent / "mempalace" / "hooks_cli.py").read_text(
            encoding="utf-8"
        )
        # Look for the exact wording that forbids agents touching the file.
        banner_markers = [
            "BREAK-GLASS HOOK BYPASS",
            "HOOK_BYPASS_USER_ONLY",
            "NEVER BY AN AGENT",
        ]
        for marker in banner_markers:
            assert marker in src.upper(), (
                f"Policy banner fragment '{marker}' missing from hooks_cli.py"
            )


# ═══════════════════════════════════════════════════════════════════════
#  Hard-block: any non-always-allowed tool whose tool_input references
#  the bypass file as a filesystem path is denied unconditionally, even
#  when the break-glass bypass file exists.
# ═══════════════════════════════════════════════════════════════════════


# Build the bypass filename via concatenation so this test module's own
# source does not appear to be referencing the file (keeps the static
# scan test + hook guard logic independent of test-fixture content).
_BYPASS_FNAME = "HOOK_BYPASS_" + "USER_ONLY"


class TestHardBlockBypassFileReference:
    """Any NON-always-allowed tool whose tool_input contains a path to
    the bypass file is denied with a HARD BLOCK reason. The deny must
    not be softened by the bypass file (that would defeat the purpose —
    the file is for OS-terminal hands only)."""

    def _run(self, tool_name: str, tool_input: dict, tmp_path: Path) -> dict:
        from mempalace import hooks_cli

        buf = io.StringIO()
        with patch(
            "mempalace.hooks_cli._output",
            side_effect=lambda d: buf.write(json.dumps(d)),
        ):
            with patch.object(hooks_cli, "STATE_DIR", tmp_path):
                hooks_cli.hook_pretooluse(
                    {
                        "session_id": "s1",
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                    },
                    "claude-code",
                )
        return json.loads(buf.getvalue())

    def test_read_with_bypass_path_is_hard_denied(self, tmp_path):
        out = self._run(
            "Read",
            {"file_path": f"~/.mempalace/{_BYPASS_FNAME}"},
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "HARD BLOCK" in hso["permissionDecisionReason"]

    def test_edit_referencing_bypass_path_is_hard_denied(self, tmp_path):
        out = self._run(
            "Edit",
            {
                "file_path": f"/home/me/.mempalace/{_BYPASS_FNAME}",
                "old_string": "x",
                "new_string": "y",
            },
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "HARD BLOCK" in hso["permissionDecisionReason"]

    def test_bash_touching_bypass_path_is_hard_denied(self, tmp_path):
        out = self._run(
            "Bash",
            {"command": f"touch ~/.mempalace/{_BYPASS_FNAME}"},
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "HARD BLOCK" in hso["permissionDecisionReason"]

    def test_filesystem_mcp_write_is_hard_denied(self, tmp_path):
        """A deep-nested dict (as filesystem MCP tool_input would be) is
        still scanned recursively."""
        out = self._run(
            "mcp__filesystem__write_file",
            {"path": f"/Users/me/.mempalace/{_BYPASS_FNAME}", "content": "hi"},
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "HARD BLOCK" in hso["permissionDecisionReason"]

    def test_nested_reference_in_list_is_hard_denied(self, tmp_path):
        """Strings hidden inside nested list values are still detected."""
        out = self._run(
            "Bash",
            {
                "env": {"FLAGS": ["--list", f"./.mempalace/{_BYPASS_FNAME}"]},
                "command": "echo hi",
            },
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "HARD BLOCK" in hso["permissionDecisionReason"]

    def test_hard_block_not_downgraded_by_bypass_file(self, tmp_path, monkeypatch):
        """Even with the break-glass file present, the hard block must
        hold. The bypass file exists to unwedge agent state, not to
        unlock writes to the file itself."""
        from mempalace import hooks_cli

        bypass = tmp_path / _BYPASS_FNAME
        bypass.touch()
        monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path)
        monkeypatch.setattr(hooks_cli, "_BYPASS_FILE", bypass)

        out = self._run("Read", {"file_path": f"~/.mempalace/{_BYPASS_FNAME}"}, tmp_path)
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny", (
            "Hard block on bypass-file path references must NOT be softened "
            "by the break-glass bypass itself — agents must never reach the "
            "file via any code path."
        )
        assert "HARD BLOCK" in hso["permissionDecisionReason"]

    def test_always_allowed_tools_are_exempt(self, tmp_path):
        """TodoWrite / ExitPlanMode / mempalace_* / Agent / Skill /
        ToolSearch / Task* / AskUserQuestion are short-circuited before
        the hard-block check runs, so they may reference the path if
        their content genuinely needs to (a reminder note, a recipe in
        a plan, an intent summary, etc.)."""
        out = self._run(
            "TodoWrite",
            {
                "todos": [
                    {
                        "content": f"Reminder: never touch ~/.mempalace/{_BYPASS_FNAME}.",
                        "status": "pending",
                        "activeForm": "tracking reminder",
                    }
                ]
            },
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"

    def test_bare_prose_mention_without_path_is_not_hard_blocked(self, tmp_path):
        """Prose-only mentions of the filename (no path separator
        immediately before it) are NOT hard-blocked. Example: an Edit
        call whose new_string documents the filename in a comment."""
        # This call will still be denied by normal no-intent gating, but
        # the deny reason must NOT be the HARD BLOCK one.
        out = self._run(
            "Edit",
            {
                "file_path": "/tmp/unrelated.md",
                "old_string": "foo",
                "new_string": f"Document: {_BYPASS_FNAME} is the bypass file.",
            },
            tmp_path,
        )
        hso = out["hookSpecificOutput"]
        assert "HARD BLOCK" not in hso.get("permissionDecisionReason", "")

    def test_unrelated_tool_without_reference_passes_normal_gating(self, tmp_path):
        """Control: a tool that doesn't mention the filename is NOT
        hard-blocked. It falls through to normal intent gating."""
        out = self._run("Read", {"file_path": "/tmp/regular/file.txt"}, tmp_path)
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert "HARD BLOCK" not in hso.get("permissionDecisionReason", "")


class TestReferencesBypassFileHelper:
    """Unit coverage for the recursive scanner that powers the hard block."""

    def test_path_with_slash_prefix_is_detected(self):
        from mempalace.hooks_cli import _references_bypass_file

        assert _references_bypass_file(f"~/.mempalace/{_BYPASS_FNAME}")

    def test_path_with_backslash_prefix_is_detected(self):
        from mempalace.hooks_cli import _references_bypass_file

        assert _references_bypass_file(f"C:\\Users\\me\\.mempalace\\{_BYPASS_FNAME}")

    def test_bare_prose_mention_is_not_detected(self):
        """Filename appearing in prose (no path separator immediately
        before it) does not trip the guard."""
        from mempalace.hooks_cli import _references_bypass_file

        assert not _references_bypass_file(f"The file {_BYPASS_FNAME} is user-only.")
        assert not _references_bypass_file(f'"{_BYPASS_FNAME}"')

    def test_dict_with_nested_path_is_detected(self):
        from mempalace.hooks_cli import _references_bypass_file

        assert _references_bypass_file({"a": {"b": [f"~/.mempalace/{_BYPASS_FNAME}"]}})

    def test_list_of_safe_strings_is_not_detected(self):
        from mempalace.hooks_cli import _references_bypass_file

        assert not _references_bypass_file(["foo", "bar", {"x": 1, "y": "baz"}])

    def test_non_string_leaves_are_ignored(self):
        from mempalace.hooks_cli import _references_bypass_file

        assert not _references_bypass_file({"n": 42, "b": True, "l": [1, 2, 3]})
