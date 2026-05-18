import pytest

import os
import json
import tempfile
from mempalace.config import MempalaceConfig


def test_default_config():
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert "palace" in cfg.palace_path
    assert cfg.collection_name == "mempalace_records"


def test_config_from_file():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({"palace_path": "/custom/palace"}, f)
    cfg = MempalaceConfig(config_dir=tmpdir)
    assert cfg.palace_path == "/custom/palace"


def test_env_override():
    os.environ["MEMPALACE_PALACE_PATH"] = "/env/palace"
    cfg = MempalaceConfig(config_dir=tempfile.mkdtemp())
    assert cfg.palace_path == "/env/palace"
    del os.environ["MEMPALACE_PALACE_PATH"]


def test_init():
    tmpdir = tempfile.mkdtemp()
    cfg = MempalaceConfig(config_dir=tmpdir)
    cfg.init()
    assert os.path.exists(os.path.join(tmpdir, "config.json"))


# ---------------------------------------------------------------------
# FINDING #N regression (v3.7.28, 2026-05-18, Adrian's third-pass
# audit): a v3.7.23 diary entry was stored in the live KG with bytes
# c3 83 c2 a2 c3 a2 2c c2 ac c3 a2 e2 82 ac (14 bytes) where the
# original em-dash U+2014 had been double-cp1252-mojibake'd. Root
# cause: mcp_server.py main() did not reconfigure sys.stdin/sys.stdout
# to utf-8, so on Windows the JSON-RPC byte stream came in via cp1252
# and Anthropic MCP payloads with unicode chars (em-dashes etc.) got
# mojibake'd BEFORE reaching sanitize_content.
#
# The fix is in mcp_server.py main() (sys.stdin.reconfigure(utf-8)).
# These tests lock the downstream contract: when properly-decoded
# Unicode text reaches sanitize_content, the punct fold reduces it
# to ASCII deterministically so the stored bytes are safe + greppable.
# If any future refactor drops a mapping or short-circuits the fold,
# this test fails loud instead of silently re-introducing the
# mojibake class.
#
# NOTE: literal U+2014 and U+2013 in Python source are blocked by
# test_no_em_dash_in_any_py_source. We use chr() to construct them at
# runtime so the test source stays gate-clean while still exercising
# the unicode chars.
# ---------------------------------------------------------------------


class TestFindingN_SanitizeContentUnicodeFold:
    def test_em_dash_folds_to_double_hyphen(self):
        from mempalace.config import sanitize_content

        em_dash = chr(0x2014)
        out = sanitize_content("hello " + em_dash + " world")
        assert out == "hello -- world"
        assert em_dash not in out

    def test_en_dash_folds_to_hyphen(self):
        from mempalace.config import sanitize_content

        en_dash = chr(0x2013)
        out = sanitize_content("range 1" + en_dash + "5")
        assert out == "range 1-5"
        assert en_dash not in out

    def test_smart_quotes_fold_to_ascii(self):
        from mempalace.config import sanitize_content

        ldq = chr(0x201C)  # left double quote
        rdq = chr(0x201D)  # right double quote
        lsq = chr(0x2018)  # left single quote
        rsq = chr(0x2019)  # right single quote
        out = sanitize_content(ldq + "quoted" + rdq + " and " + lsq + "single" + rsq)
        assert out == "\"quoted\" and 'single'"

    def test_ellipsis_folds_to_three_dots(self):
        from mempalace.config import sanitize_content

        ellipsis = chr(0x2026)
        out = sanitize_content("wait" + ellipsis + " done")
        assert out == "wait... done"

    def test_nbsp_folds_to_space(self):
        from mempalace.config import sanitize_content

        nbsp = chr(0x00A0)
        out = sanitize_content("non" + nbsp + "breaking")
        assert out == "non breaking"

    def test_zero_width_chars_disappear(self):
        from mempalace.config import sanitize_content

        zwsp = chr(0x200B)
        zwnj = chr(0x200C)
        zwj = chr(0x200D)
        bom = chr(0xFEFF)
        out = sanitize_content("a" + zwsp + "b" + zwnj + "c" + zwj + "d" + bom + "e")
        assert out == "abcde"

    def test_combined_punct_folds_deterministically(self):
        """The realistic case Adrian hit: a diary entry full of
        em-dashes + smart quotes + ellipsis. Post-sanitize the
        output must be pure ASCII for these chars so storage bytes
        match the human-readable display."""
        from mempalace.config import sanitize_content

        em = chr(0x2014)
        ldq = chr(0x201C)
        rdq = chr(0x201D)
        ell = chr(0x2026)
        in_str = (
            "v3.7.28 ship"
            + em
            + "a third-pass fix"
            + em
            + "closes "
            + ldq
            + "FINDING #N"
            + rdq
            + " (diary mojibake)"
            + ell
        )
        out = sanitize_content(in_str)
        assert out == ('v3.7.28 ship--a third-pass fix--closes "FINDING #N" (diary mojibake)...')
        # No remaining unicode punct in the output:
        for cp in (0x2014, 0x2013, 0x201C, 0x201D, 0x2018, 0x2019, 0x2026):
            assert chr(cp) not in out, "sanitize_content failed to fold U+%04X" % cp


pytestmark = pytest.mark.unit
