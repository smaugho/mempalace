"""Pure-function half of the version-consistency suite -- pyproject vs __version__.

The MCP-handshake variant lives in tests/integration/test_version_consistency.py
because it imports mempalace.mcp_server (heavy module).
"""

import re
from pathlib import Path

import pytest

from mempalace import __version__


def _expected_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match is not None, "Could not find project version in pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject():
    assert __version__ == _expected_version()


pytestmark = pytest.mark.unit
