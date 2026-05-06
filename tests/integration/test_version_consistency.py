"""Integration half of the version-consistency suite -- MCP initialize handshake.

The pure pyproject-vs-__version__ check lives in tests/unit/test_version_consistency.py.
"""

import re
from pathlib import Path

import pytest

from mempalace.mcp_server import handle_request


def _expected_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match is not None, "Could not find project version in pyproject.toml"
    return match.group(1)


def test_mcp_initialize_reports_package_version():
    response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert response["result"]["serverInfo"]["version"] == _expected_version()


pytestmark = pytest.mark.integration
