"""
Invariant tests — lock in the single-normalizer + canonical-ID contract
introduced by phases N1..N3 of docs/normalization_root_fix_plan.md.

These tests exist to prevent regression: any future PR that reintroduces a
second normalizer, or that creates a record whose stored ID diverges from
``normalize_entity_name(id)``, will fail here before it ships.
"""

from __future__ import annotations

import re
from pathlib import Path

from mempalace.knowledge_graph import normalize_entity_name


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DIR = _PROJECT_ROOT / "mempalace"

# Any function whose name STARTS with "normalize" or ends with "_slug" is
# suspicious — it's exactly the pattern that bit us (_normalize_memory_slug
# vs normalize_entity_name). Explicit allowlist below enumerates the few
# legitimate names that delegate to normalize_entity_name.
_CANONICAL_NORMALIZERS = {
    # The single source of truth for entity/record/memory identifiers.
    "normalize_entity_name",
    # Predicate canonicalisation — operates on predicates, reuses the same
    # underscore convention internally.
    "_normalize_predicate",
    # _slugify in mcp_server.py — thin wrapper that delegates to
    # normalize_entity_name and applies the max_length cap.
    "_slugify",
    # Session-id sanitiser — operates on Claude Code session IDs (UUIDs),
    # NOT entity IDs; path-traversal defence only.
    "_sanitize_session_id",
    # File-system + content normalizers — different concerns, not ID-related.
    # Kept in the allowlist so the lint focuses only on identifier
    # normalization. If any of these start being used for entity IDs, they
    # must be removed from this list and routed through normalize_entity_name.
    "normalize",  # normalize.py — file-content whitespace normaliser
    "_try_normalize_json",  # normalize.py — JSON-content normaliser
    "_normalize_win_path",  # hooks_cli.py — filesystem path normaliser
    "normalize_include_paths",  # miner.py — glob pattern normaliser
    "_suggest_slug_hint",  # config.py — human-readable hint for error msgs
}

_NORMALIZE_FUNC_PATTERN = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


class TestSingleNormalizer:
    def test_normalize_entity_name_is_idempotent(self):
        """normalize(normalize(x)) == normalize(x) for every representative input.

        Idempotence is the precondition for any callsite to safely
        re-normalize. If this fails, some input sequence causes a fixed
        point divergence and downstream lookups will keep missing.
        """
        samples = [
            "",
            "  ",
            "already_canonical_id",
            "record_ga_agent_foo-bar-baz",
            "record_ga_agent_foo_bar_baz",
            "FlowsevRepository",
            "the_big_report",
            "paperclip-server",
            "D:\\Flowsev\\repo",
            "mixed_case-With.Dots",
            "HTTPSServer42",
        ]
        for s in samples:
            first = normalize_entity_name(s)
            second = normalize_entity_name(first)
            assert first == second, (
                f"Normalization not idempotent for {s!r}: "
                f"normalize(x)={first!r}, normalize(normalize(x))={second!r}"
            )

    def test_normalize_output_has_no_hyphen(self):
        """The canonical form never contains '-'. Any callsite that stores
        the output is therefore automatically in the underscore convention.
        """
        samples = [
            "a-b-c",
            "record_x_foo-bar",
            "already_ok",
            "mixed-chars_123",
            "a.b.c",
            "foo bar",
        ]
        for s in samples:
            out = normalize_entity_name(s)
            assert "-" not in out, f"normalize({s!r}) = {out!r} contains a hyphen"

    def test_no_rival_normalizer_in_source(self):
        """Grep-level lint: no function in ``mempalace/`` other than the
        canonical allowlist may be named ``normalize_*`` or ``_normalize_*``.

        When we discover a new legitimate normalizer (say one for a file-
        format-specific concern that is NOT an entity ID), add it to
        ``_CANONICAL_NORMALIZERS`` with a comment explaining why it's a
        different concern. Otherwise this test will fail and force a DRY
        review before merge.
        """
        offenders: list[tuple[str, str]] = []
        for py in _PACKAGE_DIR.rglob("*.py"):
            # Skip the migration module itself and test files.
            if py.name.startswith("test_"):
                continue
            try:
                text = py.read_text(encoding="utf-8")
            except OSError:
                continue
            for match in _NORMALIZE_FUNC_PATTERN.finditer(text):
                name = match.group(1)
                lower = name.lower()
                if "normalize" in lower or lower.endswith("_slug"):
                    if name not in _CANONICAL_NORMALIZERS:
                        offenders.append((str(py.relative_to(_PROJECT_ROOT)), name))

        assert not offenders, (
            "Rival normalizer detected. Route to normalize_entity_name or add "
            "to _CANONICAL_NORMALIZERS with a comment explaining why this is a "
            "different concern. Offenders:\n  "
            + "\n  ".join(f"{path} :: {name}" for path, name in offenders)
        )


class TestCanonicalIdContract:
    """Given a fresh palace, every ID that any public tool writes must
    satisfy ``stored_id == normalize_entity_name(stored_id)``.

    A single representative path is exercised — record creation via
    ``_add_memory_internal``. Full coverage of every write path is
    handled by the unit tests in test_hyphen_id_migration.py; this test
    is a belt-and-suspenders check at the public-API level.
    """

    def test_record_creation_produces_canonical_id(self, monkeypatch, config, palace_path, kg):
        # Reuses the existing test fixtures' patching of _STATE onto a
        # disposable palace. See tests/conftest.py.
        from tests.test_mcp_server import _MEMORY_CONTEXT, _patch_mcp_server, _get_collection

        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client

        from mempalace.mcp_server import tool_kg_declare_entity

        # A slug with hyphens, CamelCase, and a leading article — the
        # canonical normalizer should collapse all of these to
        # underscores and strip the article.
        result = tool_kg_declare_entity(
            kind="record",
            slug="The-Big-FooBar-Report",
            content="Invariant test record.",
            summary="Invariant test: slug normalization canonical-id check.",
            content_type="fact",
            context=_MEMORY_CONTEXT,
            added_by="test_agent",
        )
        assert result["success"] is True, result
        mid = result["memory_id"]
        assert "-" not in mid, f"Stored memory_id {mid!r} contains a hyphen"
        assert mid == normalize_entity_name(mid), (
            f"Stored memory_id {mid!r} is not canonical; "
            f"normalize(id) = {normalize_entity_name(mid)!r}"
        )
