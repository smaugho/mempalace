"""
test_mandatory_entities.py — Lock in the "entities are mandatory" rule.

Commit 2 of the link-author redesign raises validate_context's
``entities_min`` default from 0 to 1 and adds a mandatory ``entities``
parameter to tool_declare_operation. Every emit site (declare_intent /
declare_operation / kg_search) now rejects a context that omits entities.

Why it matters: the link-author background process accumulates
Adamic-Adar evidence from (entity, context) co-occurrence. Zero
entities → no candidates → no authored edges → graph stays sparse.
See docs/link_author_plan.md §2.3.
"""

from __future__ import annotations

from mempalace.scoring import validate_context


# ─────────────────────────────────────────────────────────────────────
# validate_context — the shared entrypoint
# ─────────────────────────────────────────────────────────────────────


class TestValidateContextEntitiesRequired:
    def _valid_base(self):
        """A context dict with every other mandatory field filled in.

        Tests mutate entities only so the error signal isolates to that
        field.
        """
        return {
            "queries": ["first perspective", "second perspective"],
            "keywords": ["alpha", "beta"],
        }

    def test_missing_entities_key_rejected(self):
        ctx, err = validate_context(self._valid_base())
        assert ctx is None
        assert err is not None
        assert err["success"] is False
        assert "entities" in err["error"].lower()

    def test_empty_entities_list_rejected(self):
        payload = self._valid_base()
        payload["entities"] = []
        ctx, err = validate_context(payload)
        assert ctx is None
        assert err is not None
        assert "entities" in err["error"].lower()

    def test_entities_as_string_rejected_with_clear_message(self):
        """A bare string like "LoginService" must NOT be auto-wrapped into
        a list — the caller needs to learn the shape, not silently succeed."""
        payload = self._valid_base()
        payload["entities"] = "LoginService"
        ctx, err = validate_context(payload)
        assert ctx is None
        assert err is not None
        assert "list" in err["error"].lower()

    def test_entities_all_whitespace_rejected(self):
        payload = self._valid_base()
        payload["entities"] = ["   ", ""]
        ctx, err = validate_context(payload)
        assert ctx is None
        assert err is not None

    def test_entities_over_max_truncates_to_cap(self):
        """The shared _validate_string_list caps at max rather than rejecting
        (consistent behavior with queries/keywords). The cap is the abuse
        guard; callers that pass too many don't get silent failure, they
        get their top-N."""
        payload = self._valid_base()
        payload["entities"] = [f"ent_{i}" for i in range(15)]  # 15 > max 10
        ctx, err = validate_context(payload)
        assert err is None
        assert len(ctx["entities"]) == 10
        # First 10 preserved in order.
        assert ctx["entities"] == [f"ent_{i}" for i in range(10)]

    def test_single_valid_entity_accepted(self):
        payload = self._valid_base()
        payload["entities"] = ["LoginService"]
        ctx, err = validate_context(payload)
        assert err is None
        assert ctx["entities"] == ["LoginService"]

    def test_max_entities_accepted(self):
        payload = self._valid_base()
        payload["entities"] = [f"ent_{i}" for i in range(10)]
        ctx, err = validate_context(payload)
        assert err is None
        assert len(ctx["entities"]) == 10

    def test_entities_preserve_order_and_filter_empties(self):
        """Empty and whitespace-only entries are dropped; surviving entries
        are preserved verbatim (consistent with queries/keywords)."""
        payload = self._valid_base()
        payload["entities"] = ["alpha", "", "beta", "   "]
        ctx, err = validate_context(payload)
        assert err is None
        # Empties and whitespace-only entries dropped, order preserved.
        assert ctx["entities"] == ["alpha", "beta"]


# ─────────────────────────────────────────────────────────────────────
# Per-tool entrypoints — declare_intent / declare_operation / kg_search
# all route through validate_context, so they inherit the rule. These
# tests assert the rejection reaches the agent with a useful shape.
# ─────────────────────────────────────────────────────────────────────


class TestDeclareOperationEntitiesParam:
    """tool_declare_operation now takes the unified Context shape
    (context={queries, keywords, entities}) so validation flows through
    the same scoring.validate_context() path as every other emit site.
    Tests verify the signature + bound constants remain authoritative."""

    def test_signature_takes_unified_context(self):
        """declare_operation accepts `context`, not bare queries/keywords/entities."""
        import inspect

        from mempalace import intent

        sig = inspect.signature(intent.tool_declare_operation)
        assert "context" in sig.parameters
        # The old bare-params shape is gone — no standalone queries/keywords/entities.
        assert "queries" not in sig.parameters
        assert "keywords" not in sig.parameters
        assert "entities" not in sig.parameters

    def test_min_ops_entities_constant_is_one(self):
        """The MIN_OP_ENTITIES constant documents that zero is NOT a
        valid value, matching the link-author plan §2.3."""
        from mempalace import intent

        assert intent.MIN_OP_ENTITIES == 1

    def test_max_ops_entities_constant_is_ten(self):
        from mempalace import intent

        assert intent.MAX_OP_ENTITIES == 10


# ─────────────────────────────────────────────────────────────────────
# MCP dispatch schema advertises entities as required (reaches agents)
# ─────────────────────────────────────────────────────────────────────


class TestMCPSchemaAdvertisement:
    def test_declare_operation_schema_requires_context_with_entities(self):
        """The agent-facing JSON schema for mempalace_declare_operation
        advertises ``context`` (a dict) as required, and inside it
        ``entities`` as a required array of 1-10 strings. This is the
        unified shape shared by declare_intent / kg_search / kg_add /
        kg_declare_entity / kg_add_batch — ONE context shape across
        every emit site."""
        from mempalace import mcp_server

        schema = mcp_server.TOOLS["mempalace_declare_operation"]["input_schema"]
        assert "context" in schema["required"]
        ctx_spec = schema["properties"]["context"]
        assert ctx_spec["type"] == "object"
        assert "entities" in ctx_spec["required"]
        ent_spec = ctx_spec["properties"]["entities"]
        assert ent_spec["type"] == "array"
        assert ent_spec["minItems"] == 1
        assert ent_spec["maxItems"] == 10
