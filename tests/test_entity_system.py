"""
test_entity_system.py — Tests for entity declaration, collision detection,
merge, predicate constraints, cardinality enforcement, and class inheritance.

Covers the full entity lifecycle introduced in PRs #1-#15 on smaugho/mempalace.
"""

_TEST_BUDGET = {"Read": 20, "Edit": 20, "Bash": 20, "Grep": 20, "Glob": 20, "Write": 20}


def _patch_mcp(monkeypatch, config, kg, palace_path):
    """Patch mcp_server globals and reset declared entities for a clean test."""
    import chromadb
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server._STATE, "config", config)
    monkeypatch.setattr(mcp_server._STATE, "kg", kg)
    monkeypatch.setattr(mcp_server._STATE, "active_intent", None)
    monkeypatch.setattr(mcp_server._STATE, "pending_conflicts", None)
    monkeypatch.setattr(mcp_server._STATE, "declared_entities", set())

    # Ensure entity collection exists in test palace
    client = chromadb.PersistentClient(path=palace_path)
    client.get_or_create_collection("mempalace_records")
    ecol = client.get_or_create_collection("mempalace_entities")

    # Seed agent class + test_agent so added_by validation passes
    kg.add_entity("agent", kind="class", description="An AI agent", importance=5)
    kg.add_entity(
        "test_agent", kind="entity", description="Test agent for unit tests", importance=3
    )
    kg.add_triple("test_agent", "is_a", "agent")
    ecol.upsert(
        ids=["agent", "test_agent"],
        documents=["An AI agent", "Test agent for unit tests"],
        metadatas=[
            {"name": "agent", "kind": "class", "importance": 5},
            {"name": "test_agent", "kind": "entity", "importance": 3, "added_by": "test_agent"},
        ],
    )
    del client

    # Reset session state
    mcp_server._STATE.declared_entities = set()
    mcp_server._STATE.session_id = "test-session"


def _declare(
    name,
    description,
    kind="entity",
    importance=3,
    constraints=None,
    properties=None,
    added_by="test_agent",
    extra_keywords=None,
    extra_views=None,
):
    """Test fixture: build a Context from `description` + sensible defaults
    and call kg_declare_entity. Real callers must pass Context themselves;
    this helper just keeps the test surface compact.
    """
    from mempalace.mcp_server import tool_kg_declare_entity

    # Build a 2-view Context: the description + a paraphrase derived from name.
    # Tests that need richer Contexts can pass extra_views/extra_keywords.
    name_phrase = name.replace("_", " ").replace("-", " ").strip() or description
    queries = [description, name_phrase]
    if extra_views:
        queries.extend(extra_views)
    # Drop dupes while preserving order, cap at 5
    seen = set()
    queries = [q for q in queries if not (q in seen or seen.add(q))][:5]

    # Caller-mandatory keywords: derived from name tokens for the fixture (real
    # callers must provide their own). Pad to ≥2 so validate_context passes.
    base_kws = [t.lower() for t in name_phrase.split() if t]
    if extra_keywords:
        base_kws.extend(extra_keywords)
    base_kws = list(dict.fromkeys(base_kws))
    while len(base_kws) < 2:
        base_kws.append(f"kw{len(base_kws)}")
    keywords = base_kws[:5]

    kwargs = {
        "name": name,
        "context": {"queries": queries, "keywords": keywords, "entities": ["test_agent"]},
        "kind": kind,
        "importance": importance,
        "added_by": added_by,
    }
    if constraints is not None:
        props = properties or {}
        props["constraints"] = constraints
        kwargs["properties"] = props
    elif properties is not None:
        kwargs["properties"] = properties
    return tool_kg_declare_entity(**kwargs)


def _add_edge(subject, predicate, obj, context=None, statement=None):
    """Test helper: build a default Context AND statement from the triple.

    Non-skip predicates require a real statement per TripleStatementRequired
    (2026-04-19). Tests synthesise a declarative sentence from the triple
    parts so they don't have to invent prose \u2014 safe here because the
    enforcement point is the CALLER providing a value, not the value's
    quality (production callers are held to quality by code review).
    """
    from mempalace.mcp_server import tool_kg_add

    if context is None:
        context = {
            "queries": [
                f"{subject} {predicate} {obj}",
                f"edge {predicate} between {subject} and {obj}",
            ],
            "keywords": [subject, predicate, obj][:5] or ["edge", "test"],
            "entities": [subject],
        }
        # Pad keywords to \u22652 if any were empty
        kws = [k for k in context["keywords"] if k]
        while len(kws) < 2:
            kws.append(f"kw{len(kws)}")
        context["keywords"] = kws[:5]
    if statement is None:
        statement = f"{subject} {predicate} {obj} (test edge)."
    return tool_kg_add(
        subject=subject,
        predicate=predicate,
        object=obj,
        context=context,
        agent="test_agent",
        statement=statement,
    )


# ── Entity Declaration ────────────────────────────────────────────────


class TestEntityDeclaration:
    def test_declare_new_entity(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        result = _declare(
            "test-server", "A test server for unit tests", kind="entity", importance=4
        )
        assert result["success"] is True
        assert result["status"] == "created"
        assert result["entity_id"] == "test_server"
        assert result["kind"] == "entity"

    def test_declare_existing_entity_returns_exists(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        _declare("my-tool", "A development tool", kind="entity")
        result = _declare("my-tool", "A development tool", kind="entity")
        assert result["success"] is True
        assert result["status"] == "exists"

    def test_declare_entity_normalizes_name(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        result = _declare("My CamelCase Entity", "Test entity", kind="entity")
        assert result["entity_id"] == "my_camel_case_entity"

    def test_declare_entity_strips_articles(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        result = _declare("the-big-server", "A big server", kind="entity")
        assert result["entity_id"] == "big_server"

    def test_kind_is_required(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        result = _declare("no-kind", "Missing kind", kind=None)
        assert result["success"] is False
        assert "kind" in result["error"].lower() or "REQUIRED" in result["error"]

    def test_invalid_kind_rejected(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        result = _declare("bad-kind", "Bad kind", kind="widget")
        assert result["success"] is False

    def test_declare_registers_in_session(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)
        from mempalace.mcp_server import _STATE

        _declare("session-entity", "Test entity for session", kind="entity")
        assert "session_entity" in _STATE.declared_entities


# ── Predicate Declaration ─────────────────────────────────────────────


class TestPredicateDeclaration:
    def _setup_classes(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        # Need root class 'thing' for constraints
        _declare("thing", "Root class of the ontology", kind="class", importance=5)
        _declare("system", "A running infrastructure component", kind="class", importance=4)
        _declare("tool", "A software tool", kind="class", importance=4)

    def test_predicate_requires_constraints(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        result = _declare("my-predicate", "A test predicate", kind="predicate")
        assert result["success"] is False
        assert "constraints" in result["error"].lower() or "REQUIRE" in result["error"]

    def test_predicate_requires_all_5_fields(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        # Missing object_classes
        result = _declare(
            "incomplete-pred",
            "Missing fields",
            kind="predicate",
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )
        assert result["success"] is False
        assert "object_classes" in result["error"]

    def test_predicate_valid_full_constraints(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        result = _declare(
            "tested-by",
            "Subject is tested by object",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )
        assert result["success"] is True
        assert result["status"] == "created"

    def test_predicate_invalid_cardinality(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        result = _declare(
            "bad-card",
            "Bad cardinality",
            kind="predicate",
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "one-to-none",
            },
        )
        assert result["success"] is False
        assert "cardinality" in result["error"].lower()

    def test_predicate_invalid_kind_in_subject_kinds(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        result = _declare(
            "bad-kinds",
            "Bad subject kinds",
            kind="predicate",
            constraints={
                "subject_kinds": ["widget"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )
        assert result["success"] is False
        assert "widget" in result["error"]

    def test_predicate_nonexistent_class_rejected(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        result = _declare(
            "bad-class-ref",
            "References nonexistent class",
            kind="predicate",
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["nonexistent_class"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )
        assert result["success"] is False
        assert "nonexistent_class" in result["error"]

    def test_predicate_class_must_be_kind_class(self, monkeypatch, config, palace_path, kg):
        self._setup_classes(monkeypatch, config, palace_path, kg)

        # Declare a non-class entity, then try to use it as a class in constraints
        _declare("my-entity", "A regular entity", kind="entity")
        result = _declare(
            "bad-class-kind",
            "Class ref is not kind=class",
            kind="predicate",
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["my-entity"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )
        assert result["success"] is False
        assert "kind=" in result["error"]


# ── kg_add enforcement ────────────────────────────────────────────────


class TestKgAddEnforcement:
    def _setup(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        # Classes
        _declare("thing", "Root class", kind="class", importance=5)
        _declare("system", "Infrastructure component", kind="class", importance=4)
        _declare("rule", "Standing order", kind="class", importance=4)

        # Predicate
        _declare(
            "depends-on",
            "Subject depends on object",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )

        # Entities
        _declare("server-a", "Server A for testing", kind="entity")
        _declare("server-b", "Server B for testing", kind="entity")

    def test_undeclared_subject_rejected(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("unknown-entity", "depends-on", "server-b")
        assert result["success"] is False
        assert "not declared" in result["issues"][0]

    def test_undeclared_predicate_rejected(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("server-a", "unknown-pred", "server-b")
        assert result["success"] is False
        assert "not declared" in result["issues"][0]

    def test_undeclared_object_rejected(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("server-a", "depends-on", "unknown-entity")
        assert result["success"] is False
        assert "not declared" in result["issues"][0]

    def test_predicate_used_as_subject_rejected(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("depends-on", "depends-on", "server-b")
        assert result["success"] is False
        assert "predicate" in result["issues"][0].lower()

    def test_non_predicate_used_as_predicate_rejected(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("server-a", "server-b", "server-a")
        assert result["success"] is False
        assert "predicate" in result["issues"][0].lower()

    def test_valid_edge_succeeds(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("server-a", "depends-on", "server-b")
        assert result["success"] is True


# ── Class constraint enforcement ──────────────────────────────────────


class TestClassConstraints:
    def _setup(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        # Classes
        _declare("thing", "Root class", kind="class", importance=5)
        _declare("system", "Infrastructure component", kind="class", importance=4)
        _declare("process", "A workflow or procedure", kind="class", importance=4)
        _declare("rule", "A standing order", kind="class", importance=4)

        # Predicate restricted to system/process subjects
        _declare(
            "runs-in",
            "Subject runs inside object",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["system", "process"],
                "object_classes": ["system"],
                "cardinality": "many-to-one",
            },
        )

        # Entities with is-a classification
        _declare("my-server", "A test server", kind="entity")
        kg.add_triple("my_server", "is_a", "system")

        _declare("my-rule", "A test rule", kind="entity")
        kg.add_triple("my_rule", "is_a", "rule")

        _declare("docker-env", "Docker runtime", kind="entity")
        kg.add_triple("docker_env", "is_a", "system")

    def test_class_constraint_pass(self, monkeypatch, config, palace_path, kg):
        """system entity as subject of runs-in should pass (system in allowed)."""
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("my-server", "runs-in", "docker-env")
        assert result["success"] is True

    def test_class_constraint_fail_wrong_subject(self, monkeypatch, config, palace_path, kg):
        """rule entity as subject of runs-in should fail (rule not in [system, process])."""
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("my-rule", "runs-in", "docker-env")
        assert result["success"] is False
        assert "constraint_issues" in result
        assert (
            "class mismatch" in result["constraint_issues"][0].lower()
            or "Subject class" in result["constraint_issues"][0]
        )


# ── Class inheritance ─────────────────────────────────────────────────


class TestClassInheritance:
    def _setup(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        # Class hierarchy: thing -> system -> database
        _declare("thing", "Root class", kind="class", importance=5)
        _declare("system", "Infrastructure component", kind="class", importance=4)
        _declare("database", "A database system", kind="class", importance=3)
        # system is-a thing is auto-added; database is-a thing is auto-added
        # But we need database is-a system explicitly
        kg.add_triple("database", "is_a", "system")

        # Predicate allowing only ["thing"] — should accept anything via inheritance
        _declare(
            "has-property",
            "Subject has property object",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity", "literal"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )

        # Predicate allowing only ["system"] — should accept database via inheritance
        _declare(
            "hosted-by",
            "Subject is hosted by object",
            kind="predicate",
            importance=3,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["system"],
                "object_classes": ["system"],
                "cardinality": "many-to-many",
            },
        )

        # Entity classified as database
        _declare("postgres-db", "PostgreSQL database instance", kind="entity")
        kg.add_triple("postgres_db", "is_a", "database")

        # Entity classified as system
        _declare("docker-host", "Docker container host", kind="entity")
        kg.add_triple("docker_host", "is_a", "system")

        # A literal for property value
        _declare("port-5432", "PostgreSQL default port", kind="literal")

    def test_thing_accepts_any_class(self, monkeypatch, config, palace_path, kg):
        """Predicate with subject_classes=['thing'] should accept any classified entity."""
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("postgres-db", "has-property", "port-5432")
        assert result["success"] is True

    def test_inherited_class_passes(self, monkeypatch, config, palace_path, kg):
        """database is-a system, so database entity should pass constraint requiring system."""
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("postgres-db", "hosted-by", "docker-host")
        assert result["success"] is True


# ── Cardinality enforcement ───────────────────────────────────────────


class TestCardinalityEnforcement:
    def _setup(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        # Classes
        _declare("thing", "Root class", kind="class", importance=5)
        _declare("system", "Infrastructure component", kind="class", importance=4)

        # many-to-one predicate
        _declare(
            "runs-in",
            "Subject runs inside object",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-one",
            },
        )

        # many-to-many predicate
        _declare(
            "depends-on",
            "Subject depends on object",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )

        _declare("app", "Test application", kind="entity")
        _declare("container-a", "Docker container A", kind="entity")
        _declare("container-b", "Docker container B", kind="entity")
        _declare("lib-x", "Library X", kind="entity")
        _declare("lib-y", "Library Y", kind="entity")

    def test_many_to_one_first_edge_succeeds(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result = _add_edge("app", "runs-in", "container-a")
        assert result["success"] is True

    def test_many_to_one_second_edge_blocked(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        _add_edge("app", "runs-in", "container-a")
        result = _add_edge("app", "runs-in", "container-b")
        assert result["success"] is False
        assert "constraint_issues" in result
        assert "cardinality" in result["constraint_issues"][0].lower()

    def test_many_to_many_multiple_edges_ok(self, monkeypatch, config, palace_path, kg):
        self._setup(monkeypatch, config, palace_path, kg)

        result1 = _add_edge("app", "depends-on", "lib-x")
        result2 = _add_edge("app", "depends-on", "lib-y")
        assert result1["success"] is True
        assert result2["success"] is True


# ── Entity merge ──────────────────────────────────────────────────────


class TestEntityMerge:
    def test_merge_moves_edges(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        # Create two entities with edges
        _declare("thing", "Root class", kind="class", importance=5)
        _declare("old-server", "The old name for the server", kind="entity")
        _declare("new-server", "The canonical server entity", kind="entity")
        _declare(
            "depends-on",
            "Dependency",
            kind="predicate",
            importance=4,
            constraints={
                "subject_kinds": ["entity"],
                "object_kinds": ["entity"],
                "subject_classes": ["thing"],
                "object_classes": ["thing"],
                "cardinality": "many-to-many",
            },
        )
        _declare("my-db", "A database", kind="entity")

        # Add edge to old entity
        _add_edge("old-server", "depends-on", "my-db")

        # Merge old into new
        from mempalace.mcp_server import tool_kg_merge_entities

        result = tool_kg_merge_entities(
            source="old-server", target="new-server", agent="test_agent"
        )
        assert result["success"] is True
        assert result["edges_moved"] >= 1

    def test_merge_updates_declared_set(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)
        from mempalace.mcp_server import _STATE

        _declare("source-ent", "Source entity", kind="entity")
        _declare("target-ent", "Target entity", kind="entity")

        assert "source_ent" in _STATE.declared_entities
        assert "target_ent" in _STATE.declared_entities

        from mempalace.mcp_server import tool_kg_merge_entities

        tool_kg_merge_entities(source="source-ent", target="target-ent", agent="test_agent")

        assert "source_ent" not in _STATE.declared_entities
        assert "target_ent" in _STATE.declared_entities


# ── Auto is-a thing for classes ───────────────────────────────────────


class TestAutoIsAThing:
    def test_new_class_gets_is_a_thing(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        _declare("thing", "Root class", kind="class", importance=5)
        _declare("vehicle", "A vehicle class", kind="class", importance=3)

        # Check is-a thing edge was auto-added
        edges = kg.query_entity("vehicle", direction="outgoing")
        is_a_thing = [e for e in edges if e["predicate"] == "is_a" and e["object"] == "thing"]
        assert len(is_a_thing) == 1

    def test_thing_itself_no_self_loop(self, monkeypatch, config, palace_path, kg):
        _patch_mcp(monkeypatch, config, kg, palace_path)

        _declare("thing", "Root class", kind="class", importance=5)

        # thing should NOT have is-a thing (self-loop)
        edges = kg.query_entity("thing", direction="outgoing")
        is_a_self = [e for e in edges if e["predicate"] == "is_a" and e["object"] == "thing"]
        assert len(is_a_self) == 0


# ── Normalization ─────────────────────────────────────────────────────


class TestNormalization:
    def test_underscores_used(self):
        from mempalace.knowledge_graph import normalize_entity_name

        assert normalize_entity_name("my-entity") == "my_entity"
        assert normalize_entity_name("my_entity") == "my_entity"
        assert normalize_entity_name("my entity") == "my_entity"

    def test_camel_case_split(self):
        from mempalace.knowledge_graph import normalize_entity_name

        assert normalize_entity_name("CamelCase") == "camel_case"
        assert normalize_entity_name("XMLParser") == "xml_parser"

    def test_article_stripping(self):
        from mempalace.knowledge_graph import normalize_entity_name

        assert normalize_entity_name("the-big-server") == "big_server"
        assert normalize_entity_name("a-small-tool") == "small_tool"
        assert normalize_entity_name("an-example") == "example"

    def test_collapses_all_separators(self):
        from mempalace.knowledge_graph import normalize_entity_name

        assert normalize_entity_name("foo---bar") == "foo_bar"
        assert normalize_entity_name("foo___bar") == "foo_bar"
        assert normalize_entity_name("foo...bar") == "foo_bar"

    def test_empty_normalizes_to_unknown(self):
        from mempalace.knowledge_graph import normalize_entity_name

        assert normalize_entity_name("---") == "unknown"
        assert normalize_entity_name("") == "unknown"


# ── Intent Declaration ────────────────────────────────────────────────


def _setup_intent_hierarchy(monkeypatch, config, palace_path, kg):
    """Set up a minimal intent type hierarchy for testing."""
    import chromadb
    from pathlib import Path
    from mempalace import mcp_server

    _patch_mcp(monkeypatch, config, kg, palace_path)

    # Point intent state dir to temp
    state_dir = Path(palace_path) / "hook_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mcp_server, "_INTENT_STATE_DIR", state_dir)

    # Classes
    _declare("thing", "Root class", kind="class", importance=5)
    _declare("file", "A file", kind="class", importance=3)
    _declare("system", "Infrastructure", kind="class", importance=4)
    _declare("project", "A project", kind="class", importance=4)

    # intent_type class
    _declare("intent-type", "Class for intent types", kind="class", importance=5)

    # is-a predicate
    _declare(
        "is_a",
        "Taxonomic classification",
        kind="predicate",
        importance=5,
        constraints={
            "subject_kinds": ["entity", "class"],
            "object_kinds": ["class"],
            "subject_classes": ["thing"],
            "object_classes": ["thing"],
            "cardinality": "many-to-many",
        },
    )

    # Sync intent types to ChromaDB for _is_declared fallback
    client = chromadb.PersistentClient(path=palace_path)
    ecol = client.get_or_create_collection("mempalace_entities")

    # Top-level intent type: modify
    props_modify = {
        "rules_profile": {
            "slots": {"files": {"classes": ["file"], "required": True, "multiple": True}},
            "tool_permissions": [
                {"tool": "Edit", "scope": "{files}"},
                {"tool": "Write", "scope": "{files}"},
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
            ],
        }
    }
    kg.add_entity(
        "modify",
        kind="class",
        description="Intent: modify files",
        importance=4,
        properties=props_modify,
    )
    from mempalace.mcp_server import _STATE

    _STATE.declared_entities.add("modify")
    kg.add_triple("modify", "is_a", "intent_type")
    ecol.upsert(
        ids=["modify"],
        documents=["Intent: modify files"],
        metadatas=[{"name": "modify", "kind": "class", "importance": 4}],
    )

    # Child intent type: edit_file (inherits from modify, no own permissions)
    props_edit = {
        "rules_profile": {
            "slots": {"files": {"classes": ["file"], "required": True, "multiple": True}},
        }
    }
    kg.add_entity(
        "edit_file",
        kind="class",
        description="Intent: edit files",
        importance=4,
        properties=props_edit,
    )
    _STATE.declared_entities.add("edit_file")
    kg.add_triple("edit_file", "is_a", "modify")
    ecol.upsert(
        ids=["edit_file"],
        documents=["Intent: edit files"],
        metadatas=[{"name": "edit_file", "kind": "class", "importance": 4}],
    )

    # Top-level: inspect (own permissions, different slots)
    props_inspect = {
        "rules_profile": {
            "slots": {"subject": {"classes": ["thing"], "required": True, "multiple": True}},
            "tool_permissions": [
                {"tool": "Read", "scope": "*"},
                {"tool": "Grep", "scope": "*"},
            ],
        }
    }
    kg.add_entity(
        "inspect",
        kind="class",
        description="Intent: read-only observation",
        importance=4,
        properties=props_inspect,
    )
    _STATE.declared_entities.add("inspect")
    kg.add_triple("inspect", "is_a", "intent_type")
    ecol.upsert(
        ids=["inspect"],
        documents=["Intent: read-only observation"],
        metadatas=[{"name": "inspect", "kind": "class", "importance": 4}],
    )

    del client

    # Sample target entities — file entities must have file_path in properties
    _declare(
        "auth-test-ts",
        "tests/auth.test.ts — The auth test file",
        kind="entity",
        properties={"file_path": "tests/auth.test.ts"},
    )
    kg.add_triple("auth_test_ts", "is_a", "file")

    _declare(
        "main-ts",
        "src/main.ts — The main file",
        kind="entity",
        properties={"file_path": "src/main.ts"},
    )
    kg.add_triple("main_ts", "is_a", "file")

    _declare("my-server", "A server", kind="entity")
    kg.add_triple("my_server", "is_a", "system")


class TestDeclareIntent:
    def test_declare_valid_intent(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        from mempalace.mcp_server import tool_active_intent

        result = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["Adding tests", "test perspective"],
                "keywords": ["test", "declare"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is True
        # declare_intent responds with only derived data (intent_id, permissions,
        # memories) — echoed inputs live on active_intent when actually needed.
        assert result["intent_id"].startswith("intent_edit_file_")
        assert len(result["permissions"]) > 0
        active = tool_active_intent()
        assert active["intent_type"] == "edit_file"
        assert "auth_test_ts" in active["slots"]["files"]

    def test_undeclared_intent_type_rejected(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        result = tool_declare_intent(
            intent_type="nonexistent_intent",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["test", "second perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is False
        assert "not declared" in result["error"]

    def test_non_intent_type_rejected(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        # auth-test-ts is an entity but NOT an intent type
        result = tool_declare_intent(
            intent_type="auth-test-ts",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is False
        assert "not an intent type" in result["error"]

    def test_required_slot_missing(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        result = tool_declare_intent(
            intent_type="edit_file",
            slots={},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },  # files is required
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is False
        assert "slot_issues" in result
        assert any("Required slot" in e for e in result["slot_issues"])

    def test_unknown_slot_rejected(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        result = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["auth-test-ts"], "bogus": ["something"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is False
        assert any("Unknown slot" in e for e in result["slot_issues"])

    def test_slot_class_mismatch(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        # my-server is-a system, but edit_file.files expects class=file
        result = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["my-server"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["my_server"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is False
        assert any("class" in e.lower() for e in result["slot_issues"])

    def test_inherits_permissions_from_parent(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        # edit_file has no own tool_permissions, should inherit from modify
        result = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is True
        # Permissions are now strings like "Edit(path)" instead of dicts
        perms = result["permissions"]
        assert any("Edit" in p for p in perms)
        assert any("Read" in p for p in perms)

    def test_permissions_scoped_to_slot(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent

        result = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is True
        # Permissions are now strings like "Edit(tests/auth.test.ts)" or "Read(*)"
        perms = result["permissions"]
        edit_perms = [p for p in perms if "Edit" in p]
        read_perms = [p for p in perms if "Read" in p]
        assert len(edit_perms) > 0
        assert "(*)" not in edit_perms[0]  # scoped, not wildcard
        assert "(*)" in read_perms[0]  # unrestricted

    def test_new_intent_requires_finalize_first(self, monkeypatch, config, palace_path, kg):
        """Declaring a new intent without finalizing the active one fails (hard fail)."""
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent, tool_finalize_intent

        result1 = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result1["success"] is True

        # Without finalize — should fail
        result2 = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["main-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["main_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result2["success"] is False

        # After finalize — should succeed. Map-shape memory_feedback
        # attributes each entry to the active intent context.
        import mempalace.mcp_server as _mcp

        _injected = (
            _mcp._STATE.active_intent.get("injected_memory_ids", set())
            if _mcp._STATE.active_intent
            else set()
        )
        _ctx_id = (
            _mcp._STATE.active_intent.get("active_context_id", "")
            if _mcp._STATE.active_intent
            else ""
        )
        _fb = [
            {
                "context_id": _ctx_id,
                "feedback": [
                    {
                        "id": mid,
                        "relevant": False,
                        "relevance": 1,
                        "reason": "Not relevant to this test action",
                    }
                    for mid in _injected
                    if mid
                ],
            }
        ]
        tool_finalize_intent(
            slug="test-expire-prev",
            outcome="success",
            content="Done",
            summary="Done",
            agent="test_agent",
            memory_feedback=_fb,
        )
        result3 = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["main-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["main_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result3["success"] is True

    def test_single_string_slot_value(self, monkeypatch, config, palace_path, kg):
        """Slot values can be a single string (auto-wrapped to list)."""
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_active_intent, tool_declare_intent

        result = tool_declare_intent(
            intent_type="edit_file",
            slots={"files": "auth-test-ts"},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },  # string, not list
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        assert result["success"] is True
        # Normalized slot values live on active_intent, not on the declare response.
        assert "auth_test_ts" in tool_active_intent()["slots"]["files"]


class TestActiveIntent:
    def test_no_active_intent(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_active_intent
        import mempalace.mcp_server as ms

        ms._STATE.active_intent = None

        result = tool_active_intent()
        assert result["active"] is False

    def test_active_intent_after_declare(self, monkeypatch, config, palace_path, kg):
        _setup_intent_hierarchy(monkeypatch, config, palace_path, kg)
        from mempalace.mcp_server import tool_declare_intent, tool_active_intent

        tool_declare_intent(
            intent_type="edit_file",
            slots={"files": ["auth-test-ts"]},
            context={
                "queries": ["test action", "test perspective"],
                "keywords": ["test", "intent"],
                "entities": ["auth_test_ts"],
            },
            agent="test_agent",
            budget=_TEST_BUDGET,
        )
        result = tool_active_intent()
        assert result["active"] is True
        assert result["intent_type"] == "edit_file"


class TestSeedOntology:
    def test_seed_creates_classes_and_predicates(self, tmp_dir):
        """seed_ontology on empty DB creates the canonical ontology."""
        import os
        from mempalace.knowledge_graph import KnowledgeGraph

        db_path = os.path.join(tmp_dir, "seed_test.sqlite3")
        kg = KnowledgeGraph(db_path=db_path)
        kg.seed_ontology()

        # Check classes exist
        thing = kg.get_entity("thing")
        assert thing is not None
        assert thing["kind"] == "class"

        # Check predicates exist
        is_a = kg.get_entity("is_a")
        assert is_a is not None
        assert is_a["kind"] == "predicate"

        # Check intent types exist
        modify = kg.get_entity("modify")
        assert modify is not None

        # Check is-a edges
        edges = kg.query_entity("modify", direction="outgoing")
        is_a_intent = [e for e in edges if e["predicate"] == "is_a" and "intent" in e["object"]]
        assert len(is_a_intent) == 1

        kg.close()

    def test_seed_is_idempotent(self, tmp_dir):
        """Calling seed_ontology twice doesn't duplicate entities."""
        import os
        from mempalace.knowledge_graph import KnowledgeGraph

        db_path = os.path.join(tmp_dir, "seed_idem.sqlite3")
        kg = KnowledgeGraph(db_path=db_path)
        kg.seed_ontology()
        stats1 = kg.stats()
        kg.seed_ontology()
        stats2 = kg.stats()

        assert stats1["entities"] == stats2["entities"]
        kg.close()
