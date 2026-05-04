"""
conftest.py -- Shared fixtures for MemPalace tests.

Provides isolated palace and knowledge graph instances so tests never
touch the user's real data or leak temp files on failure.

HOME is redirected to a temp directory at module load time -- before any
mempalace imports -- so that module-level initialisations (e.g.
``_kg = KnowledgeGraph()`` in mcp_server) write to a throwaway location
instead of the real user profile.
"""

import os
import shutil
import tempfile
from pathlib import Path

# ── Isolate HOME before any mempalace imports ──────────────────────────
_original_env = {}
_session_tmp = tempfile.mkdtemp(prefix="mempalace_session_")

for _var in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH"):
    _original_env[_var] = os.environ.get(_var)

os.environ["HOME"] = _session_tmp
os.environ["USERPROFILE"] = _session_tmp
os.environ["HOMEDRIVE"] = os.path.splitdrive(_session_tmp)[0] or "C:"
os.environ["HOMEPATH"] = os.path.splitdrive(_session_tmp)[1] or _session_tmp
os.environ["MEMPALACE_SKIP_SEED"] = "1"  # Tests use empty KGs by design

# Now it is safe to import mempalace modules that trigger initialisation.
import chromadb  # noqa: E402
import pytest  # noqa: E402

from mempalace.config import MempalaceConfig  # noqa: E402
from mempalace.knowledge_graph import KnowledgeGraph  # noqa: E402


# ── Test pyramid: auto-classify integration vs unit ──────────────────────────
#
# Fast feedback loop: pre-commit runs only `-m unit` tests (pure functions, no
# ChromaDB, no mcp_server module globals, no real KnowledgeGraph) in seconds.
# Full suite including integration runs in CI. Classification lives here so
# there's one source of truth instead of a pytestmark line at the top of every
# test file.
#
# Detection is automatic and self-correcting:
#   1. Any test whose fixture closure pulls in a heavy fixture (kg, collection,
#      config, ...) is integration -- the fixture is what does the slow work.
#   2. Any test module that imports a heavy mempalace module (mcp_server,
#      knowledge_graph) or chromadb directly is integration -- those tests
#      exercise real backend state without going through a fixture.
#   3. Everything else is unit.
#
# Adding a new test that uses one of the heavy fixtures or imports any heavy
# module is automatically routed to the integration lane -- no manual list to
# update. If a NEW heavy fixture or a NEW heavy module needs to count, add it
# to _HEAVY_FIXTURES / _HEAVY_MODULES below.
_HEAVY_FIXTURES = frozenset(
    {
        "kg",
        "seeded_kg",
        "collection",
        "seeded_collection",
        "config",
    }
)

_HEAVY_MODULES = frozenset(
    {
        # Real ChromaDB client -- slow first-time import + per-test work.
        "chromadb",
        # mcp_server carries module-level ChromaDB caches and active-intent
        # state; importing it (or anything from it) means the test exercises
        # that state machine.
        "mempalace.mcp_server",
        # KnowledgeGraph instantiation hits SQLite + yoyo migrations.
        "mempalace.knowledge_graph",
        # Injection gate runs the Haiku tool-use shape (with mocks in tests but
        # still pulls heavy import chains).
        "mempalace.injection_gate",
        # Entity gate similarly.
        "mempalace.entity_gate",
    }
)


def _module_uses_heavy_imports(test_module):
    """True if the test module imports a mempalace module that exercises real
    backend state, or imports chromadb directly.

    Two-pass detection:
      1. Module globals scan (catches top-level `import X`, `from X import Y`).
         Cheap and exact -- looks at already-resolved objects.
      2. Source substring scan (catches lazy imports inside functions, like
         `from mempalace.knowledge_graph import KnowledgeGraph` written inside
         a helper). Essential because tests that seed a full ontology often
         hide their heavy imports behind a `_bootstrap_kg` function and would
         otherwise sneak into the unit lane.

    Catches all common patterns:
      from mempalace.mcp_server import _STATE   (value's __module__ matches)
      from mempalace import mcp_server          (value is the module itself)
      import chromadb                           (value is the module itself)
      def _foo(): from mempalace.knowledge_graph import KnowledgeGraph  (lazy)
    """
    import types

    for value in vars(test_module).values():
        if isinstance(value, types.ModuleType):
            name = value.__name__
            if name in _HEAVY_MODULES:
                return True
            # Submodules of a heavy package count too.
            for heavy in _HEAVY_MODULES:
                if name.startswith(heavy + "."):
                    return True
        else:
            origin = getattr(value, "__module__", None)
            if origin in _HEAVY_MODULES:
                return True

    # Source-file fallback: catch lazy imports inside functions / helpers.
    # Substring match is sufficient -- the heavy module names are
    # discriminative (no chance of accidental match in a comment).
    src_path = getattr(test_module, "__file__", None)
    if src_path:
        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
            for heavy in _HEAVY_MODULES:
                if heavy in src:
                    return True
        except OSError:
            pass

    return False


def pytest_collection_modifyitems(config, items):
    """Attach unit/integration markers via fixture closure + module import scan.

    Tests under tests/benchmarks/ are left untouched -- they carry their own
    markers (benchmark, slow, stress) and should never be swept into the
    unit/integration lanes just because they live in the tree.
    """
    # Cache per-module heavy-import answer to avoid scanning globals
    # once per test.
    module_heavy = {}

    for item in items:
        module_path = Path(item.module.__file__)
        if "benchmarks" in module_path.parts:
            continue

        is_heavy = False

        # 1. fixture closure -- self-correcting, catches new tests
        # that use the existing fixtures.
        if any(name in _HEAVY_FIXTURES for name in item.fixturenames):
            is_heavy = True
        else:
            # 2. heavy-module imports at the test-module level -- catches
            # tests that build their own KG / Chroma client without the
            # fixture, or call mcp_server functions directly.
            mod_id = id(item.module)
            heavy = module_heavy.get(mod_id)
            if heavy is None:
                heavy = _module_uses_heavy_imports(item.module)
                module_heavy[mod_id] = heavy
            is_heavy = heavy

        item.add_marker(pytest.mark.integration if is_heavy else pytest.mark.unit)


@pytest.fixture(autouse=True)
def _reset_mcp_cache():
    """Reset MCP server module state between tests.

    mcp_server carries several module-level globals (ChromaDB caches plus
    the active intent / pending conflicts). Without
    resetting them between tests, leaks cause false positives -- and under
    pytest-xdist workers they cause race conditions, since each worker is
    a separate Python process but individual tests inside a worker still
    share the module.
    """

    def _clear_cache():
        try:
            from mempalace import mcp_server

            mcp_server._STATE.reset_transient()
            # ChromaDB caches live on _STATE.client_cache / collection_cache.
            # reset_transient() deliberately preserves them in production (rebuild
            # is expensive), but tests need a clean slate so each palace fixture
            # sees its own collection, not the prior test's.
            mcp_server._STATE.client_cache = None
            mcp_server._STATE.collection_cache = None
        except (ImportError, AttributeError):
            pass

    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture(scope="session", autouse=True)
def _isolate_home():
    """Ensure HOME points to a temp dir for the entire test session.

    The env vars were already set at module level (above) so that
    module-level initialisations are captured.  This fixture simply
    restores the originals on teardown and cleans up the temp dir.
    """
    yield
    for var, orig in _original_env.items():
        if orig is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = orig
    shutil.rmtree(_session_tmp, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _prewarm_chroma_embedding_model():
    """Force Chroma's default embedding model (ONNX all-MiniLM-L6-v2) to
    download + load ONCE per test session.

    Without this, the 79MB ONNX model ends up loading inside the first test
    that actually embeds text, inflating its duration by 2-3s. Per-test
    fixtures create new PersistentClient instances, but Chroma caches the
    embedding function at module level in the same Python process, so a
    single warm-up in an isolated temp dir is enough.
    """
    warm_dir = tempfile.mkdtemp(prefix="mempalace_warmup_")
    try:
        client = chromadb.PersistentClient(path=warm_dir)
        col = client.get_or_create_collection("prewarm", metadata={"hnsw:space": "cosine"})
        col.add(ids=["warmup"], documents=["warmup"])
        del client
    except Exception:
        pass
    finally:
        shutil.rmtree(warm_dir, ignore_errors=True)
    yield


@pytest.fixture
def tmp_dir():
    """Create and auto-cleanup a temporary directory."""
    d = tempfile.mkdtemp(prefix="mempalace_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def palace_path(tmp_dir):
    """Path to an empty palace directory inside tmp_dir."""
    p = os.path.join(tmp_dir, "palace")
    os.makedirs(p)
    return p


@pytest.fixture
def config(tmp_dir, palace_path):
    """A MempalaceConfig pointing at the temp palace."""
    cfg_dir = os.path.join(tmp_dir, "config")
    os.makedirs(cfg_dir)
    import json

    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump({"palace_path": palace_path}, f)
    return MempalaceConfig(config_dir=cfg_dir)


@pytest.fixture
def collection(palace_path):
    """A ChromaDB collection pre-seeded in the temp palace."""
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_records")
    yield col
    client.delete_collection("mempalace_records")
    del client


@pytest.fixture
def seeded_collection(collection):
    """Collection with a handful of representative memories."""
    collection.add(
        ids=[
            "record_proj_backend_aaa",
            "record_proj_backend_bbb",
            "record_proj_frontend_ccc",
            "record_notes_planning_ddd",
        ],
        documents=[
            "The authentication module uses JWT tokens for session management. "
            "Tokens expire after 24 hours. Refresh tokens are stored in HttpOnly cookies.",
            "Database migrations are handled by Alembic. We use PostgreSQL 15 "
            "with connection pooling via pgbouncer.",
            "The React frontend uses TanStack Query for server state management. "
            "All API calls go through a centralized fetch wrapper.",
            "Sprint planning: migrate auth to passkeys by Q3. "
            "Evaluate ChromaDB alternatives for vector search.",
        ],
        metadatas=[
            {
                "source_file": "auth.py",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-01T00:00:00",
                "content_type": "fact",
            },
            {
                "source_file": "db.py",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-02T00:00:00",
                "content_type": "fact",
            },
            {
                "source_file": "App.tsx",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-03T00:00:00",
                "content_type": "fact",
            },
            {
                "source_file": "sprint.md",
                "chunk_index": 0,
                "added_by": "miner",
                "filed_at": "2026-01-04T00:00:00",
                "content_type": "event",
            },
        ],
    )
    return collection


@pytest.fixture
def kg(tmp_dir):
    """An isolated KnowledgeGraph using a temp SQLite file."""
    db_path = os.path.join(tmp_dir, "test_kg.sqlite3")
    return KnowledgeGraph(db_path=db_path)


@pytest.fixture
def seeded_kg(kg):
    """KnowledgeGraph pre-loaded with sample triples."""
    # every write tool requires a declared agent, so test fixtures
    # seed a `test_agent` (is_a agent) here. Tests pass agent='test_agent'
    # to all write-tool calls.
    kg.add_entity("test_agent", kind="entity", content="Test agent for unit tests")
    kg.add_entity("agent", kind="class", content="Agent class for is_a")
    kg.add_triple("test_agent", "is_a", "agent")

    kg.add_entity("Alice", kind="entity", content="A person named Alice")
    kg.add_entity("Max", kind="entity", content="A person named Max")
    kg.add_entity("swimming", kind="entity", content="The sport of swimming")
    kg.add_entity("chess", kind="entity", content="The board game chess")
    # Cold-start lock 2026-05-01: add_triple no longer phantom-creates
    # missing endpoints, so every entity referenced in a triple below
    # MUST be declared upfront. Pre-declare the work-history endpoints.
    kg.add_entity("Acme Corp", kind="entity", content="A company called Acme Corp")
    kg.add_entity("NewCo", kind="entity", content="A company called NewCo")

    # Non-skip predicates require caller-provided statements post-2026-04-19
    # (see TripleStatementRequired in knowledge_graph.py). Seed with short
    # natural-language sentences so retrieval tests behave realistically.
    kg.add_triple(
        "Alice",
        "parent_of",
        "Max",
        valid_from="2015-04-01",
        statement="Alice is the parent of Max.",
    )
    kg.add_triple(
        "Max",
        "does",
        "swimming",
        valid_from="2025-01-01",
        statement="Max swims (sport of swimming since 2025).",
    )
    kg.add_triple(
        "Max",
        "does",
        "chess",
        valid_from="2024-06-01",
        statement="Max plays chess (started mid-2024).",
    )
    kg.add_triple(
        "Alice",
        "works_at",
        "Acme Corp",
        valid_from="2020-01-01",
        valid_to="2024-12-31",
        statement="Alice worked at Acme Corp from 2020 until end-2024.",
    )
    kg.add_triple(
        "Alice",
        "works_at",
        "NewCo",
        valid_from="2025-01-01",
        statement="Alice started at NewCo in January 2025.",
    )

    return kg
