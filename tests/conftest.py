"""
conftest.py — Shared fixtures for MemPalace tests.

Provides isolated palace and knowledge graph instances so tests never
touch the user's real data or leak temp files on failure.

HOME is redirected to a temp directory at module load time — before any
mempalace imports — so that module-level initialisations (e.g.
``_kg = KnowledgeGraph()`` in mcp_server) write to a throwaway location
instead of the real user profile.
"""

import os
import shutil
import tempfile

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


@pytest.fixture(autouse=True)
def _reset_mcp_cache():
    """Reset the MCP server's cached ChromaDB client/collection between tests."""

    def _clear_cache():
        try:
            from mempalace import mcp_server

            mcp_server._client_cache = None
            mcp_server._collection_cache = None
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
    kg.add_entity("test_agent", kind="entity", description="Test agent for unit tests")
    kg.add_entity("agent", kind="class", description="Agent class for is_a")
    kg.add_triple("test_agent", "is_a", "agent")

    kg.add_entity("Alice", kind="entity", description="A person named Alice")
    kg.add_entity("Max", kind="entity", description="A person named Max")
    kg.add_entity("swimming", kind="entity", description="The sport of swimming")
    kg.add_entity("chess", kind="entity", description="The board game chess")

    kg.add_triple("Alice", "parent_of", "Max", valid_from="2015-04-01")
    kg.add_triple("Max", "does", "swimming", valid_from="2025-01-01")
    kg.add_triple("Max", "does", "chess", valid_from="2024-06-01")
    kg.add_triple("Alice", "works_at", "Acme Corp", valid_from="2020-01-01", valid_to="2024-12-31")
    kg.add_triple("Alice", "works_at", "NewCo", valid_from="2025-01-01")

    return kg
