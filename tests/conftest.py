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
# Tests that explicitly call ``kg.seed_ontology()`` (e.g. test_task_kind's
# ``_bootstrap_kg`` helper) need the canonical SQLite ontology to be
# present, but the per-entity Chroma sync inside ``seed_ontology`` runs
# ONNX embeddings for ~50 entities and dominates 30-78s of test runtime
# on cold caches. Tests almost never read mcp_server-owned Chroma for
# seeded entities (they assert against SQLite tables or use per-test
# ``collection`` fixtures with their own palace path). Opting out here
# turns a 60s seed into a sub-second SQL-only insert. Tests that
# genuinely need Chroma rows for seeded entities can call
# ``kg.backfill_seed_chroma()`` explicitly after the seed.
os.environ["MEMPALACE_SKIP_SEED_CHROMA_SYNC"] = "1"

# Now it is safe to import mempalace modules that trigger initialisation.
import pytest  # noqa: E402

from mempalace.config import MempalaceConfig  # noqa: E402
from mempalace.knowledge_graph import KnowledgeGraph  # noqa: E402


# ── Test pyramid: directory split + drift warning ──────────────────────────
#
# Layout is authoritative now (Adrian directive 2026-05-06 -- replaced the
# auto-classifier magic with explicit dir + markers, the canonical Brian
# Okken / pytest-docs convention):
#
#   tests/unit/          -- @pytest.mark.unit  (pure functions, fast lane)
#   tests/integration/   -- @pytest.mark.integration (real Chroma / KG)
#   tests/benchmarks/    -- benchmark / slow / stress (opt-in)
#
# Each test file carries `pytestmark = pytest.mark.{unit|integration}` at
# module level. CI selects the lane by directory path. Markers are
# redundant with the dir layout but kept so `pytest -m unit` / `-m integration`
# still works for ad-hoc runs.
#
# The block below is a DRIFT DETECTOR, not a classifier:
#
#   - When a test file under tests/unit/ imports a heavy mempalace module
#     (mcp_server / knowledge_graph / injection_gate / entity_gate) or
#     chromadb directly, we emit a pytest warning at collection time.
#     The test still runs in the unit lane (the marker on the file is
#     authoritative); the warning surfaces a likely misclassification so
#     the maintainer can either (a) move the file into tests/integration/
#     or (b) extract the pure helper they imported into a leaf module
#     so the import chain stays light.
#
# Enable PYTEST_TESTS_DRIFT_FAIL=1 to escalate the warning to a hard error
# (useful in CI; default is warn-only so drift doesn't block local runs).
_HEAVY_MODULES = frozenset(
    {
        "chromadb",
        "mempalace.mcp_server",
        "mempalace.knowledge_graph",
        "mempalace.injection_gate",
        "mempalace.entity_gate",
        # v3.7.12 (Adrian directive 2026-05-17): intent + scoring are
        # heavy-by-transitive-pull -- importing either drags in the
        # knowledge_graph + chromadb chain. Adding them to the heavy
        # list closes the second of the two drift gaps Adrian flagged
        # (the first -- form gap -- is closed by the v3.7.12 regex
        # tighten below). Pre-v3.7.12 a unit test importing
        # `mempalace.intent` or `mempalace.scoring` slipped past the
        # detector silently; now it warns at collection time.
        "mempalace.intent",
        "mempalace.scoring",
    }
)


import re  # noqa: E402

# Short names of mempalace submodules that count as heavy when imported
# via the `from mempalace import X` form (form #2 in the audit). Derived
# from _HEAVY_MODULES -- just the bit after the dot for entries that
# start with "mempalace.".
_HEAVY_SHORT_NAMES = frozenset(
    name.split(".", 1)[1] for name in _HEAVY_MODULES if name.startswith("mempalace.")
)


# Drift detector regex -- v3.7.12 (Adrian directive 2026-05-17):
# tightened to catch BOTH forms of heavy-module import that a unit
# test might use:
#
#   Form 1 (qualified path): `import X` or `from X[.sub] import ...`
#     where X is one of the modules in _HEAVY_MODULES. Matches
#     `import chromadb`, `from mempalace.mcp_server import _STATE`,
#     `from chromadb.utils import foo`. This is the original pattern.
#
#   Form 2 (package + name): `from mempalace import HEAVY` where HEAVY
#     is one of the short names in _HEAVY_SHORT_NAMES. Matches
#     `from mempalace import mcp_server` and `from mempalace import
#     intent, scoring`. The original pattern missed this form because
#     `mempalace` alone (without `.X`) wasn't a heavy module; the
#     scanner walked past without realising the heavy name appeared
#     after the second `import` keyword on the same line. This let
#     test_bg_state_judge.py (and similar) sneak into tests/unit/
#     pre-v3.7.11.
#
# Both forms emit the same drift warning. Anchored to import keywords
# so string literals like "chromadb-setup" in fake-payload test data
# don't match.
_HEAVY_IMPORT_RE = re.compile(
    # Form 1: import HEAVY  OR  from HEAVY[.sub] import ...
    r"^(?:import|from)\s+(?:"
    + r"|".join(re.escape(h) for h in _HEAVY_MODULES)
    + r")\b"
    # Form 2: from mempalace import (... HEAVY_SHORT_NAME ...). Match
    # any HEAVY_SHORT_NAME appearing after `from mempalace import`,
    # on the same line, with a word-boundary so partial matches
    # (e.g. `mcp_server_helper`) don't false-positive.
    + r"|^from\s+mempalace\s+import\s+[^\n]*\b(?:"
    + r"|".join(re.escape(n) for n in _HEAVY_SHORT_NAMES)
    + r")\b",
    re.MULTILINE,
)
# v3.7.13 (Adrian directive 2026-05-17): both anchors are `^` with NO
# `\s*` so only column-0 imports match. Function-body imports
# (indented under `def`/`class`/inside helpers) are the DEFERRED-LOAD
# pattern -- the test file itself stays light at collection time and
# pays the heavy load only when the specific test runs. That's the
# behaviour the drift detector should reward, not punish. Pre-v3.7.13
# the `^\s*` anchor matched indented imports too, generating false
# positives on test_bg_status.py + test_split_for_surface.py (both
# use deferred imports correctly). True module-level imports almost
# always sit at column 0 in real code; rare try/except module-level
# wrappers around an import are still drift in spirit (they pull the
# module at collection time) -- if one shows up we'll address it
# then.


def _module_uses_heavy_imports(test_module) -> bool:
    """True if a unit-tree test module imports a heavy module.

    Regex scan over the source file: matches `import X` / `from X[.sub] import`
    lines only. Non-import occurrences (string literals, docstrings, comments)
    do NOT count -- those are not real drift, just incidental matches.
    """
    src_path = getattr(test_module, "__file__", None)
    if not src_path:
        return False
    try:
        with open(src_path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
    except OSError:
        return False
    return bool(_HEAVY_IMPORT_RE.search(src))


def pytest_collection_modifyitems(config, items):
    """Drift warning: flag tests/unit/ files that pull heavy imports.

    Markers are NOT applied here -- each test file carries its own
    module-level `pytestmark`. This hook only warns when layout drifts.
    """
    seen_modules: set = set()
    drift: list = []
    for item in items:
        module_path = Path(item.module.__file__)
        # Only check files actually under tests/unit/ -- benchmarks,
        # integration, and other trees are out of scope.
        if "unit" not in module_path.parts:
            continue
        mod_id = id(item.module)
        if mod_id in seen_modules:
            continue
        seen_modules.add(mod_id)
        if _module_uses_heavy_imports(item.module):
            drift.append(str(module_path.relative_to(module_path.parents[2])))

    if drift:
        # v3.7.12 (Adrian directive 2026-05-17): derive the heavy-module
        # list from _HEAVY_MODULES itself so the warning text can never
        # drift out of sync with the regex. Pre-v3.7.12 the list was
        # hardcoded and silently went stale when intent + scoring were
        # added to the frozenset.
        heavy_list = " / ".join(sorted(_HEAVY_MODULES))
        msg = (
            "Test layout drift detected -- the following tests/unit/ files "
            f"import heavy modules ({heavy_list}) and probably belong in "
            "tests/integration/ or should extract the pure helper they import "
            "into a leaf module:\n  " + "\n  ".join(drift)
        )
        if os.environ.get("PYTEST_TESTS_DRIFT_FAIL") == "1":
            raise pytest.UsageError(msg)
        else:
            import warnings

            warnings.warn(msg, stacklevel=1)


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
def _prewarm_embedding_model():
    """Force fastembed's default model (ONNX all-MiniLM-L6-v2) to
    download + load ONCE per test session so the first embedding
    call in a real test doesn't eat 2-3s warming the model.

    Post-chromadb-removal (2026-05-12) this calls fastembed directly
    via :func:`mempalace.embedder.get_default_embedder`. Best-effort:
    if fastembed isn't available, the suite still runs (tests that
    actually need vectors pass embeddings explicitly)."""
    try:
        from mempalace.embedder import get_default_embedder

        embedder = get_default_embedder()
        if embedder is not None:
            embedder(["warmup"])
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _reset_singletons_around_test(monkeypatch):
    """Drop VectorStore + mcp_server _STATE singletons before AND after
    every test.

    Without the singleton reset, a test that opens a VectorStore at one
    palace_path leaves the cached store + the mcp_server
    _STATE.client_cache live for the next test (which typically uses a
    different palace_path).

    Post-chromadb-removal (2026-05-12): the old version of this
    fixture also monkeypatched ``chromadb.PersistentClient`` to inject
    Settings(anonymized_telemetry=False). That branch is gone now --
    chromadb is no longer a runtime dep, so there is no global client
    constructor to wrap.
    """
    from mempalace.vector_store import reset_singletons

    reset_singletons()
    try:
        from mempalace import mcp_server as _mcp

        if hasattr(_mcp, "_STATE"):
            _mcp._STATE.client_cache = None
            _mcp._STATE.collection_cache = None
    except Exception:
        pass

    yield
    reset_singletons()
    try:
        from mempalace import mcp_server as _mcp

        if hasattr(_mcp, "_STATE"):
            _mcp._STATE.client_cache = None
            _mcp._STATE.collection_cache = None
    except Exception:
        pass


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
    """A ChromaDB collection pre-seeded in the temp palace.

    Routes through mempalace.vector_store.get_vector_store so the
    underlying chromadb.PersistentClient uses the same Settings
    (anonymized_telemetry=False) the production VectorStore opens
    with -- otherwise the second open fails with `ValueError: An
    instance of Chroma already exists for ... with different settings`
    (caught 2026-05-09 by the d6c8a71 root-cause-surfacing fix).

    Also passes hnsw:sync_threshold=3 (Chroma rejects 1-2; 3 is the
    minimum that flushes after every batch) so 4-row test seeds become
    immediately visible to vs.query. Production uses 100 (slice 16
    SIGSEGV prevention) but at that threshold small test seeds never
    reach HNSW.
    """
    # This fixture historically yielded a raw chromadb Collection;
    # downstream tests call col.add() / col.query() / col.get() /
    # col.upsert() / col.delete() with chromadb's native API. After
    # chromadb removal (2026-05-12) we yield a thin adapter that
    # proxies the chromadb-Collection surface to the VectorStore --
    # downstream tests keep working unchanged.
    from mempalace.vector_store import (
        RECORDS_COLLECTION,
        get_vector_store,
        reset_singletons,
    )

    reset_singletons()
    vs = get_vector_store(palace_path)
    vs._open(RECORDS_COLLECTION, create=True)

    class _CollectionApiAdapter:
        """Chromadb-Collection-shaped facade over a VectorStore +
        collection name. Returns chromadb-shaped dicts (with ``ids``
        / ``documents`` / ``metadatas`` / ``distances`` keys) so
        existing chromadb-flavoured assertions continue to work."""

        def __init__(self, store, name):
            self._vs = store
            self._name = name

        @property
        def name(self):
            return self._name

        def count(self):
            return int(self._vs.count(self._name))

        def add(self, ids, documents=None, metadatas=None, embeddings=None):
            return self._vs.add(
                self._name,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        def upsert(self, ids, documents=None, metadatas=None, embeddings=None):
            return self._vs.upsert(
                self._name,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        def update(self, ids, documents=None, metadatas=None, embeddings=None):
            return self._vs.update(
                self._name,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        def get(self, ids=None, where=None, include=None, limit=None, offset=None):
            g = self._vs.get(
                self._name,
                ids=ids,
                where=where,
                limit=limit,
                include=include,
            )
            out = {"ids": g.ids}
            if include is None or "documents" in include:
                out["documents"] = g.documents
            if include is None or "metadatas" in include:
                out["metadatas"] = g.metadatas
            if include and "embeddings" in include:
                out["embeddings"] = g.embeddings
            return out

        def query(
            self,
            query_texts=None,
            query_embeddings=None,
            n_results=10,
            where=None,
            where_document=None,
            include=None,
        ):
            # Mirror chromadb default include (documents/metadatas/distances)
            effective_include = (
                include
                if include is not None
                else [
                    "documents",
                    "metadatas",
                    "distances",
                ]
            )
            q = self._vs.query(
                self._name,
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=effective_include,
            )
            out = {"ids": q.ids}
            if "documents" in effective_include:
                out["documents"] = q.documents
            if "metadatas" in effective_include:
                out["metadatas"] = q.metadatas
            if "distances" in effective_include:
                out["distances"] = q.distances
            return out

        def delete(self, ids=None, where=None):
            return self._vs.delete(self._name, ids=ids, where=where)

    col = _CollectionApiAdapter(vs, RECORDS_COLLECTION)
    yield col
    try:
        vs.delete_collection(RECORDS_COLLECTION)
    except Exception:
        pass
    reset_singletons()


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


# compat shim -- patch only canonical homes (intent.py for
# tool_declare_intent / tool_finalize_intent / tool_extend_feedback /
# tool_declare_operation; tool_mutate.py for tool_kg_declare_entity). Other
# modules (mcp_server, tool_lifecycle) forward to these via attribute lookup
# at call time, so a single patch on the canonical home applies everywhere
# without recursion. Auto-injects /11c/11e/12 mandatory args
# so legacy tests exercise the system rather than the validation gate.
@pytest.fixture(autouse=True)
def _v3_slice11_defaults(monkeypatch):
    import functools

    try:
        from mempalace import mcp_server, intent as _intent_mod, tool_mutate as _tool_mutate
    except Exception:
        yield
        return

    di_orig = getattr(_intent_mod, "tool_declare_intent", None)
    if di_orig is not None:

        @functools.wraps(di_orig)
        def di_wrapped(*a, **kw):
            kw.setdefault("initial_intent_state", {"todos": []})
            kw.setdefault("cause_id", "autonomous")
            return di_orig(*a, **kw)

        monkeypatch.setattr(_intent_mod, "tool_declare_intent", di_wrapped)

    de_orig = getattr(_tool_mutate, "tool_kg_declare_entity", None)
    if de_orig is not None:

        @functools.wraps(de_orig)
        def de_wrapped(*a, **kw):
            if kw.get("kind") == "entity":
                kg = getattr(mcp_server._STATE, "kg", None)
                if kg is not None:
                    try:
                        if not kg.get_entity("thing"):
                            kg.add_entity(
                                "thing",
                                kind="class",
                                content="Root class for all entities",
                                importance=5,
                            )
                    except Exception:
                        pass
                kw.setdefault("is_a", "thing")
            if kw.get("kind") == "record":
                kw.setdefault("entity", kw.get("name", "thing"))
            return de_orig(*a, **kw)

        monkeypatch.setattr(_tool_mutate, "tool_kg_declare_entity", de_wrapped)
        # mcp_server imports the function directly via 'from tool_mutate import tool_kg_declare_entity'
        # which binds the original; patch mcp_server's binding too.
        if hasattr(mcp_server, "tool_kg_declare_entity"):
            monkeypatch.setattr(mcp_server, "tool_kg_declare_entity", de_wrapped)

    # Adrian directive 2026-05-11 (judge-gated coverage, commit 1db3de6):
    # the old surfaced-instances always-cover rule is GONE -- state_deltas
    # are only demanded for entities the state_judge flags. The legacy
    # _augment_state_deltas helper that auto-injected agent +
    # active_context_id 'unchanged' entries to tool_declare_operation /
    # tool_finalize_intent / tool_extend_feedback is now harmful: the new
    # validator rejects unchanged-for-non-flagged entities, so the auto-
    # injection breaks every test that passes state_deltas=[]. Removed.
    # Tests that genuinely need state_deltas should pass them explicitly.
    yield
