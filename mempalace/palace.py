"""
palace.py -- Shared palace operations.

Consolidates vector-backend access patterns used by both miners and the
MCP server. Client construction goes through
:func:`mempalace.vector_store.make_vector_client` -- the single owner
of backend-client lifetime (Adrian directive 2026-05-11, branch-by-
abstraction Phase 1).
"""

import os

from .vector_store import (
    DEFAULT_COLLECTION_METADATA,
    make_vector_client,
)

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".mempalace",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}


def get_collection(palace_path: str, collection_name: str = "mempalace_records"):
    """Get or create the palace ChromaDB collection.

    Pinned to cosine distance -- the rest of the retrieval pipeline
    (MaxSim, 1-distance similarity) assumes cosine unconditionally, so we
    make it explicit at creation time rather than relying on ChromaDB's
    default (which could change).
    """
    os.makedirs(palace_path, exist_ok=True)
    try:
        os.chmod(palace_path, 0o700)
    except (OSError, NotImplementedError):
        pass
    # Client construction lives in vector_store.make_vector_client so
    # Settings(anonymized_telemetry=False) is applied once, in one
    # place -- matches the cache key VectorStore uses so a second-open
    # at the same palace doesn't raise ``ValueError: An instance of
    # Chroma already exists for ... with different settings`` (caught
    # 2026-05-09 by d6c8a71's _last_open_errors capture).
    client = make_vector_client(palace_path)
    try:
        return client.get_collection(collection_name)
    except Exception:
        # Slice 16 metadata (hnsw:space=cosine + hnsw:sync_threshold=100)
        # is owned by vector_store.DEFAULT_COLLECTION_METADATA so the
        # SIGSEGV-prevention threshold is set in exactly one place.
        return client.create_collection(
            collection_name,
            metadata=dict(DEFAULT_COLLECTION_METADATA),
        )


def file_already_mined(collection, source_file: str, check_mtime: bool = False) -> bool:
    """Check if a file has already been filed in the palace.

    When check_mtime=True (used by project miner), returns False if the file
    has been modified since it was last mined, so it gets re-mined.
    When check_mtime=False (used by convo miner), just checks existence.
    """
    try:
        results = collection.get(where={"source_file": source_file}, limit=1)
        if not results.get("ids"):
            return False
        if check_mtime:
            stored_meta = results.get("metadatas", [{}])[0]
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return float(stored_mtime) == current_mtime
        return True
    except Exception:
        return False
