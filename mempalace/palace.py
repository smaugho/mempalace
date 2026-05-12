"""
palace.py -- Shared palace operations.

Consolidates vector-backend access patterns used by both miners and
the MCP server. Post-chromadb-removal (Adrian directive 2026-05-12)
the only backend is :class:`SqliteVecVectorStore`; ``get_collection``
returns a thin chromadb-Collection-shaped adapter over the active
:class:`VectorStore` so older miner code (``col.add(ids=..., docs=...,
metadatas=...)``, ``col.get(where=...)``) keeps working unchanged.
"""

import os

from .vector_store import (
    RECORDS_COLLECTION,
    VectorStore,
    get_vector_store,
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


class _PalaceCollectionAdapter:
    """Chromadb-Collection-shaped facade over a :class:`VectorStore` +
    collection name. Returns chromadb-shaped dicts (``ids`` /
    ``documents`` / ``metadatas`` / ``distances`` keys) so legacy
    callers (miner, convo_miner, ``file_already_mined``) keep working
    after chromadb removal."""

    def __init__(self, store: VectorStore, name: str):
        self._vs = store
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def count(self) -> int:
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

    def get(self, ids=None, where=None, include=None, limit=None, offset=None) -> dict:
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
    ) -> dict:
        # Match chromadb's Collection.query default: when include is
        # None, return documents + metadatas + distances. Several call
        # sites (entity_gate._find_identity_match etc.) read
        # ``results["distances"]`` without passing an explicit include
        # because they relied on this default.
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


def get_collection(palace_path: str, collection_name: str = RECORDS_COLLECTION):
    """Return a chromadb-Collection-shaped handle for the named palace
    collection.

    Today this routes through :func:`get_vector_store` (sqlite_vec
    backend) wrapped in :class:`_PalaceCollectionAdapter`. The shape
    of the returned object matches what callers used to get from
    ``chromadb.PersistentClient.get_or_create_collection`` -- enough
    of it for the miner + ``file_already_mined`` to work unchanged.
    """
    os.makedirs(palace_path, exist_ok=True)
    try:
        os.chmod(palace_path, 0o700)
    except (OSError, NotImplementedError):
        pass
    vs = get_vector_store(palace_path)
    # Register the collection name so list_collections / health reflect
    # it even before the first write lands.
    try:
        vs._open(collection_name, create=True)
    except Exception:
        # _open is a backwards-compat shim and not strictly required --
        # writes auto-create. Swallow and proceed.
        pass
    return _PalaceCollectionAdapter(vs, collection_name)


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
