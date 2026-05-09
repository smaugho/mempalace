#!/usr/bin/env python3
"""
searcher.py -- Find anything. Exact words.

Semantic search against the palace.
Returns verbatim text -- the actual words, never summaries.

Repository pattern (Adrian directive 2026-05-09): all Chroma access
goes through ``mempalace.vector_store.VectorStore`` -- this module
no longer imports chromadb directly. The poisoning short-circuit, the
typed result objects, and any future DB-backend swap all live in
``vector_store.py``.
"""

import logging
from pathlib import Path

from mempalace.vector_store import RECORDS_COLLECTION, get_vector_store

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def search(query: str, palace_path: str, added_by: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim memory content.
    Optionally filter by added_by (agent name).
    """
    vs = get_vector_store(palace_path)
    where = {"added_by": added_by} if added_by else None

    qres = vs.query(
        RECORDS_COLLECTION,
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    if qres.is_degraded:
        # "unavailable" maps to the prior "no palace found" CLI behaviour;
        # poisoned / failed collapses to "no results" so degradation is
        # visible but doesn't crash the CLI.
        if "unavailable" in qres.degraded_reason:
            print(f"\n  No palace found at {palace_path}")
            print("  Run: mempalace init <dir> then mempalace mine <dir>")
            raise SearchError(f"No palace found at {palace_path}")
        print(f"\n  Search degraded: {qres.degraded_reason}")
        return

    docs = qres.documents[0] if qres.documents else []
    metas = qres.metadatas[0] if qres.metadatas else []
    dists = qres.distances[0] if qres.distances else []

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if added_by:
        print(f"  Agent: {added_by}")
    print(f"{'=' * 60}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        similarity = round(1 - dist, 3)
        source = Path(meta.get("source_file", "?")).name
        agent = meta.get("added_by", "?")
        content_type = meta.get("content_type", "?")

        print(f"  [{i}] {agent} / {content_type}")
        print(f"      Source: {source}")
        print(f"      Match:  {similarity}")
        print()
        # Print the verbatim text, indented
        for line in doc.strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def search_memories(query: str, palace_path: str, added_by: str = None, n_results: int = 5) -> dict:
    """
    Programmatic search -- returns a dict instead of printing.
    Used by the MCP server and other callers that need data.
    """
    vs = get_vector_store(palace_path)
    where = {"added_by": added_by} if added_by else None

    qres = vs.query(
        RECORDS_COLLECTION,
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    if qres.is_degraded:
        if "unavailable" in qres.degraded_reason:
            logger.error("No palace found at %s: %s", palace_path, qres.degraded_reason)
            return {
                "error": "No palace found",
                "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
            }
        # Poisoned / failed -- empty results with degraded marker so
        # callers can distinguish "real empty" from "skipped due to corruption".
        return {
            "query": query,
            "filters": {"added_by": added_by},
            "results": [],
            "degraded": True,
            "degraded_reason": qres.degraded_reason,
        }

    ids = qres.ids[0] if qres.ids else []
    docs = qres.documents[0] if qres.documents else []
    metas = qres.metadatas[0] if qres.metadatas else []
    dists = qres.distances[0] if qres.distances else []

    hits = []
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        hits.append(
            {
                "id": rid,
                # Vocab lock 2026-05-01: rendered memory preview is the
                # canonical "summary_text" key everywhere it appears in a
                # response payload. The doc here is the Chroma stored
                # document for the record, which IS the rendered prose.
                "summary_text": doc,
                "added_by": meta.get("added_by", "unknown"),
                "content_type": meta.get("content_type", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(1 - dist, 3),
                "metadata": meta,  # Full metadata for re-ranking (agent affinity, etc.)
            }
        )

    return {
        "query": query,
        "filters": {"added_by": added_by},
        "results": hits,
    }
