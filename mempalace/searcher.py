#!/usr/bin/env python3
"""
searcher.py -- Find anything. Exact words.

Semantic search against the palace.
Returns verbatim text -- the actual words, never summaries.
"""

import logging
from pathlib import Path

import chromadb

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def search(query: str, palace_path: str, added_by: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim memory content.
    Optionally filter by added_by (agent name).
    """
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_records")
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}")

    # Build where filter
    where = {}
    if added_by:
        where = {"added_by": added_by}

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

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
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_records")
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    # Build where filter
    where = {}
    if added_by:
        where = {"added_by": added_by}

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
    except Exception as e:
        return {"error": f"Search error: {e}"}

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for rid, doc, meta, dist in zip(ids, docs, metas, dists):
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
