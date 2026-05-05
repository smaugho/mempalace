#!/usr/bin/env python3
"""
mempalace migrate -- Recover a palace created with a different ChromaDB version.

Reads documents and metadata directly from the palace's SQLite database
(bypassing ChromaDB's API, which fails on version-mismatched palaces),
then re-imports everything into a fresh palace using the currently installed
ChromaDB version.

This fixes the 3.0.0 → 3.1.0 upgrade path where chromadb was downgraded
from 1.5.x to 0.6.x, breaking the on-disk storage format.

Usage:
    mempalace migrate                          # migrate default palace
    mempalace migrate --palace /path/to/palace  # migrate specific palace
    mempalace migrate --dry-run                # show what would be migrated
"""

import os
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime


def extract_drawers_from_sqlite(db_path: str) -> list:
    """Read all memories directly from ChromaDB's SQLite, bypassing the API.

    Works regardless of which ChromaDB version created the database.
    Returns list of dicts with 'id', 'document', and 'metadata' keys.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get all embedding IDs and their documents
    rows = conn.execute("""
        SELECT e.embedding_id,
               MAX(CASE WHEN em.key = 'chroma:document' THEN em.string_value END) as document
        FROM embeddings e
        JOIN embedding_metadata em ON em.id = e.id
        GROUP BY e.embedding_id
    """).fetchall()

    memories = []
    for row in rows:
        embedding_id = row["embedding_id"]
        document = row["document"]
        if not document:
            continue

        # Get metadata for this embedding
        meta_rows = conn.execute(
            """
            SELECT em.key, em.string_value, em.int_value, em.float_value, em.bool_value
            FROM embedding_metadata em
            JOIN embeddings e ON e.id = em.id
            WHERE e.embedding_id = ?
              AND em.key NOT LIKE 'chroma:%'
        """,
            (embedding_id,),
        ).fetchall()

        metadata = {}
        for mr in meta_rows:
            key = mr["key"]
            if mr["string_value"] is not None:
                metadata[key] = mr["string_value"]
            elif mr["int_value"] is not None:
                metadata[key] = mr["int_value"]
            elif mr["float_value"] is not None:
                metadata[key] = mr["float_value"]
            elif mr["bool_value"] is not None:
                metadata[key] = bool(mr["bool_value"])

        memories.append(
            {
                "id": embedding_id,
                "document": document,
                "metadata": metadata,
            }
        )

    conn.close()
    return memories


def detect_chromadb_version(db_path: str) -> str:
    """Detect which ChromaDB version created the database by checking schema."""
    conn = sqlite3.connect(db_path)
    try:
        # 1.x has schema_str column in collections table
        cols = [r[1] for r in conn.execute("PRAGMA table_info(collections)").fetchall()]
        if "schema_str" in cols:
            return "1.x"
        # 0.6.x has embeddings_queue but no schema_str
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        if "embeddings_queue" in tables:
            return "0.6.x"
        return "unknown"
    finally:
        conn.close()


def migrate(palace_path: str, dry_run: bool = False):
    """Migrate a palace to the currently installed ChromaDB version."""
    import chromadb

    palace_path = os.path.expanduser(palace_path)
    db_path = os.path.join(palace_path, "chroma.sqlite3")

    if not os.path.isfile(db_path):
        print(f"\n  No palace database found at {db_path}")
        return False

    print(f"\n{'=' * 60}")
    print("  MemPalace Migrate")
    print(f"{'=' * 60}\n")
    print(f"  Palace:    {palace_path}")
    print(f"  Database:  {db_path}")
    print(f"  DB size:   {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")

    # Detect version
    source_version = detect_chromadb_version(db_path)
    print(f"  Source:    ChromaDB {source_version}")
    print(f"  Target:    ChromaDB {chromadb.__version__}")

    # Try reading with current chromadb first
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_records")
        count = col.count()
        print(f"\n  Palace is already readable by chromadb {chromadb.__version__}.")
        print(f"  {count} memories found. No migration needed.")
        return True
    except Exception:
        print(f"\n  Palace is NOT readable by chromadb {chromadb.__version__}.")
        print("  Extracting from SQLite directly...")

    # Extract all memories via raw SQL
    memories = extract_drawers_from_sqlite(db_path)
    print(f"  Extracted {len(memories)} memories from SQLite")

    if not memories:
        print("  Nothing to migrate.")
        return True

    # Show summary
    by_agent = defaultdict(lambda: defaultdict(int))
    for d in memories:
        agent = d["metadata"].get("added_by", "?")
        ctype = d["metadata"].get("content_type", "?")
        by_agent[agent][ctype] += 1

    print("\n  Summary:")
    for agent, types in sorted(by_agent.items()):
        total = sum(types.values())
        print(f"    Agent: {agent} ({total} memories)")
        for ctype, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"      {ctype:30} {count:5}")

    if dry_run:
        print("\n  DRY RUN -- no changes made.")
        print(f"  Would migrate {len(memories)} memories.")
        return True

    # Backup the old palace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{palace_path}.pre-migrate.{timestamp}"
    print(f"\n  Backing up to {backup_path}...")
    shutil.copytree(palace_path, backup_path)

    # Build fresh palace in a temp directory (avoids chromadb reading old state)
    import tempfile

    temp_palace = tempfile.mkdtemp(prefix="mempalace_migrate_")
    print(f"  Creating fresh palace in {temp_palace}...")
    client = chromadb.PersistentClient(path=temp_palace)
    # Slice 16: low sync_threshold so an interrupted migration leaves
    # at most ~100 unprocessed rows in embeddings_queue (vs the 1000
    # default that triggers the C-level _apply_batch SIGSEGV on next
    # boot).
    col = client.get_or_create_collection(
        "mempalace_records",
        metadata={"hnsw:space": "cosine", "hnsw:sync_threshold": 100},
    )

    # Re-import in batches
    batch_size = 500
    imported = 0
    for i in range(0, len(memories), batch_size):
        batch = memories[i : i + batch_size]
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        imported += len(batch)
        print(f"  Imported {imported}/{len(memories)} memories...")

    # Verify before swapping
    final_count = col.count()
    del col
    del client

    # Swap: remove old palace, move new one into place
    print("  Swapping old palace for migrated version...")
    shutil.rmtree(palace_path)
    shutil.move(temp_palace, palace_path)

    print("\n  Migration complete.")
    print(f"  Memories migrated: {final_count}")
    print(f"  Backup at: {backup_path}")

    if final_count != len(memories):
        print(f"  WARNING: Expected {len(memories)}, got {final_count}")

    print(f"\n{'=' * 60}\n")
    return True
