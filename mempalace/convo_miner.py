#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the knowledge store.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to store.

Same store as project mining. Different ingest strategy.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .normalize import normalize
from .palace import SKIP_DIRS, get_collection, file_already_mined


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

MIN_CHUNK_SIZE = 30
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB — skip files larger than this


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(content: str) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.
    """
    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines)
    else:
        return _chunk_by_paragraph(content)


def _chunk_by_exchange(lines: list) -> list:
    """One user turn (>) + the AI response that follows = one chunk."""
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith("---"):
                    break
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            ai_response = " ".join(ai_lines[:8])
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            if len(content.strip()) > MIN_CHUNK_SIZE:
                chunks.append(
                    {
                        "content": content,
                        "chunk_index": len(chunks),
                    }
                )
        else:
            i += 1

    return chunks


def _chunk_by_paragraph(content: str) -> list:
    """Fallback: chunk by paragraph breaks."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > 20:
        lines = content.split("\n")
        for i in range(0, len(lines), 25):
            group = "\n".join(lines[i : i + 25]).strip()
            if len(group) > MIN_CHUNK_SIZE:
                chunks.append({"content": group, "chunk_index": len(chunks)})
        return chunks

    for para in paragraphs:
        if len(para) > MIN_CHUNK_SIZE:
            chunks.append({"content": para, "chunk_index": len(chunks)})

    return chunks


# =============================================================================
# CONTENT TYPE CLASSIFICATION — topic-based for conversations
# =============================================================================

CONTENT_TYPE_KEYWORDS = {
    "fact": [
        "code",
        "python",
        "function",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
    ],
    "event": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
    ],
    "discovery": [
        "bug",
        "error",
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
        "debug",
        "refactor",
    ],
    "preference": [
        "prefer",
        "like",
        "dislike",
        "favorite",
        "always",
        "never",
        "style",
        "convention",
    ],
    "advice": [
        "trade-off",
        "alternative",
        "option",
        "approach",
        "priority",
        "recommendation",
        "suggestion",
        "best practice",
    ],
}


def detect_content_type(content: str) -> str:
    """Score conversation content against content_type keywords."""
    content_lower = content[:3000].lower()
    scores = {}
    for ctype, keywords in CONTENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[ctype] = score
    if scores:
        return max(scores, key=scores.get)
    return "fact"


# =============================================================================
# SCAN FOR CONVERSATION FILES
# =============================================================================


def scan_convos(convo_dir: str) -> list:
    """Find all potential conversation files."""
    convo_path = Path(convo_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files
                if filepath.is_symlink():
                    continue
                try:
                    if filepath.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


def mine_convos(
    convo_dir: str,
    palace_path: str,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
):
    """Mine a directory of conversation files into the store.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions
    """

    convo_path = Path(convo_dir).expanduser().resolve()

    files = scan_convos(convo_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  MemPalace Mine — Conversations")
    print(f"{'=' * 55}")
    print(f"  Agent:   {agent}")
    print(f"  Source:  {convo_path}")
    print(f"  Files:   {len(files)}")
    print(f"  Store:   {palace_path}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'-' * 55}\n")

    collection = get_collection(palace_path) if not dry_run else None

    total_drawers = 0
    files_skipped = 0
    type_counts = defaultdict(int)

    for i, filepath in enumerate(files, 1):
        source_file = str(filepath)

        # Skip if already filed
        if not dry_run and file_already_mined(collection, source_file):
            files_skipped += 1
            continue

        # Normalize format
        try:
            content = normalize(str(filepath))
        except (OSError, ValueError):
            continue

        if not content or len(content.strip()) < MIN_CHUNK_SIZE:
            continue

        # Chunk — either exchange pairs or general extraction
        if extract_mode == "general":
            from .general_extractor import extract_memories

            chunks = extract_memories(content)
            # Each chunk already has memory_type; use it as the content_type
        else:
            chunks = chunk_exchanges(content)

        if not chunks:
            continue

        # Detect content_type from content (general mode uses memory_type instead)
        if extract_mode != "general":
            content_type = detect_content_type(content)
        else:
            content_type = None  # set per-chunk below

        if dry_run:
            if extract_mode == "general":
                from collections import Counter

                ct_counts = Counter(c.get("memory_type", "fact") for c in chunks)
                types_str = ", ".join(f"{t}:{n}" for t, n in ct_counts.most_common())
                print(f"    [DRY RUN] {filepath.name} → {len(chunks)} records ({types_str})")
            else:
                print(f"    [DRY RUN] {filepath.name} → {content_type} ({len(chunks)} records)")
            total_drawers += len(chunks)
            # Track type counts
            if extract_mode == "general":
                for c in chunks:
                    type_counts[c.get("memory_type", "fact")] += 1
            else:
                type_counts[content_type] += 1
            continue

        if extract_mode != "general":
            type_counts[content_type] += 1

        # File each chunk
        drawers_added = 0
        for chunk in chunks:
            chunk_type = (
                chunk.get("memory_type", content_type)
                if extract_mode == "general"
                else content_type
            )
            if extract_mode == "general":
                type_counts[chunk_type] += 1
            slug = hashlib.sha256((source_file + str(chunk["chunk_index"])).encode()).hexdigest()[
                :24
            ]
            record_id = f"record_{agent}_{slug}"
            try:
                collection.upsert(
                    documents=[chunk["content"]],
                    ids=[record_id],
                    metadatas=[
                        {
                            "source_file": source_file,
                            "chunk_index": chunk["chunk_index"],
                            "added_by": agent,
                            "content_type": chunk_type,
                            "filed_at": datetime.now().isoformat(),
                            "ingest_mode": "convos",
                            "extract_mode": extract_mode,
                        }
                    ],
                )
                drawers_added += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

        total_drawers += drawers_added
        print(f"  ✓ [{i:4}/{len(files)}] {filepath.name[:50]:50} +{drawers_added}")

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Records filed: {total_drawers}")
    if type_counts:
        print("\n  By content_type:")
        for ctype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {ctype:20} {count} files")
    print('\n  Next: mempalace search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--palace PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from .config import MempalaceConfig

    mine_convos(sys.argv[1], palace_path=MempalaceConfig().palace_path)
