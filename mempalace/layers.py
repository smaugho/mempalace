#!/usr/bin/env python3
"""
layers.py — 2-Layer Memory Stack for mempalace
===================================================

Load only what you need, when you need it.

    Layer 0: Identity       (~100 tokens)   — Always loaded. "Who am I?"
    Layer 1: Essential Story (~500-800)      — Always loaded. Top moments ranked by
                                               importance + decay + agent affinity.

Wake-up cost: ~600-900 tokens (L0+L1). Leaves 95%+ of context free.
Deep search is handled by kg_search (scoring.multi_channel_search).

Reads directly from ChromaDB (mempalace_records)
and ~/.mempalace/identity.txt.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from .config import MempalaceConfig
from .scoring import hybrid_score as _hybrid_score_fn, adaptive_k


TIER_MULTIPLIER = 10.0  # importance tier gap — ensures higher tier ALWAYS wins
DECAY_WEIGHT = 0.5  # log10-decay weight applied to age_days within a tier


def _parse_iso_datetime(value: str):
    """Parse ISO-format datetime string, tolerant of trailing Z and timezone."""
    if not value or not isinstance(value, str):
        return None
    try:
        s = value.strip()
        # Normalize trailing 'Z' to +00:00 for fromisoformat
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


# compute_decay_score() removed: use scoring.hybrid_score(mode="l1") directly.
# Invariants preserved by the unified scoring function:
#   - importance=5 always outranks lower tiers regardless of age
#   - within a tier, newer memories score higher
#   - missing date treated as 365 days old


# ---------------------------------------------------------------------------
# Layer 0 — Identity
# ---------------------------------------------------------------------------


class Layer0:
    """
    ~100-300 tokens. Always loaded.

    Identity is derived from the KG entity graph: finds the agent entity's
    described-by memories and loads them as identity text. Falls back to
    ~/.mempalace/identity.txt if no KG identity is found.

    The agent entity is looked up directly by name (e.g. ga_agent).
    """

    def __init__(self, identity_path: str = None, palace_path: str = None, agent: str = None):
        if identity_path is None:
            identity_path = os.path.expanduser("~/.mempalace/identity.txt")
        self.path = identity_path
        self.palace_path = palace_path
        self.agent = agent
        self._text = None

    def _load_from_kg(self) -> str:
        """Try to load identity from KG entity's described-by memories."""
        if not self.agent or not self.palace_path:
            return None
        try:
            from .knowledge_graph import KnowledgeGraph, normalize_entity_name

            kg = KnowledgeGraph()

            agent_id = normalize_entity_name(self.agent)
            entity = kg.get_entity(agent_id)
            if not entity:
                return None

            # Get described-by edges -> load those memories
            edges = kg.query_entity(agent_id, direction="outgoing")
            described_by_ids = [
                e["object"] for e in edges if e["predicate"] == "described_by" and e["current"]
            ]

            if not described_by_ids:
                return None

            # Load memory content from ChromaDB
            client = chromadb.PersistentClient(path=self.palace_path)
            col = client.get_collection("mempalace_records")

            # Try both underscore and hyphen forms of IDs
            all_ids = []
            for did in described_by_ids:
                all_ids.append(did)
                all_ids.append(did.replace("-", "_"))

            result = col.get(ids=all_ids, include=["documents"])
            if not result["documents"]:
                return None

            # Build identity text from memory content
            parts = [f"## L0 — IDENTITY (from entity: {entity['name']})"]
            parts.append(f"Description: {entity['description']}")
            parts.append("")
            for doc in result["documents"]:
                if doc:
                    snippet = doc.strip()
                    if len(snippet) > 500:
                        snippet = snippet[:497] + "..."
                    parts.append(snippet)
                    parts.append("")

            return "\n".join(parts)
        except Exception:
            return None

    def render(self) -> str:
        """Return identity text from KG, falling back to identity.txt file."""
        if self._text is not None:
            return self._text

        # Try KG-based identity first
        kg_identity = self._load_from_kg()
        if kg_identity:
            self._text = kg_identity
            return self._text

        # Fall back to file
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self._text = f.read().strip()
        else:
            self._text = (
                "## L0 — IDENTITY\nNo identity configured. "
                "Declare an agent entity with described-by memories, "
                "or create ~/.mempalace/identity.txt"
            )

        return self._text

    def token_estimate(self) -> int:
        return len(self.render()) // 4


# ---------------------------------------------------------------------------
# Layer 1 — Essential Story (auto-generated from palace)
# ---------------------------------------------------------------------------


class Layer1:
    """
    ~500-800 tokens. Always loaded.
    Auto-generated from top-scoring records, ranked by
    importance with power-law decay time factor.

    See `scoring.hybrid_score(mode="l1")` for details. Importance=5 always
    outranks lower tiers for typical ages; within a tier, newer records win.
    """

    MAX_DRAWERS = 15  # at most 15 moments in wake-up
    MAX_CHARS = 3200  # hard cap on total L1 text (~800 tokens)

    def __init__(self, palace_path: str = None, agent: str = None):
        cfg = MempalaceConfig()
        self.palace_path = palace_path or cfg.palace_path
        self.agent = agent

    def generate(self) -> str:  # noqa: C901
        """Pull top entries from BOTH ChromaDB collections and format as compact L1 text.

        Unified retrieval. L1 includes entity descriptions
        alongside record content. Rules, concepts, gotchas, and past
        execution summaries compete with prose records for the top-K slots.
        Both collections are fetched in batches, scored identically by
        importance + power-law decay + agent affinity, and merged before
        adaptive-K cuts.
        """
        client = None
        try:
            client = chromadb.PersistentClient(path=self.palace_path)
        except Exception:
            return "## L1 — No palace found. Run: mempalace mine <dir>"

        docs, metas = [], []
        _BATCH = 500

        # Fetch from record collection (prose records)
        try:
            col = client.get_collection("mempalace_records")
            offset = 0
            while True:
                kwargs = {
                    "include": ["documents", "metadatas"],
                    "limit": _BATCH,
                    "offset": offset,
                }
                if self.agent:
                    kwargs["where"] = {"added_by": self.agent}
                try:
                    batch = col.get(**kwargs)
                except Exception:
                    break
                batch_docs = batch.get("documents", [])
                batch_metas = batch.get("metadatas", [])
                if not batch_docs:
                    break
                docs.extend(batch_docs)
                metas.extend(batch_metas)
                offset += len(batch_docs)
                if len(batch_docs) < _BATCH:
                    break
        except Exception:
            pass

        # Also fetch from entity collection (rules, concepts, gotchas,
        # past executions). Filter to high-importance entities only (≥4) to
        # avoid flooding L1 with low-value entity descriptions.
        try:
            ecol = client.get_collection("mempalace_entities")
            offset = 0
            while True:
                kwargs = {
                    "include": ["documents", "metadatas"],
                    "limit": _BATCH,
                    "offset": offset,
                }
                try:
                    batch = ecol.get(**kwargs)
                except Exception:
                    break
                batch_docs = batch.get("documents", [])
                batch_metas = batch.get("metadatas", [])
                if not batch_docs:
                    break
                for doc, meta in zip(batch_docs, batch_metas):
                    meta = meta or {}
                    # Only include importance ≥ 4 entities in L1 wake_up
                    try:
                        imp = float(meta.get("importance", 3))
                    except (TypeError, ValueError):
                        imp = 3.0
                    if imp >= 4.0:
                        docs.append(doc)
                        metas.append(meta)
                offset += len(batch_docs)
                if len(batch_docs) < _BATCH:
                    break
        except Exception:
            pass

        if not docs:
            return "## L1 — No entries yet."

        # Score each record: importance with power-law decay time factor
        scored = []
        for doc, meta in zip(docs, metas):
            importance = 3.0
            # Try multiple metadata keys that might carry weight info
            for key in ("importance", "emotional_weight", "weight"):
                val = meta.get(key)
                if val is not None:
                    try:
                        importance = float(val)
                    except (ValueError, TypeError):
                        pass
                    break
            # Pull date for decay calculation; prefer date_added, fall back to filed_at
            date_iso = meta.get("date_added") or meta.get("filed_at") or ""
            # Pull last_relevant_at to reset decay clock on recently-used records
            last_relevant_iso = meta.get("last_relevant_at") or None
            # Provenance affinity: boost items from same agent/session
            agent_match = bool(self.agent and meta.get("added_by") == self.agent)
            session_match = bool(
                hasattr(self, "_session_id")
                and self._session_id
                and meta.get("session_id") == self._session_id
            )
            score = _hybrid_score_fn(
                similarity=0.0,
                importance=importance,
                date_iso=date_iso,
                agent_match=agent_match,
                last_relevant_iso=last_relevant_iso,
                relevance_feedback=0,
                mode="l1",
                session_match=session_match,
            )
            scored.append((score, importance, meta, doc))

        # Sort by combined score descending, use adaptive-K
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) > 1:
            k = adaptive_k([s[0] for s in scored], max_k=self.MAX_DRAWERS, min_k=3)
        else:
            k = len(scored)
        top = [(importance, meta, doc) for (_score, importance, meta, doc) in scored[:k]]

        # Group by content_type for readability
        from collections import defaultdict

        by_type = defaultdict(list)
        for imp, meta, doc in top:
            ct = meta.get("content_type") or meta.get("type") or "general"
            by_type[ct].append((imp, meta, doc))

        # Build compact text
        lines = ["## L1 — ESSENTIAL STORY"]

        total_len = 0
        for ct, entries in sorted(by_type.items()):
            type_line = f"\n[{ct}]"
            lines.append(type_line)
            total_len += len(type_line)

            for imp, meta, doc in entries:
                source = Path(meta.get("source_file", "")).name if meta.get("source_file") else ""

                # Truncate doc to keep L1 compact
                snippet = doc.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."

                entry_line = f"  - {snippet}"
                if source:
                    entry_line += f"  ({source})"

                if total_len + len(entry_line) > self.MAX_CHARS:
                    lines.append("  ... (more in search)")
                    return "\n".join(lines)

                lines.append(entry_line)
                total_len += len(entry_line)

        return "\n".join(lines)


# only — no multi-view, no keywords, no graph, no entity collection.
# kg_search (via scoring.multi_channel_search) IS the real deep search.


class _Layer3Removed:
    """Stub so legacy callers don't crash. search() returns empty."""

    def __init__(self, palace_path: str = None):
        pass

    def search(self, query: str, **kwargs) -> str:
        return "Layer3 removed. Use mempalace_kg_search instead."

    def search_raw(self, query: str, **kwargs) -> list:
        return []


# ---------------------------------------------------------------------------
# MemoryStack — unified interface
# ---------------------------------------------------------------------------


class MemoryStack:
    """
    The memory stack. One class, one palace, everything works.

        stack = MemoryStack()
        print(stack.wake_up(agent="ga_agent"))  # L0 + L1 (~600-900 tokens)
    """

    def __init__(self, palace_path: str = None, identity_path: str = None):
        cfg = MempalaceConfig()
        self.palace_path = palace_path or cfg.palace_path
        self.identity_path = identity_path or os.path.expanduser("~/.mempalace/identity.txt")

        self.l0 = Layer0(self.identity_path, palace_path=self.palace_path)
        self.l1 = Layer1(self.palace_path)
        self.l3 = _Layer3Removed(self.palace_path)

    def wake_up(self, agent: str = None) -> str:
        """
        Generate wake-up text: L0 (identity) + L1 (essential story).
        Typically ~600-900 tokens. Inject into system prompt or first message.

        Args:
            agent: Agent identity for affinity scoring in L1 ranking.
        """
        parts = []

        # L0: Identity (pass agent so it can find agent entity)
        self.l0.agent = agent
        parts.append(self.l0.render())
        parts.append("")

        # L1: Essential Story
        if agent:
            self.l1.agent = agent
        parts.append(self.l1.generate())

        return "\n".join(parts)

    def search(self, query: str, n_results: int = 5) -> str:
        """Deep search — use mempalace_kg_search instead."""
        return self.l3.search(query, n_results=n_results)

    def status(self) -> dict:
        """Status of all layers."""
        result = {
            "palace_path": self.palace_path,
            "L0_identity": {
                "path": self.identity_path,
                "exists": os.path.exists(self.identity_path),
                "tokens": self.l0.token_estimate(),
            },
            "L1_essential": {
                "description": "Auto-generated from top records",
            },
        }

        # Count records
        try:
            client = chromadb.PersistentClient(path=self.palace_path)
            col = client.get_collection("mempalace_records")
            count = col.count()
            result["total_records"] = count
        except Exception:
            result["total_records"] = 0

        return result


# ---------------------------------------------------------------------------
# CLI (standalone)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def usage():
        print("layers.py — 2-Layer Memory Stack")
        print()
        print("Usage:")
        print("  python layers.py wake-up               Show L0 + L1")
        print("  python layers.py wake-up --agent=NAME   Wake-up for a specific agent")
        print("  python layers.py status                 Show layer status")
        sys.exit(0)

    if len(sys.argv) < 2:
        usage()

    cmd = sys.argv[1]

    # Parse flags
    flags = {}
    positional = []
    for arg in sys.argv[2:]:
        if arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
            flags[key.lstrip("-")] = val
        elif not arg.startswith("--"):
            positional.append(arg)

    palace_path = flags.get("palace")
    stack = MemoryStack(palace_path=palace_path)

    if cmd in ("wake-up", "wakeup"):
        agent = flags.get("agent")
        text = stack.wake_up(agent=agent)
        tokens = len(text) // 4
        print(f"Wake-up text (~{tokens} tokens):")
        print("=" * 50)
        print(text)

    elif cmd == "status":
        s = stack.status()
        print(json.dumps(s, indent=2))

    else:
        usage()
