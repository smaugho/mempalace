"""
server_state.py -- Instance-scoped MemPalace server state.

Per-session transient state (active intent, pending conflicts, declared
entities, session id, ChromaDB client + collection caches, one-shot
migration flags, plus the config + KnowledgeGraph handles) all live on a
single ``ServerState`` instance. mcp_server.py constructs a module-level
``_STATE`` at import time; every handler, helper, intent.py path, and test
patcher reaches state exclusively through that instance.

The previous design kept each field as a bare module global on mcp_server.
That shape made tests fragile -- state leaked between test cases and
pytest-xdist workers couldn't be trusted -- and blocked any future where a
single process hosts more than one MCP client. ServerState fixes both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ServerState:
    """Mutable per-instance state for a MemPalace MCP server.

    All fields default to None/empty so a caller can construct a bare
    ``ServerState()`` for tests and fill in only what the test exercises.
    Production code builds it with concrete ``config`` and ``kg``.
    """

    config: Any = None
    kg: Any = None

    # ChromaDB lazy caches.
    client_cache: Any = None
    collection_cache: Any = None

    # Active intent lifecycle -- at most one per session.
    active_intent: Optional[dict] = None

    # Blocks-all-tools state: unresolved conflicts.
    pending_conflicts: Optional[list] = None

    # Intent-scope declared entities (in-memory cache backed by KG).
    declared_entities: set = field(default_factory=set)

    # Per-session isolation so multiple callers sharing one process don't
    # stomp each other. Empty string means "no sid known" -- state-writing
    # tools refuse to proceed in that case (see _require_sid), and no
    # cross-agent fallback file is ever written.
    session_id: str = ""
    session_state: dict = field(default_factory=dict)

    # One-time migration flags (tripped on first-touch, then idempotent).
    entity_views_migrated: bool = False
    kind_rename_migrated: bool = False
    # N3 identifier normalization: rename legacy hyphenated IDs (Chroma +
    # SQLite) to their canonical underscored form from normalize_entity_name.
    hyphen_ids_migrated: bool = False
    # M1 physical merge: absorbs legacy mempalace_entities collection into
    # the unified mempalace_records collection, then drops the legacy one.
    entity_collection_merged: bool = False

    # P3 polish: one-shot drop of the retired mempalace_feedback_contexts
    # Chroma collection. Migration 015 dropped the SQLite keyword_feedback
    # / edge_traversal_feedback tables, but the Chroma collection can't
    # be dropped from SQL -- so we do it on startup, once, gated here.
    feedback_contexts_dropped: bool = False

    def reset_transient(self) -> None:
        """Clear per-test / per-session transient state.

        Keeps config + kg + ChromaDB caches (those are expensive to rebuild).
        """
        self.active_intent = None
        self.pending_conflicts = None
        self.declared_entities = set()
        self.session_id = ""
        self.session_state = {}
