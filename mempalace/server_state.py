"""
server_state.py — Instance-scoped MemPalace server state.

Module-level globals in mcp_server.py (_active_intent, _pending_conflicts,
_pending_enrichments, _client_cache, _collection_cache, _declared_entities,
_session_id, _session_state, migration flags) used to live as bare module
attributes. That shape made tests fragile — state leaked between test cases
and parallel pytest-xdist workers couldn't be trusted — and blocked any
future where a single process hosts more than one MCP client.

ServerState collects them on an instance. mcp_server.py holds a single
default `_STATE` and reassigns its module globals through this object, so
external references (intent.py helpers, tests) keep working because the
globals still exist — they just point at fields on a single state object.
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

    # Active intent lifecycle — at most one per session.
    active_intent: Optional[dict] = None

    # Blocks-all-tools state: unresolved conflicts + enrichments.
    pending_conflicts: Optional[list] = None
    pending_enrichments: Optional[list] = None

    # Intent-scope declared entities (in-memory cache backed by KG).
    declared_entities: set = field(default_factory=set)

    # Per-session isolation so multiple callers sharing one process don't
    # stomp each other. session_id == "" means "default".
    session_id: str = ""
    session_state: dict = field(default_factory=dict)

    # One-time migration flags (tripped on first-touch, then idempotent).
    entity_views_migrated: bool = False
    kind_rename_migrated: bool = False

    def reset_transient(self) -> None:
        """Clear per-test / per-session transient state.

        Keeps config + kg + ChromaDB caches (those are expensive to rebuild).
        """
        self.active_intent = None
        self.pending_conflicts = None
        self.pending_enrichments = None
        self.declared_entities = set()
        self.session_id = ""
        self.session_state = {}
