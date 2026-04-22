"""
MemPalace configuration system.

Priority: env vars > config file (~/.mempalace/config.json) > defaults
"""

import json
import os
import re
from pathlib import Path


# ── Input validation ──────────────────────────────────────────────────────────
# Shared sanitizers for entity names. Prevents path traversal,
# excessively long strings, and special characters that could cause issues
# in SQLite or ChromaDB metadata.

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_ .'-]{0,126}[a-zA-Z0-9]?$")


def _suggest_slug_hint(value: str) -> str:
    """Produce a human-readable slug suggestion for error-message hints.

    NOT an identifier normalizer — the only consumer is sanitize_name's error
    path, which shows the user a suggested replacement when their input is
    rejected. The canonical identifier normalizer is normalize_entity_name.

    Replaces path separators, colons, and other invalid characters with hyphens,
    collapses repeats, and trims edges.
    """
    if not isinstance(value, str):
        return "entity"
    # Replace invalid chars with hyphen
    slug = re.sub(r"[^a-zA-Z0-9_ .'-]", "-", value.strip())
    # Collapse repeated hyphens
    slug = re.sub(r"-+", "-", slug)
    # Trim hyphens/underscores/dots at edges
    slug = slug.strip("-_. ")
    # Ensure starts/ends alphanumeric (required by _SAFE_NAME_RE)
    if slug and not slug[0].isalnum():
        slug = "x" + slug
    if slug and not slug[-1].isalnum():
        slug = slug + "x"
    # Truncate
    if len(slug) > MAX_NAME_LENGTH:
        slug = slug[:MAX_NAME_LENGTH].rstrip("-_. ")
    return slug or "entity"


def sanitize_name(value: str, field_name: str = "name") -> str:
    """Validate and sanitize an entity name.

    Raises ValueError with actionable hints (suggested slug) if invalid.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string (got {value!r}). "
            f"Example: 'ga', 'secrets', 'flowsev-repo'."
        )

    value = value.strip()

    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(
            f"{field_name} '{value[:40]}...' exceeds max length {MAX_NAME_LENGTH} "
            f"(got {len(value)} chars). Shorten it or split across multiple memories."
        )

    # Block path traversal — give the user a concrete fix
    bad_chars = [c for c in ("..", "/", "\\", ":") if c in value]
    if bad_chars:
        suggestion = _suggest_slug_hint(value)
        raise ValueError(
            f"{field_name} '{value}' rejected: contains path characters "
            f"{bad_chars} (path traversal protection). "
            f"Use a hyphenated slug instead — e.g. '{suggestion}'."
        )

    # Block null bytes
    if "\x00" in value:
        raise ValueError(
            f"{field_name} contains null bytes (check source data for stray \\x00). "
            f"Strip with `value.replace('\\x00', '')` before passing."
        )

    # Enforce safe character set — tell them the rule and suggest a slug
    if not _SAFE_NAME_RE.match(value):
        suggestion = _suggest_slug_hint(value)
        raise ValueError(
            f"{field_name} '{value}' rejected: must start and end with "
            f"alphanumeric, and contain only letters, digits, spaces, "
            f"underscores, hyphens, dots, or apostrophes (max 128 chars). "
            f"Try: '{suggestion}'."
        )

    return value


def sanitize_content(value: str, max_length: int = 100_000) -> str:
    """Validate memory/diary content length. Raises ValueError with hints."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"content must be a non-empty string (got {type(value).__name__}: {str(value)[:50]!r}). "
            f"Pass the verbatim text you want to store in the memory."
        )
    if len(value) > max_length:
        raise ValueError(
            f"content exceeds maximum length {max_length} (got {len(value)} chars). "
            f"Split the content into multiple records, each focused on one topic. "
            f"Large imports should use the miner (mempalace mine) instead of kg_declare_entity(kind='record')."
        )
    if "\x00" in value:
        raise ValueError(
            "content contains null bytes (check source data for stray \\x00). "
            "Strip with `value.replace('\\x00', '')` before passing."
        )
    return value


DEFAULT_PALACE_PATH = os.path.expanduser("~/.mempalace/palace")
DEFAULT_COLLECTION_NAME = "mempalace_records"

VALID_CONTENT_TYPES = {"fact", "event", "discovery", "preference", "advice", "diary"}


class MempalaceConfig:
    """Configuration manager for MemPalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        """Initialize config.

        Args:
            config_dir: Override config directory (useful for testing).
                        Defaults to ~/.mempalace.
        """
        self._config_dir = (
            Path(config_dir) if config_dir else Path(os.path.expanduser("~/.mempalace"))
        )
        self._config_file = self._config_dir / "config.json"
        self._people_map_file = self._config_dir / "people_map.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        """Path to the memory palace data directory.

        Resolution order (highest-priority first):
          1. ``MEMPALACE_PALACE_PATH`` / ``MEMPAL_PALACE_PATH`` env var.
             Set by the MCP server at startup when invoked with
             ``--palace``; also honored for direct CLI use.
          2. ``~/.mempalace/hook_state/active_palace.txt``: a hint file
             written by the MCP server at startup that the hook
             subprocess can read even though the hook doesn't inherit
             the server's environment. This closes the 2026-04-20
             live bug where ``python -m mempalace hook run`` resolved
             to the empty default palace, making retrieval silently
             return zero hits against the wrong store.
          3. ``config.json`` ``palace_path`` key.
          4. ``DEFAULT_PALACE_PATH`` (``~/.mempalace/palace``).
        """
        env_val = os.environ.get("MEMPALACE_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            return env_val
        # Hook-subprocess fallback: read the hint file the MCP server
        # writes on startup. Per-host single-active-palace assumption
        # (if multi-palace support is needed later, switch to a
        # session-id-scoped hint file).
        hint = self._config_dir / "hook_state" / "active_palace.txt"
        if hint.is_file():
            try:
                p = hint.read_text(encoding="utf-8").strip()
                if p:
                    return p
            except OSError:
                # Fail-forward: the file exists but is unreadable. Note
                # we do NOT record a hook error here because this code
                # is in Config, shared by MCP server + hook + CLI.
                pass
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

    @property
    def collection_name(self):
        """ChromaDB collection name."""
        return self._file_config.get("collection_name", DEFAULT_COLLECTION_NAME)

    @property
    def people_map(self):
        """Mapping of name variants to canonical names."""
        if self._people_map_file.exists():
            try:
                with open(self._people_map_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._file_config.get("people_map", {})

    @property
    def link_author(self):
        """Link-author pipeline configuration.

        Per docs/link_author_plan.md §5.5 — all values overrideable via
        ``config.json`` under a ``"link_author"`` key. Defaults are sane
        for the out-of-the-box experience; tuning happens after the
        first few runs produce real telemetry on candidate depth +
        jury-verdict distribution (see plan §9 post-reinstall notes).

        Model IDs are verified at implementation time against the
        ``anthropic`` SDK constants — bump them in config when a new
        Opus/Haiku lands without touching code.
        """
        overrides = self._file_config.get("link_author", {}) or {}
        defaults = {
            # Anthropic API — read key from this env var; .env file at
            # <palace>/.env is loaded on startup so a dedicated mempalace
            # key lives separately from your shell environment.
            "api_key_env": "ANTHROPIC_API_KEY",
            "dotenv_path": None,  # None → <palace_path>/.env
            # Pipeline model IDs. Verify against the anthropic SDK's
            # model list at implementation time; overrideable per-palace.
            "jury_design_model": "claude-opus-4-5",
            "jury_execution_model": "claude-haiku-4-5",
            "synthesis_model": "claude-haiku-4-5",
            # Per-stage token caps.
            "design_max_tokens": 1024,
            "juror_max_tokens": 512,
            "synthesis_max_tokens": 512,
            # Cost optimisation — batch design calls across similar
            # candidates whose domain-hint embeddings have cosine >= 0.9.
            "batch_design_by_domain_similarity": True,
            "batch_domain_cosine_threshold": 0.9,
            # Dispatch cadence from finalize.
            "interval_hours": 1,
            # Candidate filtering.
            "threshold": 1.5,
            "max_per_run": 50,
            # Failure handling.
            "retry_uncertain_next_run": True,
            "rejection_cooldown_days": 30,
            "escalate_uncertain_to": None,  # e.g. "claude-sonnet-4-5" to enable
        }
        defaults.update(overrides)
        return defaults

    def init(self):
        """Create config directory and write default config.json if it doesn't exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # Restrict directory permissions to owner only (Unix)
        try:
            self._config_dir.chmod(0o700)
        except (OSError, NotImplementedError):
            pass  # Windows doesn't support Unix permissions
        if not self._config_file.exists():
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "collection_name": DEFAULT_COLLECTION_NAME,
            }
            with open(self._config_file, "w") as f:
                json.dump(default_config, f, indent=2)
            # Restrict config file to owner read/write only
            try:
                self._config_file.chmod(0o600)
            except (OSError, NotImplementedError):
                pass
        return self._config_file

    def save_people_map(self, people_map):
        """Write people_map.json to config directory.

        Args:
            people_map: Dict mapping name variants to canonical names.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._people_map_file, "w") as f:
            json.dump(people_map, f, indent=2)
        return self._people_map_file
