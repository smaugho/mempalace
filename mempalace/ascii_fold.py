"""ASCII transliteration helpers for metadata fields.

Adrian's design lock 2026-04-27: every metadata field that flows into
embeddings, ids, or display labels — entity names, summary
``{what, why, scope}``, context ``queries`` / ``keywords`` — is silently
folded to ASCII at the validation boundary. The long-form ``content``
column on records (``entities.description`` today, ``entities.content``
post-rename) stays UTF-8 verbatim because preserving the verbatim text
of the source memory is the whole point of that field.

Library choice: ``anyascii`` (ISC license, pure-Python). Picked over
the more famous ``unidecode`` because mempalace ships under MIT and
``unidecode``'s GPL would force the whole package to GPL. ``anyascii``
covers the same Unicode → ASCII transliteration surface (Latin diacritics,
CJK pinyin/romaji, Cyrillic, Greek, Arabic, Hebrew, plus full punctuation)
without the license tax.

This module is the single import point for the fold. Validators hook it
in once each:

- ``knowledge_graph.coerce_summary_for_persist`` — folds ``what`` /
  ``why`` / ``scope`` after the dict-shape and length checks pass, so
  error messages still show the writer's original characters.
- ``knowledge_graph.normalize_entity_name`` — folds the input string
  before the CamelCase split + ``[^a-z0-9]+`` collapse, so "café" lands
  as the stable id ``cafe`` instead of the lossy ``caf``.
- ``knowledge_graph.add_entity`` — folds the raw ``name`` parameter
  before binding it to ``INSERT INTO entities (..., name, ...)`` so the
  display column matches the id family.
- ``scoring.validate_context`` — folds ``queries`` and ``keywords``
  list entries after ``_validate_string_list`` returns the cleaned
  list. Same chokepoint serves WRITE tools (declare_intent,
  declare_operation, kg_declare_entity, kg_add) and READ tools
  (kg_search, kg_query) — fold-on-read lets users still type "café"
  in a query and match a folded entity ``cafe``.

The fold is silent (no warning, no return-flag) by user request. If you
need to detect whether a string was non-ASCII before fold, use
``contains_non_ascii(s)`` BEFORE calling ``fold_ascii(s)`` — the helpers
are pure and side-effect-free.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

try:
    from anyascii import anyascii as _any_to_ascii
except ImportError as exc:  # pragma: no cover - dependency required
    raise ImportError(
        "mempalace requires the 'anyascii' package for metadata-field "
        "ASCII transliteration. Install it with `pip install anyascii` "
        "or reinstall mempalace which pins it as a runtime dependency."
    ) from exc


# Lone UTF-16 surrogate halves (U+D800-U+DFFF). These are not real
# characters — they're the high/low pair codepoints used inside UTF-16
# byte sequences. When one half ends up in a Python string without its
# matching pair (Windows cp1252 transcoding, surrogateescape error mode,
# truncated MCP transport, etc.), it's not encodable to UTF-8 and crashes
# json.dumps + Chroma's tokenizer + Anthropic's HTTP client at the very
# bottom of the stack with errors like
# "'utf-8' codec can't encode character '\\udc9d' in position 50:
#  surrogates not allowed". We strip them at fold time so those failure
# modes can't surface from any caller's metadata input.
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


__all__ = [
    "fold_ascii",
    "fold_string_list",
    "fold_summary",
    "contains_non_ascii",
    "is_ascii",
]


def is_ascii(value: Any) -> bool:
    """True iff ``value`` is a string containing only 7-bit ASCII codepoints.

    Non-strings return ``False`` (defensive default — callers passing
    ``None`` or a number should hit the explicit type checks earlier in
    the pipeline, not silently slip through this helper).
    """
    return isinstance(value, str) and value.isascii()


def contains_non_ascii(value: Any) -> bool:
    """True iff ``value`` is a string with at least one non-ASCII codepoint.

    Convenience inverse of :func:`is_ascii` for the common
    "did the writer pass unicode?" branch in tests / cleanup scripts.
    """
    return isinstance(value, str) and not value.isascii()


def fold_ascii(value: Any) -> str:
    """Transliterate ``value`` to 7-bit ASCII via ``anyascii``.

    - ``str`` inputs run through ``anyascii`` and are returned as a new
      string. Already-ASCII strings round-trip unchanged.
    - Non-strings are coerced via ``str()`` first (covers numeric / bool
      summary fields that some callers serialize loosely).
    - ``None`` returns the empty string so callers can chain
      ``fold_ascii(d.get("scope"))`` without a guard.

    The fold is idempotent: ``fold_ascii(fold_ascii(s)) == fold_ascii(s)``
    for every input. ``anyascii`` itself is deterministic and stateless,
    so this helper is safe to call from validation hot paths.

    Examples
    --------
    >>> fold_ascii("Café")
    'Cafe'
    >>> fold_ascii("résumé — naïve")
    'resume -- naive'
    >>> fold_ascii("北京")
    'Bei Jing '
    >>> fold_ascii("Москва")
    'Moskva'
    >>> fold_ascii(None)
    ''
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    # Strip lone UTF-16 surrogates first — they're not encodable to UTF-8
    # and would crash downstream embed / JSON / HTTP write paths even
    # AFTER an anyascii pass. The strip is unconditional because the fast
    # `isascii()` path can return True for a string that nonetheless
    # contains a surrogate (isascii is False on surrogates, but the
    # strip is cheap and the regex short-circuits when there's no match).
    if _LONE_SURROGATE_RE.search(value):
        value = _LONE_SURROGATE_RE.sub("", value)
    if value.isascii():
        return value
    return _any_to_ascii(value)


def fold_string_list(values: Iterable[Any]) -> list[str]:
    """Apply :func:`fold_ascii` element-wise; preserve list order.

    Non-string entries are coerced via ``fold_ascii``'s ``str()``
    fallback. ``None`` and empty entries are kept as empty strings —
    structural validation upstream is responsible for the min-length /
    presence checks, this helper only folds.
    """
    return [fold_ascii(v) for v in values]


def fold_summary(summary: dict) -> dict:
    """Fold the ``{what, why, scope?}`` triple in a validated summary dict.

    Returns a NEW dict; does not mutate the caller's. Intended to run
    after :func:`mempalace.knowledge_graph.validate_summary` has already
    checked dict shape, field presence, and length floors — so the
    caller has a normalized ``{what, why}`` plus optional ``scope``
    when it lands here.

    Non-dict inputs are returned unchanged so the helper composes
    cleanly with legacy-read paths that still tolerate prose strings
    (``serialize_summary_for_embedding``); validation rejects strings
    on writes before they reach the fold step.
    """
    if not isinstance(summary, dict):
        return summary
    out = dict(summary)
    # Fold ONLY string-typed values. Non-string values (None, ints,
    # missing keys) pass through unchanged so validate_summary still
    # raises its "must be a string" / "missing 'what'" / "missing 'why'"
    # errors against the original types — the fold is a transliteration
    # step, not a type coercion.
    for key in ("what", "why", "scope"):
        val = out.get(key)
        if isinstance(val, str):
            out[key] = fold_ascii(val)
    return out
