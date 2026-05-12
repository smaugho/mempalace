"""Embedding-function owner -- the single import site for mempalace.

Today this thinly wraps chromadb's default ONNX MiniLM-L6-v2 embedder
because that's what the existing palaces were embedded with. Future
backends (sqlite-vec + sentence-transformers, FastEmbed, etc.) plug
in here behind the same callable shape:

    embedder([text1, text2, ...]) -> [[float, float, ...], ...]

Callers never import from ``chromadb.utils.embedding_functions`` --
they import :func:`get_default_embedder` from this module so the
embedder can be swapped without touching every retrieval/search site.

Adrian directive 2026-05-11 (branch-by-abstraction, Phase 1 of the
chromadb sealing). The lift from this thin wrapper to a chroma-free
embedder is small: replace the body with a sentence-transformers or
FastEmbed call. Same input contract; same vector dimensionality
(384) when staying on MiniLM-L6-v2 so existing vectors remain
compatible.
"""

from __future__ import annotations

from typing import Any, Sequence

# Single process-wide cache. The underlying chroma DefaultEmbeddingFunction
# loads a ~79 MB ONNX model on first instantiation; we don't want every
# caller paying that. None until first call. Use get_default_embedder()
# rather than touching this directly.
_DEFAULT_EMBEDDER: Any | None = None


# Embedding-function callable shape: ``Sequence[str] -> Sequence[Sequence[float]]``.
# Kept as ``Any`` rather than a Protocol for now -- chroma's
# DefaultEmbeddingFunction inherits from chromadb's typed base and our
# Protocol would lock in their attribute names. Once we swap to
# sentence-transformers / FastEmbed we can tighten this.
Embedder = Any


def get_default_embedder() -> Embedder | None:
    """Return the process-wide default embedder, or ``None`` if the
    backing library isn't installed.

    Idempotent + cached after the first successful call. Returns
    ``None`` (rather than raising) when chromadb / its onnxruntime
    dependency isn't available -- callers should treat ``None`` as
    "embedding unavailable" and skip embedding-dependent paths
    rather than fail loud. This mirrors the historical try/except
    ImportError pattern at every call site, now centralised here.

    Future backends: when sqlite-vec ships, replace the chromadb
    import with sentence-transformers / FastEmbed. The cache + None-
    on-missing contract stays the same so callers don't change.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER
    try:
        from chromadb.utils import embedding_functions as ef  # noqa: PLC0415
    except ImportError:
        return None
    try:
        _DEFAULT_EMBEDDER = ef.DefaultEmbeddingFunction()
    except Exception:
        # Some chromadb installs ship without onnxruntime; degrade
        # to None rather than crash retrieval globally.
        return None
    return _DEFAULT_EMBEDDER


def embed(texts: Sequence[str]) -> list[list[float]] | None:
    """Convenience: embed a batch of strings, returning a list of
    vectors or ``None`` if no embedder is available. Callers that
    already hold the embedder object via :func:`get_default_embedder`
    can call it directly; this helper is for one-shot sites that
    don't want to keep a reference."""
    embedder = get_default_embedder()
    if embedder is None:
        return None
    try:
        return list(embedder(list(texts)))
    except Exception:
        return None


__all__ = ["Embedder", "get_default_embedder", "embed"]
