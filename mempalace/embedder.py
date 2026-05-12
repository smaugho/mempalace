"""Embedding-function owner -- the single import site for mempalace.

Wraps ``fastembed.TextEmbedding`` (the Qdrant project's lightweight
ONNX-only embedder) under the same model that chromadb used to ship:
``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, L2-normalised
cosine vectors).

Why fastembed (Adrian directive 2026-05-11, Phase 1 follow-on of the
chromadb sealing): we needed an embedder that doesn't require chromadb
to be installed, so the long-term path is to ``pip uninstall chromadb``
once SqliteVecVectorStore (Phase 3) lands. fastembed is the right fit
because:

  * Pure-ONNX runtime (no PyTorch, no GPU drivers). ~50 MB total dep
    surface vs sentence-transformers' ~2 GB PyTorch tower.
  * Same exact MiniLM-L6-v2 ONNX model that chromadb used internally
    via ``onnxruntime``. Verified empirically 2026-05-11: same text
    through chromadb's DefaultEmbeddingFunction and through fastembed
    produces cosine similarity 1.000000 -- vectors are IDENTICAL, so
    existing palaces are drop-in compatible.
  * Available as a pre-built wheel on win_amd64 (works under Windows
    on ARM via x64 emulation on the project's primary dev box).

Callers never import from ``chromadb.utils.embedding_functions``;
they call :func:`get_default_embedder` from this module. The
embedder object is callable -- ``embedder(["text1", "text2"])`` returns
a ``list[list[float]]``, matching chromadb's API so call sites that
used to take a chromadb embedding function don't need rewriting.

Returns ``None`` (rather than raising) when fastembed isn't installed
or fails to load. Callers should treat ``None`` as "embedding
unavailable" and skip embedding-dependent paths -- this mirrors the
historical try/except ImportError pattern, now centralised.
"""

from __future__ import annotations

from typing import Sequence

# Default model -- same one chromadb shipped, so existing palace
# vectors stay cosine-compatible. fastembed downloads + caches via
# the HuggingFace hub on first instantiation.
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Process-wide cache of the wrapped embedder. The underlying ONNX
# model load takes ~3 seconds (cold) / ~1 second (warm cache) on the
# primary dev box; we don't want every caller paying that. None until
# the first successful get_default_embedder() call.
_DEFAULT_EMBEDDER: "Embedder | None" = None
_INIT_TRIED: bool = False  # prevents repeated import-time retries


class Embedder:
    """Callable wrapper over fastembed's ``TextEmbedding`` with the
    chromadb-compatible call signature (``embedder(texts: list[str])
    -> list[list[float]]``).

    Construction is lazy + cached at module scope -- call
    :func:`get_default_embedder` rather than constructing this
    directly. The instance is safe to share across threads;
    fastembed's underlying ``onnxruntime.InferenceSession`` is
    thread-safe per ONNX Runtime docs.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME):
        # Late import so a missing fastembed wheel doesn't break
        # ``import mempalace.embedder`` -- callers will just get
        # ``None`` back from get_default_embedder() and fail open.
        from fastembed import TextEmbedding  # noqa: PLC0415

        self._model = TextEmbedding(model_name=model_name)
        self.model_name = model_name

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        # fastembed returns a generator of numpy arrays. Materialise
        # eagerly and coerce to plain Python lists so the return shape
        # matches the chromadb embedding-function contract (call sites
        # serialize via json.dumps and compare element-wise).
        return [list(float(x) for x in vec) for vec in self._model.embed(list(texts))]


def get_default_embedder() -> Embedder | None:
    """Return the process-wide default embedder, or ``None`` if the
    backing library isn't installed / fails to load.

    Idempotent + cached after the first successful call. Failure is
    sticky: once ``None``, we don't retry every call (which would
    keep re-paying the ImportError cost) -- callers should treat
    None as a stable signal that embedding is unavailable this
    process and degrade accordingly.

    Cosine-compatibility contract: the model resolved here MUST
    produce vectors compatible with existing palace data. Today
    that means MiniLM-L6-v2 (384-dim). Changing the default model
    is a corpus-rewrite migration, not a code change.
    """
    global _DEFAULT_EMBEDDER, _INIT_TRIED
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER
    if _INIT_TRIED:
        return None
    _INIT_TRIED = True
    try:
        _DEFAULT_EMBEDDER = Embedder()
    except ImportError:
        # fastembed not installed in this environment. Callers see
        # None and skip embedding-dependent paths.
        return None
    except Exception:
        # Model download failure, ONNX runtime issue, etc. Degrade
        # rather than crash retrieval globally.
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
        return embedder(list(texts))
    except Exception:
        return None


__all__ = ["Embedder", "get_default_embedder", "embed"]
