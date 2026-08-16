"""
Thin client for a text embedding model, via any OpenAI-compatible API
(bukan cuma OpenAI resmi — bisa diarahkan ke provider lain lewat EMBED_BASE_URL).

Kept separate from feature extraction so the embedding backend (model
name, host, batching strategy) can change without touching extract.py.

Env vars:
    EMBED_BASE_URL    default "https://api.openai.com/v1"
    EMBED_API_KEY     API key provider
    EMBED_MODEL       default "text-embedding-3-small"
"""
from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI, OpenAIError

EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# text-embedding-3-small produces 1536-dim embeddings. If you swap models,
# update this — it's used to size the zero-vector fallback consistently.
EMBEDDING_DIM = 1536

_client: OpenAI | None = None


class EmbeddingBackendError(RuntimeError):
    """Raised when the embedding backend is unreachable or returns something unexpected."""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=EMBED_BASE_URL, api_key=os.environ.get("EMBED_API_KEY", ""))
    return _client


def embed_text(text: str, timeout: float = 30.0) -> list[float]:
    """
    Return a single embedding vector for `text` via the /embeddings endpoint.
    Raises EmbeddingBackendError on any failure — callers decide whether to
    fall back (see ml.embeddings.features for the fallback policy used by
    Layer 1 features).
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    try:
        resp = _get_client().embeddings.create(
            model=EMBED_MODEL,
            input=text,
            timeout=timeout,
        )
    except OpenAIError as e:
        raise EmbeddingBackendError(f"Embedding request failed: {e}") from e

    if not resp.data or not resp.data[0].embedding:
        raise EmbeddingBackendError(f"Unexpected embedding response shape: {resp!r}")

    return resp.data[0].embedding


@lru_cache(maxsize=1)
def _check_backend_once() -> bool:
    """Best-effort reachability check, cached for the process lifetime."""
    try:
        _get_client().models.list()
        return True
    except OpenAIError:
        return False


def backend_available() -> bool:
    return _check_backend_once()
