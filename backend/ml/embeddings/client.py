"""
Thin client for a local Ollama embedding model (default: bge-m3).

Kept separate from feature extraction so the embedding backend (model
name, host, batching strategy) can change without touching extract.py.

Env vars:
    OLLAMA_HOST           default "http://localhost:11434"
    OLLAMA_EMBED_MODEL    default "bge-m3:latest"
"""
from __future__ import annotations

import os
from functools import lru_cache

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest")

# bge-m3 produces 1024-dim embeddings. If you swap models, update this —
# it's used to size the zero-vector fallback consistently.
EMBEDDING_DIM = 1024


class EmbeddingBackendError(RuntimeError):
    """Raised when Ollama is unreachable or returns something unexpected."""


def embed_text(text: str, timeout: float = 30.0) -> list[float]:
    """
    Return a single embedding vector for `text` via Ollama's /api/embeddings.
    Raises EmbeddingBackendError on any failure — callers decide whether to
    fall back (see ml.embeddings.features for the fallback policy used by
    Layer 1 features).
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise EmbeddingBackendError(f"Ollama request failed: {e}") from e

    embedding = data.get("embedding")
    if not embedding or not isinstance(embedding, list):
        raise EmbeddingBackendError(f"Unexpected Ollama response shape: {data!r}")

    return embedding


@lru_cache(maxsize=1)
def _check_backend_once() -> bool:
    """Best-effort reachability check, cached for the process lifetime."""
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0).raise_for_status()
        return True
    except requests.RequestException:
        return False


def backend_available() -> bool:
    return _check_backend_once()
