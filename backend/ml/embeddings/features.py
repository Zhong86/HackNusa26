"""
Turn a raw embedding vector into a small number of numeric features
suitable for the existing tabular XGBoost model.

Design choice: we do NOT feed the raw 1024-dim embedding into XGBoost.
With a training set of a few thousand rows, 1024 extra columns would
swamp the 8 structural features and invite overfitting. Instead we
compute cosine similarity against a handful of reference centroids
(a "phishing" centroid and a "benign" centroid, built once from labeled
training data) and use those similarity scores as features. This keeps
the semantic signal dense and interpretable: "how semantically close is
this email to known phishing" is one clean number.

Centroids are computed offline by ml/embeddings/build_centroids.py and
cached to ml/models/embedding_centroids.joblib. If that file doesn't
exist yet (e.g. first run before any training has happened) or the
Ollama backend is unreachable, embedding features degrade to neutral
zeros rather than raising — Layer 1 should never go down because a
local embedding model isn't running.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np

from ml.embeddings.client import embed_text, EmbeddingBackendError, backend_available

log = logging.getLogger(__name__)

CENTROIDS_PATH = Path(__file__).resolve().parents[1] / "models" / "embedding_centroids.joblib"

# Raw-vector cache, keyed by a hash of the (subject, body) text — not by
# row index, since callers here (preprocess.py per-row, classifier.py
# per-request) have no shared/stable index. Content-hash keying means
# identical emails (duplicate rows, or repeated scoring during iteration)
# never re-hit Ollama, and — like build_centroids.py's per-row cache —
# we persist after every single embedding call so an interrupted
# preprocess.py run resumes instead of re-embedding rows already paid for.
CACHE_PATH = Path(__file__).resolve().parents[1] / "models" / "_embedding_cache" / "vector_cache.joblib"

EMBEDDING_FEATURE_NAMES = [
    "embed_phishing_similarity",
    "embed_benign_similarity",
    "embed_similarity_margin",  # phishing_sim - benign_sim, the useful discriminator
]

_centroids_cache = None
_vector_cache: dict[str, list[float]] | None = None
_warned_unavailable = [False]  # list as a mutable module-level flag


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_vector_cache() -> dict[str, list[float]]:
    global _vector_cache
    if _vector_cache is None:
        _vector_cache = joblib.load(CACHE_PATH) if CACHE_PATH.exists() else {}
    return _vector_cache


def _save_vector_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_vector_cache, CACHE_PATH)


def _cached_embed_text(text: str) -> list[float]:
    """embed_text(), but checks/writes the on-disk cache first."""
    cache = _load_vector_cache()
    key = _text_key(text)
    if key in cache:
        return cache[key]

    vec = embed_text(text)
    cache[key] = vec
    _save_vector_cache()  # flush immediately — resumable across interrupts
    return vec


def _load_centroids():
    global _centroids_cache
    if _centroids_cache is not None:
        return _centroids_cache

    if not CENTROIDS_PATH.exists():
        log.warning(
            "No embedding centroids found at %s — embedding features will be neutral (0.0) "
            "until ml/embeddings/build_centroids.py has been run.",
            CENTROIDS_PATH,
        )
        _centroids_cache = None
        return None

    _centroids_cache = joblib.load(CENTROIDS_PATH)
    return _centroids_cache


def embedding_features(subject: str, body: str) -> dict:
    """
    Returns EMBEDDING_FEATURE_NAMES -> float. Never raises — falls back to
    neutral zeros if the embedding backend or centroids aren't available,
    so a local Ollama outage degrades Layer 1 rather than breaking it.
    """
    neutral = {name: 0.0 for name in EMBEDDING_FEATURE_NAMES}

    centroids = _load_centroids()
    if centroids is None:
        return neutral

    text = f"{subject or ''} {body or ''}".strip()

    key = _text_key(text)
    cache = _load_vector_cache()
    if key not in cache and not backend_available():
        if not _warned_unavailable[0]:
            log.warning(
                "Ollama embedding backend unreachable — using neutral embedding features "
                "(further occurrences this run are logged at DEBUG level)"
            )
            _warned_unavailable[0] = True
        else:
            log.debug("Ollama embedding backend unreachable — using neutral embedding features")
        return neutral

    try:
        vec = np.array(_cached_embed_text(text))
    except EmbeddingBackendError as e:
        log.warning("Embedding call failed, using neutral features: %s", e)
        return neutral

    phishing_sim = _cosine(vec, centroids["phishing_centroid"])
    benign_sim = _cosine(vec, centroids["benign_centroid"])

    return {
        "embed_phishing_similarity": phishing_sim,
        "embed_benign_similarity": benign_sim,
        "embed_similarity_margin": phishing_sim - benign_sim,
    }
