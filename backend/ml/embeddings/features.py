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

import logging
from pathlib import Path

import joblib
import numpy as np

from ml.embeddings.client import embed_text, EmbeddingBackendError, backend_available

log = logging.getLogger(__name__)

CENTROIDS_PATH = Path(__file__).resolve().parents[1] / "models" / "embedding_centroids.joblib"

EMBEDDING_FEATURE_NAMES = [
    "embed_phishing_similarity",
    "embed_benign_similarity",
    "embed_similarity_margin",  # phishing_sim - benign_sim, the useful discriminator
]

_centroids_cache = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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

    if not backend_available():
        log.warning("Ollama embedding backend unreachable — using neutral embedding features")
        return neutral

    text = f"{subject or ''} {body or ''}".strip()
    try:
        vec = np.array(embed_text(text))
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