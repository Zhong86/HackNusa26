"""
Compute the phishing/benign embedding centroids used by
ml/embeddings/features.py, from the *raw* labeled dataset (not the
already-featurized train.csv, since we need subject/body text here,
not the numeric feature table).

Run this once after preprocessing raw data, and again any time the
training set changes meaningfully (e.g. after retrain.py folds in new
agent-caught examples), so the centroids stay representative.

Resumable: each embedded vector is appended to a per-class cache file
as soon as it's computed. If you Ctrl+C partway through, re-running
this script picks up where it left off instead of starting over — it
only calls Ollama for rows it hasn't already cached.

Usage:
    python -m ml.embeddings.build_centroids
    python -m ml.embeddings.build_centroids --fresh   # ignore cache, start over
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ml.embeddings.client import embed_text, backend_available, EmbeddingBackendError
from ml.training.preprocess import load_raw

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_PATH = MODELS_DIR / "embedding_centroids.joblib"

# Per-class incremental caches (row_index -> vector), written after every
# successful embedding call so progress survives Ctrl+C / crashes.
CACHE_DIR = MODELS_DIR / "_embedding_cache"
PHISHING_CACHE = CACHE_DIR / "phishing_vecs.joblib"
BENIGN_CACHE = CACHE_DIR / "benign_vecs.joblib"

# Cap how many rows per class we embed — centroids stabilize well before
# using the full dataset, and this keeps the one-time build fast. At ~0.5-2s
# per embedding call (sequential, local Ollama), 1500/class can take 25-60+
# minutes; 300-500/class is usually enough for a stable centroid and finishes
# in a few minutes.
MAX_PER_CLASS = 1200


def _load_cache(path: Path) -> dict[int, list[float]]:
    if path.exists():
        return joblib.load(path)
    return {}


def _save_cache(path: Path, cache: dict[int, list[float]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, path)


def _embed_rows(rows: pd.DataFrame, cache_path: Path, label: str) -> np.ndarray:
    """
    rows: dataframe slice to embed, indexed by original dataframe index.
    cache_path: where to persist {row_index: vector} incrementally.
    Returns the array of vectors in `rows` order, using cached values
    where available and only calling Ollama for missing ones.
    """
    cache = _load_cache(cache_path)
    already_done = sum(1 for idx in rows.index if idx in cache)
    total = len(rows)

    if already_done:
        print(f"  [{label}] resuming: {already_done}/{total} already cached")

    start = time.time()
    done_this_run = 0

    for i, (idx, row) in enumerate(rows.iterrows()):
        if idx in cache:
            continue  # already embedded in a previous run

        text = f"{row.get('subject', '')} {row.get('body', '')}".strip()
        try:
            cache[idx] = embed_text(text)
            done_this_run += 1
            print(f" [{idx}] embed success")
        except EmbeddingBackendError as e:
            print(f"  [{label}] warning: skipping row {idx}, embedding failed: {e}")
            continue
        finally:
            # save after every row — if interrupted, at most one in-flight
            # call is lost, never previously-completed work.
            _save_cache(cache_path, cache)

        completed = sum(1 for r_idx in rows.index[: i + 1] if r_idx in cache)
        if completed % 25 == 0 or completed == total:
            elapsed = time.time() - start
            rate = done_this_run / elapsed if elapsed > 0 and done_this_run else 0
            remaining_rows = total - completed
            eta = remaining_rows / rate if rate > 0 else float("inf")
            print(
                f"  [{label}] {completed}/{total} embedded "
                f"({elapsed:.0f}s this run, ~{eta:.0f}s remaining)"
            )

    _save_cache(cache_path, cache)  # final flush
    return np.array([cache[idx] for idx in rows.index if idx in cache])


def build_centroids(max_per_class: int = MAX_PER_CLASS, fresh: bool = False) -> dict:
    if not backend_available():
        raise RuntimeError(
            "Ollama embedding backend is not reachable. Start it (`ollama serve`) "
            "and make sure the model is pulled (`ollama pull bge-m3:latest`) before "
            "building centroids."
        )

    if fresh:
        for p in (PHISHING_CACHE, BENIGN_CACHE):
            if p.exists():
                p.unlink()
        print("--fresh: cleared existing embedding cache, starting over")

    df = load_raw()

    phishing_rows = df[df["label"] == 1].sample(
        n=min(max_per_class, (df["label"] == 1).sum()), random_state=42
    )
    benign_rows = df[df["label"] == 0].sample(
        n=min(max_per_class, (df["label"] == 0).sum()), random_state=42
    )

    print(f"embedding up to {len(phishing_rows)} phishing examples...")
    phishing_vecs = _embed_rows(phishing_rows, PHISHING_CACHE, "phishing")
    print(f"embedding up to {len(benign_rows)} benign examples...")
    benign_vecs = _embed_rows(benign_rows, BENIGN_CACHE, "benign")

    centroids = {
        "phishing_centroid": phishing_vecs.mean(axis=0),
        "benign_centroid": benign_vecs.mean(axis=0),
        "n_phishing": len(phishing_vecs),
        "n_benign": len(benign_vecs),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(centroids, OUT_PATH)
    print(f"centroids saved to {OUT_PATH} (phishing n={len(phishing_vecs)}, benign n={len(benign_vecs)})")

    # Cache files are intentionally left in place after a successful build —
    # re-running build_centroids (e.g. after retrain.py adds new examples)
    # will reuse them for any row index that repeats, and only embed new ones.
    return centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any cached embeddings and start over from scratch.",
    )
    args = parser.parse_args()
    build_centroids(fresh=args.fresh)
