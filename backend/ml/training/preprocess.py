"""
Load raw dataset, clean it, extract features, handle class imbalance.
Run once to produce ml/data/processed/train.csv and test.csv.

By default this includes embedding-based semantic features (see
ml/embeddings/), which requires:
  1. Ollama running locally with the embedding model pulled
     (`ollama pull bge-m3:latest`)
  2. Centroids already built via `python -m ml.embeddings.build_centroids`
     (run that against the raw data BEFORE this script, so centroids exist
     when this script embeds each row)

This step embeds EVERY row in the raw dataset (not a sample like
build_centroids.py), so on a full dataset it can take a long time and
will occasionally hit transient failures (timeouts, 500s from Ollama).
To handle that:
  - Every row's outcome (embedded / cached / failed) is logged, so you
    can see progress and failures as they happen instead of only a
    periodic checkpoint line.
  - Embeddings are cached to disk per-row as they're computed
    (ml/data/processed/_embed_cache.joblib), so Ctrl+C or a crash loses
    at most the one in-flight row — re-running resumes instead of
    starting over.
  - Set LOG_LEVEL=DEBUG for a log line on every single row; default INFO
    logs failures immediately and successes every 50 rows.

Pass --no-embeddings to build the structural-features-only table instead
(useful for quick iteration without Ollama running).
Pass --fresh to ignore any cached embeddings and recompute everything.
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from ml.features.extract import extract_features, get_feature_names, STRUCTURAL_FEATURE_NAMES
from ml.embeddings.client import embed_text, EmbeddingBackendError, backend_available
from ml.embeddings.features import EMBEDDING_FEATURE_NAMES, _load_centroids, _cosine
from logger import get_logger

log = get_logger(__name__)

RAW_PATH = "ml/data/raw/CEAS_08.csv"   # adjust to your actual Kaggle file
OUT_TRAIN = "ml/data/processed/train.csv"
OUT_TEST = "ml/data/processed/test.csv"

# Resumable cache: row index -> raw embedding vector (list[float]). Similarity
# features are recomputed from this on every run, so if centroids change you
# don't need to re-embed, only re-run the (cheap) cosine step.
EMBED_CACHE_PATH = Path("ml/data/processed/_embed_cache.joblib")


def _split_sender(raw_sender: str) -> tuple[str, str]:
    if not isinstance(raw_sender, str):
        return "", ""
    match = re.match(r"^(.*?)\s*<(.+?)>\s*$", raw_sender.strip())
    if match:
        display_name, email = match.groups()
        return display_name.strip(), email.strip()
    return "", raw_sender.strip()


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["label"])

    # CEAS_08 columns: sender, receiver, date, subject, body, label, urls
    # "sender" here is the raw "Display Name <email@domain>" or bare email string —
    # display_name isn't a separate column in this dataset, so we derive both from it.
    for col in ["sender", "subject", "body"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    display_names, emails = [], []
    for raw in df["sender"]:
        name, addr = _split_sender(raw)
        display_names.append(name)
        emails.append(addr)
    df["display_name"] = display_names
    df["sender_email"] = emails

    # CEAS_08's "urls" column is a 0/1 flag (email contains a URL), not a list of links —
    # extract_features' _url_features() just wants that flag, so pass it straight through.
    df["has_url_flag"] = df["urls"].fillna(0).astype(int) if "urls" in df.columns else 0

    return df


def _load_embed_cache() -> dict:
    if EMBED_CACHE_PATH.exists():
        return joblib.load(EMBED_CACHE_PATH)
    return {}


def _save_embed_cache(cache: dict) -> None:
    EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, EMBED_CACHE_PATH)


def _get_row_embedding(idx: int, subject: str, body: str, cache: dict) -> list[float] | None:
    """Return a cached or freshly-computed embedding for one row, or None on failure."""
    if idx in cache:
        return cache[idx]

    text = f"{subject or ''} {body or ''}".strip()
    if not text:
        cache[idx] = [0.0] * 1024  # matches EMBEDDING_DIM in client.py
        return cache[idx]

    try:
        vec = embed_text(text)
    except EmbeddingBackendError as e:
        log.warning("row=%s embedding FAILED: %s", idx, e)
        return None

    cache[idx] = vec
    return vec


def build_feature_table(df: pd.DataFrame, include_embedding: bool = True, fresh: bool = False) -> pd.DataFrame:
    feature_names = get_feature_names(include_embedding=include_embedding)
    total = len(df)

    if not include_embedding:
        rows = []
        for _, row in df.iterrows():
            email = {
                "sender": row.get("sender_email", ""),
                "display_name": row.get("display_name", ""),
                "subject": row.get("subject", ""),
                "body": row.get("body", ""),
                "has_url_flag": row.get("has_url_flag", 0),
            }
            feats = extract_features(email, include_embedding=False)
            feats["label"] = int(row["label"])
            rows.append(feats)
        return pd.DataFrame(rows)[feature_names + ["label"]]

    # --- embedding path: logged, cached, resumable ---
    if fresh and EMBED_CACHE_PATH.exists():
        EMBED_CACHE_PATH.unlink()
        log.info("fresh=True: cleared embedding cache at %s", EMBED_CACHE_PATH)

    if not backend_available():
        raise RuntimeError(
            "Ollama embedding backend is not reachable at the configured OLLAMA_HOST. "
            "Start it (`ollama serve`) or re-run with --no-embeddings."
        )

    centroids = _load_centroids()
    if centroids is None:
        raise RuntimeError(
            "No embedding centroids found. Run `python -m ml.embeddings.build_centroids` "
            "before preprocessing with embeddings enabled."
        )

    cache = _load_embed_cache()
    already_cached = sum(1 for idx in df.index if idx in cache)
    log.info(
        "starting embedding pass: %d/%d rows already cached, %d remaining",
        already_cached, total, total - already_cached,
    )

    rows = []
    failed_indices = []
    start = time.time()
    embedded_this_run = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        email = {
            "sender": row.get("sender_email", ""),
            "display_name": row.get("display_name", ""),
            "subject": row.get("subject", ""),
            "body": row.get("body", ""),
            "has_url_flag": row.get("has_url_flag", 0),
        }

        was_cached = idx in cache
        vec = _get_row_embedding(idx, row.get("subject", ""), row.get("body", ""), cache)

        if vec is None:
            failed_indices.append(idx)
            # persist cache progress even on failure — earlier successes must not be lost
            _save_embed_cache(cache)
            continue

        if not was_cached:
            embedded_this_run += 1
            _save_embed_cache(cache)  # persist after every new embedding
            log.debug("row=%s embedded OK (%d chars)", idx, len(f"{email['subject']} {email['body']}"))

        # structural + embedding-similarity features for this row
        feats = extract_features(email, include_embedding=False)  # structural only here
        import numpy as np
        vec_arr = np.array(vec)
        phishing_sim = _cosine(vec_arr, centroids["phishing_centroid"])
        benign_sim = _cosine(vec_arr, centroids["benign_centroid"])
        feats["embed_phishing_similarity"] = phishing_sim
        feats["embed_benign_similarity"] = benign_sim
        feats["embed_similarity_margin"] = phishing_sim - benign_sim
        feats["label"] = int(row["label"])
        rows.append(feats)

        completed = i + 1
        if embedded_this_run and (embedded_this_run % 50 == 0):
            elapsed = time.time() - start
            rate = embedded_this_run / elapsed if elapsed > 0 else 0
            remaining = total - completed
            eta = remaining / rate if rate > 0 else float("inf")
            log.info(
                "progress: %d/%d rows (%d newly embedded this run, %d failed so far) "
                "— %.0fs elapsed, ~%.0fs remaining",
                completed, total, embedded_this_run, len(failed_indices), elapsed, eta,
            )

    if failed_indices:
        log.warning(
            "%d/%d rows failed to embed and were skipped (not included in output): %s%s",
            len(failed_indices), total, failed_indices[:20],
            " ...(truncated)" if len(failed_indices) > 20 else "",
        )
    log.info(
        "embedding pass complete: %d rows succeeded, %d failed, %d newly embedded this run",
        len(rows), len(failed_indices), embedded_this_run,
    )

    feat_df = pd.DataFrame(rows)
    return feat_df[feature_names + ["label"]]


def main(include_embedding: bool = True, fresh: bool = False):
    raw = load_raw()
    feat_df = build_feature_table(raw, include_embedding=include_embedding, fresh=fresh)

    train_df, test_df = train_test_split(
        feat_df, test_size=0.2, random_state=42, stratify=feat_df["label"]
    )

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)
    log.info("train: %d rows, test: %d rows", len(train_df), len(test_df))
    log.info("train label balance:\n%s", train_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Build structural features only, skip Ollama embedding calls (fast iteration).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any cached embeddings and recompute all of them.",
    )
    args = parser.parse_args()
    main(include_embedding=not args.no_embeddings, fresh=args.fresh)
