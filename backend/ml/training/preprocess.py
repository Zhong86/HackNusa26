"""
Load raw dataset, clean it, extract features, handle class imbalance.
Run once to produce ml/data/processed/train.csv and test.csv.

For quick local testing, cap the number of raw rows processed with
--sample-size (CLI) or the SAMPLE_SIZE env var — feature extraction is
the slow part (~2s/row, more when Ollama embeddings are involved), so
capping upstream keeps a smoke-test run fast. Sampling is stratified by
label so both classes still show up even in a small sample.

Row-level feature caching: extracted features for each raw row are
cached to ml/data/_feature_cache/row_features.joblib, keyed by the raw
dataset's original row index (see load_raw — the index is intentionally
NOT reset after sampling, so it means the same physical row across runs).
This means running --sample-size 300 and then --sample-size 600 only
computes features (and calls Ollama) for the ~300 *new* rows the bigger
sample pulls in, not all 600 again. Written per-row, like
ml/embeddings/build_centroids.py's cache, so a Ctrl+C or a failed run
doesn't lose already-completed work. Use --fresh to ignore the cache and
recompute everything (e.g. if Ollama was down for a prior run and you
want to redo it with real embeddings instead of neutral fallbacks).

Examples:
    python -m ml.training.preprocess --sample-size 200
    SAMPLE_SIZE=200 python -m ml.training.preprocess
    python -m ml.training.preprocess --sample-size 600 --fresh
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import re
import joblib
from ml.features.extract import extract_features, FEATURE_NAMES
from logger import get_logger

log = get_logger(__name__)

RAW_PATH = "ml/data/raw/CEAS_08.csv"   # adjust to your actual Kaggle file
OUT_TRAIN = "ml/data/processed/train.csv"
OUT_TEST = "ml/data/processed/test.csv"

# Row-level feature cache: {raw_row_index: feature_dict (incl. "label")}.
# Keyed on the ORIGINAL raw dataset index, not a per-run position — see
# load_raw()'s sampling step for why that distinction matters.
CACHE_DIR = Path("ml/data/_feature_cache")
FEATURE_CACHE_PATH = CACHE_DIR / "row_features.joblib"

# Optional cap on total raw rows processed. None/0 = use the full dataset.
# Can also be set via env var SAMPLE_SIZE or the --sample-size CLI flag.
SAMPLE_SIZE = int(os.environ["SAMPLE_SIZE"]) if os.environ.get("SAMPLE_SIZE") else None

# CEAS_08's "urls" column is just a 0/1 flag (does the body contain a link),
# not the actual link text — the real URLs live inline in the raw body, e.g.
# "...Become a lover no woman will be able to resist!\nhttp://whitedone.com/...".
# We regex them out here so training sees the same real url list[str] shape
# that classifier.py already gets from EmailPayload.urls at serving time —
# no train/serve skew, and _url_features() gets real data to work with
# instead of a single flag bit.
_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')


def _extract_urls(body: str) -> list[str]:
    if not isinstance(body, str) or not body:
        return []
    return [u.rstrip('.,;:!?') for u in _URL_RE.findall(body)]


def _split_sender(raw_sender: str) -> tuple[str, str]:
    if not isinstance(raw_sender, str):
        return "", ""
    match = re.match(r"^(.*?)\s*<(.+?)>\s*$", raw_sender.strip())
    if match:
        display_name, email = match.groups()
        return display_name.strip(), email.strip()
    return "", raw_sender.strip()


def load_raw(sample_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["label"])

    # CEAS_08 columns: sender, receiver, date, subject, body, label, urls
    # "sender" here is the raw "Display Name <email@domain>" or bare email string —
    # display_name isn't a separate column in this dataset, so we derive both from it.
    for col in ["sender", "subject", "body"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Cap row count BEFORE the (slow) feature extraction step. Stratified by
    # label so a small sample still has both phishing and safe examples.
    if sample_size:
        n_classes = df["label"].nunique()
        if sample_size < n_classes:
            raise ValueError(
                f"sample_size={sample_size} is smaller than the number of "
                f"label classes ({n_classes}); pick a larger sample size."
            )
        if sample_size < len(df):
            df, _ = train_test_split(
                df,
                train_size=sample_size,
                random_state=42,
                stratify=df["label"],
            )
            # NOTE: intentionally NOT calling reset_index(drop=True) here.
            # The row-level feature cache in build_feature_table() is keyed
            # on this index — it has to keep meaning "raw row N" regardless
            # of --sample-size, so a bigger run can reuse features a smaller
            # run already computed for the same physical rows. Resetting to
            # 0..len(df) would renumber rows per-run and defeat the cache
            # (worse: it would silently attach the WRONG cached features to
            # rows if two runs disagreed on what index N pointed to).

    display_names, emails = [], []
    for raw in df["sender"]:
        name, addr = _split_sender(raw)
        display_names.append(name)
        emails.append(addr)
    df["display_name"] = display_names
    df["sender_email"] = emails

    # Real URL extraction from body text — see _extract_urls() docstring above.
    # Stored as a list per row; build_feature_table() passes it straight to
    # extract_features() as email["urls"].
    df["extracted_urls"] = df["body"].apply(_extract_urls)

    return df


def _load_feature_cache() -> dict:
    if FEATURE_CACHE_PATH.exists():
        return joblib.load(FEATURE_CACHE_PATH)
    return {}


def _save_feature_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, FEATURE_CACHE_PATH)


def build_feature_table(df: pd.DataFrame, fresh: bool = False) -> pd.DataFrame:
    """
    df must retain its original raw-dataset index (see load_raw) — that
    index is the cache key. Rows already in the cache skip extract_features()
    entirely (no recompute, no Ollama call); new rows are computed and
    written to the cache immediately (write-through, one row at a time, so
    interrupting a run never loses previously-completed rows).

    fresh=True ignores any existing cache and recomputes every row (use
    this if a prior run cached bad data, e.g. Ollama was down and you now
    want real embeddings instead of neutral fallbacks for those rows).
    """
    total = len(df)

    cache = {} if fresh else _load_feature_cache()
    if fresh and FEATURE_CACHE_PATH.exists():
        log.info("--fresh: ignoring existing feature cache, recomputing all rows")

    already_cached = sum(1 for idx in df.index if idx in cache)
    if already_cached:
        log.info(
            "Feature cache hit: %d/%d rows already computed, skipping those",
            already_cached, total,
        )

    log.info("Feature extraction starting: %d rows (%d cached, %d to compute)",
              total, already_cached, total - already_cached)

    n_success = 0
    n_failed = 0
    n_from_cache = 0

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        if idx in cache:
            n_from_cache += 1
        else:
            email = {
                "sender": row.get("sender_email", ""),
                "display_name": row.get("display_name", ""),
                "subject": row.get("subject", ""),
                "body": row.get("body", ""),
                "urls": row.get("extracted_urls", []),
            }
            try:
                feats = extract_features(email)
                feats["label"] = int(row["label"])
                cache[idx] = feats
                n_success += 1
            except Exception:
                n_failed += 1
                log.exception(
                    "Feature extraction failed for row %d (raw_idx=%s, sender=%s) — skipping",
                    i, idx, email.get("sender"),
                )
                continue
            finally:
                # Write through after every newly-computed row — at most one
                # in-flight row is lost on interruption, never prior work.
                _save_feature_cache(cache)

        if i % 50 == 0 or i == total:
            log.info(
                "Feature extraction progress: %d/%d done (%d from cache, %d newly computed, %d failed)",
                i, total, n_from_cache, n_success, n_failed,
            )

    log.info(
        "Feature extraction finished: %d/%d in table (%d from cache, %d newly computed, %d failed)",
        n_from_cache + n_success, total, n_from_cache, n_success, n_failed,
    )

    rows = [cache[idx] for idx in df.index if idx in cache]
    feat_df = pd.DataFrame(rows)
    return feat_df[FEATURE_NAMES + ["label"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Cap on total raw rows processed (stratified by label). "
             "Omit or pass 0 to use the full dataset.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any cached row features and recompute everything from scratch.",
    )
    args = parser.parse_args()
    sample_size = args.sample_size or None

    raw = load_raw(sample_size=sample_size)
    if sample_size:
        log.info("Sampling: using %d of the raw rows (sample_size=%d)", len(raw), sample_size)

    feat_df = build_feature_table(raw, fresh=args.fresh)

    # test_size as a fraction is fine at full scale, but for very small
    # samples an absolute floor of 1 example per class avoids empty splits.
    n_classes = feat_df["label"].nunique()
    test_size = 0.2
    if len(feat_df) * test_size < n_classes:
        test_size = n_classes  # falls back to an absolute count sklearn accepts

    train_df, test_df = train_test_split(
        feat_df, test_size=test_size, random_state=42, stratify=feat_df["label"]
    )

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)
    log.info("Saved train.csv: %d rows -> %s", len(train_df), OUT_TRAIN)
    log.info("Saved test.csv: %d rows -> %s", len(test_df), OUT_TEST)
    log.info("Train label balance:\n%s", train_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
