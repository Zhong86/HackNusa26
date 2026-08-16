"""
Load raw dataset, clean it, extract features, handle class imbalance.
Run once to produce ml/data/processed/train.csv and test.csv.

For quick local testing, cap the number of raw rows processed with
--sample-size (CLI) or the SAMPLE_SIZE env var — feature extraction is
the slow part (~2s/row), so capping upstream keeps a smoke-test run
fast. Sampling is stratified by label so both classes still show up
even in a small sample.

Examples:
    python -m ml.training.preprocess --sample-size 200
    SAMPLE_SIZE=200 python -m ml.training.preprocess
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import re
from ml.features.extract import extract_features, FEATURE_NAMES
from logger import get_logger

log = get_logger(__name__)

RAW_PATH = "ml/data/raw/CEAS_08.csv"   # adjust to your actual Kaggle file
OUT_TRAIN = "ml/data/processed/train.csv"
OUT_TEST = "ml/data/processed/test.csv"

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
            df = df.reset_index(drop=True)

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


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    n_success = 0
    n_failed = 0

    log.info("Feature extraction starting: %d rows", total)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
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
            rows.append(feats)
            n_success += 1
        except Exception:
            n_failed += 1
            log.exception(
                "Feature extraction failed for row %d (sender=%s) — skipping",
                i, email.get("sender"),
            )
            continue

        if i % 50 == 0 or i == total:
            log.info(
                "Feature extraction progress: %d/%d done (%d ok, %d failed)",
                i, total, n_success, n_failed,
            )

    log.info(
        "Feature extraction finished: %d/%d succeeded, %d failed",
        n_success, total, n_failed,
    )

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
    args = parser.parse_args()
    sample_size = args.sample_size or None

    raw = load_raw(sample_size=sample_size)
    if sample_size:
        log.info("Sampling: using %d of the raw rows (sample_size=%d)", len(raw), sample_size)

    feat_df = build_feature_table(raw)

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
