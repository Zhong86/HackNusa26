"""
Train the Layer 1 classifier (XGBoost) with class imbalance handling.
Saves the fitted model to ml/models/layer1_v1.joblib.

For quick smoke tests, cap the number of training rows used with
--sample-size (CLI) or the SAMPLE_SIZE env var. This trims train.csv
itself (post feature-extraction), so it's the fast knob to use when
you already have a full-size train.csv but just want a quick fit —
use preprocess.py's --sample-size instead if you want to skip the
(slower) feature-extraction step on most rows too.

Examples:
    python -m ml.training.train --sample-size 200
    SAMPLE_SIZE=200 python -m ml.training.train
"""
import argparse
import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from ml.features.extract import FEATURE_NAMES
from logger import get_logger

log = get_logger(__name__)

TRAIN_PATH = "ml/data/processed/train.csv"
MODEL_OUT = "ml/models/layer1_v1.joblib"

# Optional cap on training rows used. None/0 = use the full file.
SAMPLE_SIZE = int(os.environ["SAMPLE_SIZE"]) if os.environ.get("SAMPLE_SIZE") else None


def train(
    train_path: str = TRAIN_PATH,
    model_out: str = MODEL_OUT,
    use_smote: bool = True,
    sample_size: int | None = None,
):
    df = pd.read_csv(train_path)
    log.info("Loaded %d rows from %s", len(df), train_path)

    if sample_size and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size, random_state=42, stratify=df["label"]
        )
        df = df.reset_index(drop=True)
        log.info("Sampling: training on %d of the rows (sample_size=%d)", len(df), sample_size)

    X = df[FEATURE_NAMES]
    y = df["label"]

    if use_smote:
        n_before = len(X)
        X, y = SMOTE(random_state=42).fit_resample(X, y)
        log.info("SMOTE resampling: %d -> %d rows", n_before, len(X))

    log.info("Fitting XGBClassifier on %d rows, %d features", len(X), len(FEATURE_NAMES))
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    log.info("Training complete: %d rows fit successfully", len(X))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_out)
    log.info("Model saved to %s", model_out)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Cap on training rows used (stratified by label). Omit or pass 0 for the full file.",
    )
    args = parser.parse_args()
    train(sample_size=args.sample_size or None)
