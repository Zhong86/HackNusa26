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

    if sample_size and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size, random_state=42, stratify=df["label"]
        )
        df = df.reset_index(drop=True)
        print(f"sampling: training on {len(df)} of the rows in {train_path} (sample_size={sample_size})")

    X = df[FEATURE_NAMES]
    y = df["label"]

    if use_smote:
        X, y = SMOTE(random_state=42).fit_resample(X, y)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"model saved to {model_out}")
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