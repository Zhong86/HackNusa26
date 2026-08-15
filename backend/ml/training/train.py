"""
Train the Layer 1 classifier (XGBoost) with class imbalance handling.
Saves the fitted model to ml/models/layer1_v1.joblib.

Feature columns are read directly from train.csv's header rather than
hardcoded, so this stays in sync with whichever mode preprocess.py was
run in (with or without embedding features) — the model just fits
whatever columns are present besides "label".
"""
import os

import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

TRAIN_PATH = "ml/data/processed/train.csv"
MODEL_OUT = "ml/models/layer1_v1.joblib"


def train(train_path: str = TRAIN_PATH, model_out: str = MODEL_OUT, use_smote: bool = True):
    df = pd.read_csv(train_path)
    feature_names = [c for c in df.columns if c != "label"]
    X = df[feature_names]
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
    joblib.dump({"model": model, "feature_names": feature_names}, model_out)
    print(f"model saved to {model_out} ({len(feature_names)} features: {feature_names})")
    return model


if __name__ == "__main__":
    train()