"""
Evaluate a trained model: precision/recall/F1, confusion matrix.
Used both for the initial baseline and for before/after retrain comparisons.
"""
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

TEST_PATH = "ml/data/processed/test.csv"
MODEL_PATH = "ml/models/layer1_v1.joblib"


def evaluate(model_path: str = MODEL_PATH, test_path: str = TEST_PATH) -> dict:
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        feature_names = loaded["feature_names"]
    else:
        model = loaded
        from ml.features.extract import STRUCTURAL_FEATURE_NAMES
        feature_names = STRUCTURAL_FEATURE_NAMES

    df = pd.read_csv(test_path)
    X = df[feature_names]
    y_true = df["label"]

    y_pred = model.predict(X)

    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(classification_report(y_true, y_pred, target_names=["safe", "phishing"]))
    print("confusion matrix:", metrics["confusion_matrix"])
    return metrics


if __name__ == "__main__":
    evaluate()