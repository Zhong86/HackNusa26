"""
Retrain loop: fold agent-caught mutated examples back into the training set
and retrain, so Layer 1 catches what it missed before.
Designed to be fast enough to run live during the demo.
"""
import pandas as pd
import joblib
from ml.features.extract import extract_features, get_feature_names
from ml.training.train import train

TRAIN_PATH = "ml/data/processed/train.csv"
MODEL_OUT = "ml/models/layer1_v1.joblib"


def add_examples_and_retrain(caught_examples: list[dict], label: int = 1):
    """
    caught_examples: list of raw email dicts (the mutated phishing emails
    that Layer 2's agent caught but Layer 1 missed).
    label: 1 = phishing (these were confirmed catches).
    """
    df = pd.read_csv(TRAIN_PATH)
    # feature set already in train.csv tells us whether embeddings are in play
    include_embedding = any(c.startswith("embed_") for c in df.columns)
    feature_names = get_feature_names(include_embedding=include_embedding)

    new_rows = []
    for email in caught_examples:
        feats = extract_features(email, include_embedding=include_embedding)
        feats["label"] = label
        new_rows.append(feats)

    new_df = pd.DataFrame(new_rows)[feature_names + ["label"]]
    updated = pd.concat([df, new_df], ignore_index=True)
    updated.to_csv(TRAIN_PATH, index=False)

    # Fast refit — no full hyperparameter search, keep this snappy for live demo
    model = train(train_path=TRAIN_PATH, model_out=MODEL_OUT, use_smote=True)
    print(f"retrained on {len(updated)} rows ({len(new_rows)} new)")

    if include_embedding:
        print(
            "note: new phishing examples were added — consider rebuilding embedding "
            "centroids (python -m ml.embeddings.build_centroids) so they stay representative."
        )
    return model


if __name__ == "__main__":
    # example: simulate one caught mutated email
    caught = [{
        "sender": "support@paypa1-secure.com",
        "display_name": "PayPal Support",
        "subject": "Account review needed",
        "body": "We noticed some irregular activity on your account and would appreciate your help confirming a few details.",
        "urls": ["http://paypa1-secure.com/verify-now"],
    }]
    add_examples_and_retrain(caught)