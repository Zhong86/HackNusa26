"""
Load raw dataset, clean it, extract features, handle class imbalance.
Run once to produce ml/data/processed/train.csv and test.csv.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.features.extract import extract_features, FEATURE_NAMES

RAW_PATH = "ml/data/raw/phishing_emails.csv"   # adjust to your actual Kaggle file
OUT_TRAIN = "ml/data/processed/train.csv"
OUT_TEST = "ml/data/processed/test.csv"


def load_raw() -> pd.DataFrame:
    """
    Expected raw columns (adjust to whatever the chosen Kaggle dataset actually has):
      sender, display_name, subject, body, urls (comma-separated string), label (0/1)
    """
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["label"])
    df["urls"] = df["urls"].fillna("").apply(
        lambda s: [u.strip() for u in s.split(",") if u.strip()]
    )
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        email = {
            "sender": row.get("sender", ""),
            "display_name": row.get("display_name", ""),
            "subject": row.get("subject", ""),
            "body": row.get("body", ""),
            "urls": row.get("urls", []),
        }
        feats = extract_features(email)
        feats["label"] = int(row["label"])
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    # keep stable column order: features then label
    return feat_df[FEATURE_NAMES + ["label"]]


def main():
    raw = load_raw()
    feat_df = build_feature_table(raw)

    train_df, test_df = train_test_split(
        feat_df, test_size=0.2, random_state=42, stratify=feat_df["label"]
    )

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")
    print(f"train label balance:\n{train_df['label'].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()