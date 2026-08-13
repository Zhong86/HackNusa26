"""
Load raw dataset, clean it, extract features, handle class imbalance.
Run once to produce ml/data/processed/train.csv and test.csv.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from ml.features.extract import extract_features, FEATURE_NAMES

RAW_PATH = "ml/data/raw/CEAS_08.csv"   # adjust to your actual Kaggle file
OUT_TRAIN = "ml/data/processed/train.csv"
OUT_TEST = "ml/data/processed/test.csv"

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

    for col in ["sender", "display_name", "subject", "body"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["urls"] = df["urls"].fillna("").apply(
        lambda s: [u.strip() for u in s.split(",") if u.strip()]
    )
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        email = {
            "sender": row.get("sender_email", ""),
            "display_name": row.get("display_name", ""),
            "subject": row.get("subject", ""),
            "body": row.get("body", ""),
            "has_url_flag": row.get("has_url_flag", 0),
        }
        feats = extract_features(email)
        feats["label"] = int(row["label"])
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
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