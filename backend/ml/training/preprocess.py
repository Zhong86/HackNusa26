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