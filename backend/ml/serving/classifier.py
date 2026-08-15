"""
The one function Person B's backend depends on.
score_email(email) -> schemas.Layer1Score (the same Layer1Score used
everywhere else in the app — no separate serving-only type).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from schemas import EmailPayload, Layer1Score
from ml.features.extract import extract_features, FEATURE_NAMES
from ml.serving.model_store import load_model

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "layer1_v1.joblib"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model(str(MODEL_PATH))
    return _model


def score_email(email: Union[EmailPayload, dict]) -> Layer1Score:
    email_dict = email.model_dump() if isinstance(email, EmailPayload) else dict(email)
    email_dict["has_url_flag"] = int(bool(email_dict.get("urls")))

    features = extract_features(email_dict)
    ordered = [features[name] for name in FEATURE_NAMES]
    proba = _get_model().predict_proba([ordered])[0][1]
    return Layer1Score(score=float(proba), features=features)


if __name__ == "__main__":
    sample = {
        "sender": "support@paypa1-secure.com",
        "display_name": "PayPal Support",
        "subject": "Your account has been suspended",
        "body": "Dear user, we detected unusual activity...",
        "urls": ["http://paypa1-secure.com/verify-now"],
    }
    print(score_email(sample).model_dump())