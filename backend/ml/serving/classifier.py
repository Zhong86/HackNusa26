"""
The one function Person B's backend depends on.
score_email(email) -> object matching backend/schemas.py::Layer1Score
"""
from dataclasses import dataclass, asdict
from ml.features.extract import extract_features, FEATURE_NAMES
from ml.serving.model_store import load_model

MODEL_PATH = "ml/models/layer1_v1.joblib"
_model = load_model(MODEL_PATH)


@dataclass
class Layer1Score:
    score: float   # calibrated probability of phishing, 0.0-1.0
    features: dict


def score_email(email: dict) -> Layer1Score:
    """
    email: {sender, display_name, subject, body, urls}
    """
    features = extract_features(email)
    ordered = [features[name] for name in FEATURE_NAMES]
    proba = _model.predict_proba([ordered])[0][1]
    return Layer1Score(score=float(proba), features=features)


if __name__ == "__main__":
    # quick manual smoke test
    sample = {
        "sender": "support@paypa1-secure.com",
        "display_name": "PayPal Support",
        "subject": "Your account has been suspended",
        "body": "Dear user, we detected unusual activity...",
        "urls": ["http://paypa1-secure.com/verify-now"],
    }
    result = score_email(sample)
    print(asdict(result))