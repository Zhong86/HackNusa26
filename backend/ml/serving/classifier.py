"""
The one function Person B's backend depends on.
score_email(email) -> schemas.Layer1Score (the same Layer1Score used
everywhere else in the app — no separate serving-only type).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from schemas import EmailPayload, Layer1Score
from ml.features.extract import extract_features
from ml.serving.model_store import load_model

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "layer1_v1.joblib"

_bundle = None


def _get_bundle():
    """
    Returns {"model": ..., "feature_names": [...]}. Model artifacts saved
    by the current train.py are bundles like this; older artifacts saved
    as a bare model object are still supported for backwards compat, using
    the legacy structural-only feature order.
    """
    global _bundle
    if _bundle is None:
        loaded = load_model(str(MODEL_PATH))
        if isinstance(loaded, dict) and "model" in loaded:
            _bundle = loaded
        else:
            # legacy artifact: bare model, structural features only
            from ml.features.extract import STRUCTURAL_FEATURE_NAMES
            _bundle = {"model": loaded, "feature_names": STRUCTURAL_FEATURE_NAMES}
    return _bundle


def score_email(email: Union[EmailPayload, dict]) -> Layer1Score:
    email_dict = email.model_dump() if isinstance(email, EmailPayload) else dict(email)

    bundle = _get_bundle()
    feature_names = bundle["feature_names"]
    include_embedding = any(name.startswith("embed_") for name in feature_names)

    features = extract_features(email_dict, include_embedding=include_embedding)
    ordered = [features[name] for name in feature_names]
    proba = bundle["model"].predict_proba([ordered])[0][1]
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
