"""
Feature engineering shared by training and serving.

Two feature families:
  1. Structural/keyword features (fast, deterministic, no model access) —
     kept dependency-free so they can run with zero setup.
  2. Embedding features (via ml.embeddings.client) — semantic signal from
     a local embedding model (Ollama / bge-m3 by default). These require
     the embedding backend to be reachable; callers that can't guarantee
     that (e.g. quick unit tests) can pass include_embedding=False.
"""
import re
from urllib.parse import urlparse
from datetime import datetime

SUSPICIOUS_KEYWORDS = [
    "verify", "suspended", "urgent", "confirm your account",
    "click here", "unusual activity", "limited time", "act now",
    "update your information", "security alert",
]

# Brands commonly impersonated in phishing display names. Only names that
# reference one of these are checked against the sending domain — this is
# a targeted impersonation check, not a "does this person's name appear in
# the domain" check (real senders' names never appear in their company's
# domain, and that's normal, not suspicious).
IMPERSONATED_BRANDS = {
    "paypal": "paypal.com",
    "amazon": "amazon.com",
    "microsoft": "microsoft.com",
    "google": "google.com",
    "apple": "apple.com",
    "netflix": "netflix.com",
    "bank of america": "bankofamerica.com",
    "chase": "chase.com",
    "wells fargo": "wellsfargo.com",
    "docusign": "docusign.com",
    "linkedin": "linkedin.com",
    "dropbox": "dropbox.com",
}


def _url_features(has_url_flag: int) -> dict:
    return {"has_url": int(has_url_flag)}


def _text_features(subject: str, body: str) -> dict:
    text = f"{subject or ''} {body or ''}".lower()
    keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text)
    return {
        "keyword_hits": keyword_hits,
        "subject_len": len(subject or ""),
        "body_len": len(body or ""),
        "num_exclamations": text.count("!"),
    }


def _sender_features(sender: str, display_name: str) -> dict:
    sender = sender or ""
    display_name = display_name or ""
    domain = sender.split("@")[-1] if "@" in sender else ""

    # crude lookalike check: digits substituted for letters (paypa1.com etc.)
    has_digit_in_domain = int(any(c.isdigit() for c in domain.split(".")[0])) if domain else 0

    # brand-impersonation check: does the display name *claim* to be a known
    # brand while the sending domain doesn't match that brand? A personal
    # name ("Jane Doe") never matches this — only names that mention a
    # brand we track are evaluated, so ordinary senders never get flagged.
    display_lower = display_name.lower()
    mentioned_brand = next((b for b in IMPERSONATED_BRANDS if b in display_lower), None)
    display_mismatch = int(
        bool(mentioned_brand) and not domain.lower().endswith(IMPERSONATED_BRANDS[mentioned_brand])
    ) if domain else 0

    return {
        "domain_length": len(domain),
        "has_digit_in_domain": has_digit_in_domain,
        "display_name_mismatch": display_mismatch,
    }


def extract_features(email: dict, include_embedding: bool = True) -> dict:
    features = {}
    features.update(_sender_features(email.get("sender"), email.get("display_name")))
    features.update(_text_features(email.get("subject"), email.get("body")))
    features.update(_url_features(email.get("has_url_flag", 0)))

    if include_embedding:
        from ml.embeddings.features import embedding_features
        features.update(embedding_features(email.get("subject"), email.get("body")))

    return features


STRUCTURAL_FEATURE_NAMES = [
    "domain_length", "has_digit_in_domain", "display_name_mismatch",
    "keyword_hits", "subject_len", "body_len", "num_exclamations",
    "has_url",
]

# Populated lazily so importing this module doesn't require the embedding
# backend to be configured (e.g. for quick tests of the structural features
# alone). Call get_feature_names() to get the full ordered list actually
# used by a given model.
def get_feature_names(include_embedding: bool = True) -> list[str]:
    if not include_embedding:
        return list(STRUCTURAL_FEATURE_NAMES)
    from ml.embeddings.features import EMBEDDING_FEATURE_NAMES
    return STRUCTURAL_FEATURE_NAMES + EMBEDDING_FEATURE_NAMES


# Backwards-compatible name some modules import directly. Defaults to the
# full feature set (structural + embedding) since that's what train.py /
# classifier.py should use going forward.
FEATURE_NAMES = get_feature_names(include_embedding=True)