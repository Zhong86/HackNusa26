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


# Known URL shorteners — a redirect through one of these hides the real
# destination, a common phishing trick to dodge naive domain checks.
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
}

# TLDs disproportionately used for throwaway/phishing domains (cheap or
# free registration, weak abuse enforcement). Not proof of malice on their
# own — just a mild risk signal, same spirit as has_digit_in_domain.
SUSPICIOUS_TLDS = {"xyz", "tk", "ml", "ga", "cf", "top", "click", "link", "work", "gq"}

_IP_HOST_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def _url_features(urls: list[str]) -> dict:
    """
    urls: list of URL strings found in the email (see preprocess.py's
    _extract_urls() for how these are pulled out of raw CEAS_08 body text
    at training time; at serving time these come straight from
    EmailPayload.urls). Both paths converge on the same list[str] shape
    so this function never has to know which caller it's serving.
    """
    if not urls:
        return {
            "has_url": 0,
            "num_urls": 0,
            "has_ip_url": 0,
            "has_shortener_url": 0,
            "suspicious_tld_flag": 0,
            "max_url_path_depth": 0,
        }

    has_ip = 0
    has_shortener = 0
    suspicious_tld = 0
    max_depth = 0

    for u in urls:
        # be forgiving of bare domains with no scheme (e.g. "example.com/x")
        candidate = u if "://" in u else f"http://{u}"
        try:
            parsed = urlparse(candidate)
        except ValueError:
            continue

        host = (parsed.hostname or "").lower()
        if not host:
            continue

        if _IP_HOST_RE.match(host):
            has_ip = 1
        if host in URL_SHORTENERS:
            has_shortener = 1

        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        if tld in SUSPICIOUS_TLDS:
            suspicious_tld = 1

        depth = len([p for p in parsed.path.split("/") if p])
        max_depth = max(max_depth, depth)

    return {
        "has_url": 1,
        "num_urls": len(urls),
        "has_ip_url": has_ip,
        "has_shortener_url": has_shortener,
        "suspicious_tld_flag": suspicious_tld,
        "max_url_path_depth": max_depth,
    }


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
    features.update(_url_features(email.get("urls") or []))

    if include_embedding:
        from ml.embeddings.features import embedding_features
        features.update(embedding_features(email.get("subject"), email.get("body")))

    return features


STRUCTURAL_FEATURE_NAMES = [
    "domain_length", "has_digit_in_domain", "display_name_mismatch",
    "keyword_hits", "subject_len", "body_len", "num_exclamations",
    "has_url", "num_urls", "has_ip_url", "has_shortener_url",
    "suspicious_tld_flag", "max_url_path_depth",
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
