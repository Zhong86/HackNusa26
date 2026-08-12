"""
Feature engineering shared by training and serving.
Keep this deterministic and dependency-free (no model access here).
"""
import re
from urllib.parse import urlparse
from datetime import datetime

SUSPICIOUS_KEYWORDS = [
    "verify", "suspended", "urgent", "confirm your account",
    "click here", "unusual activity", "limited time", "act now",
    "update your information", "security alert",
]

def _url_features(urls: list[str]) -> dict:
    if not urls:
        return {
            "num_urls": 0,
            "has_ip_url": 0,
            "avg_url_length": 0,
            "has_at_symbol": 0,
            "num_subdomains_max": 0,
            "has_https": 0,
        }

    lengths = []
    has_ip = 0
    has_at = 0
    max_subdomains = 0
    has_https = 0

    ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

    for u in urls:
        parsed = urlparse(u)
        lengths.append(len(u))
        host = parsed.hostname or ""
        if ip_pattern.match(host):
            has_ip = 1
        if "@" in u:
            has_at = 1
        max_subdomains = max(max_subdomains, host.count("."))
        if parsed.scheme == "https":
            has_https = 1

    return {
        "num_urls": len(urls),
        "has_ip_url": has_ip,
        "avg_url_length": sum(lengths) / len(lengths),
        "has_at_symbol": has_at,
        "num_subdomains_max": max_subdomains,
        "has_https": has_https,
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

    # display name doesn't match sender domain, e.g. "PayPal Support" from a random domain
    display_mismatch = int(
        bool(display_name) and display_name.split()[0].lower() not in domain.lower()
    ) if domain else 0

    return {
        "domain_length": len(domain),
        "has_digit_in_domain": has_digit_in_domain,
        "display_name_mismatch": display_mismatch,
    }


def extract_features(email: dict) -> dict:
    """
    email: {sender, display_name, subject, body, urls}
    Returns a flat dict of numeric features, stable key order.
    """
    features = {}
    features.update(_sender_features(email.get("sender"), email.get("display_name")))
    features.update(_text_features(email.get("subject"), email.get("body")))
    features.update(_url_features(email.get("urls") or []))
    return features


FEATURE_NAMES = [
    "domain_length", "has_digit_in_domain", "display_name_mismatch",
    "keyword_hits", "subject_len", "body_len", "num_exclamations",
    "num_urls", "has_ip_url", "avg_url_length", "has_at_symbol",
    "num_subdomains_max", "has_https",
]