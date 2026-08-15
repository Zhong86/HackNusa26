"""
Mock stand-in for Person A's score_email().

Contract (agreed in writing, per the checklist):
    score_email(email: EmailPayload) -> Layer1Score
        score: float in [0, 1], probability the email is phishing
        features: dict of feature-name -> value, model-internal, opaque to Person B

This mock uses cheap heuristics (lookalike domains, suspicious keywords,
raw IP / no-https URLs) just so the graph has *something* realistic to
branch on before the real model exists. On integration day, replace the
body of score_email() with a call to Person A's actual endpoint/function
and nothing else in the graph needs to change.
"""

from __future__ import annotations

import re

from schemas import EmailPayload, Layer1Score

SUSPICIOUS_KEYWORDS = [
    "suspended",
    "verify now",
    "verify your account",
    "unusual activity",
    "click here",
    "act now",
    "confirm your identity",
    "urgent",
    "limited time",
]

# crude lookalike detector: digits substituted for letters in known brand names
LOOKALIKE_PATTERN = re.compile(r"(paypa1|amaz0n|micr0soft|g00gle|app1e|netfl1x)", re.IGNORECASE)


def score_email(email: EmailPayload) -> Layer1Score:
    """Cheap heuristic mock of Layer 1. Deterministic given the same input."""
    features: dict = {}
    signal = 0.0

    # sender domain lookalike check
    domain_match = LOOKALIKE_PATTERN.search(email.sender)
    features["lookalike_domain"] = bool(domain_match)
    if domain_match:
        signal += 0.45

    # keyword scan across subject + body
    haystack = f"{email.subject} {email.body}".lower()
    hit_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in haystack]
    features["keyword_hits"] = hit_keywords
    signal += min(len(hit_keywords) * 0.12, 0.36)

    # url red flags: no https, raw IP, or lookalike domain in URL
    url_flags = []
    for url in email.urls:
        flags = []
        if not url.startswith("https://"):
            flags.append("no_https")
        if re.search(r"://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            flags.append("raw_ip")
        if LOOKALIKE_PATTERN.search(url):
            flags.append("lookalike_domain")
        if flags:
            url_flags.append({"url": url, "flags": flags})
    features["url_flags"] = url_flags
    signal += min(len(url_flags) * 0.15, 0.3)

    # display name / sender domain mismatch (e.g. "PayPal" from a non-paypal.com domain)
    brand_domains = {
        "paypal": "paypal.com",
        "amazon": "amazon.com",
        "microsoft": "microsoft.com",
        "google": "google.com",
        "apple": "apple.com",
        "netflix": "netflix.com",
    }
    display_lower = email.display_name.lower()
    sender_domain = email.sender.split("@")[-1].lower()
    mentioned_brand = next((b for b in brand_domains if b in display_lower), None)
    mismatch = bool(mentioned_brand) and not sender_domain.endswith(brand_domains[mentioned_brand])
    features["brand_display_mismatch"] = mismatch
    if mismatch:
        signal += 0.25

    score = max(0.0, min(signal, 1.0))
    return Layer1Score(score=round(score, 4), features=features)