"""
Layer 2 context tools. All stubbed with deterministic fake data for now.

Each function is written as a plain callable, not bound to any
LangGraph/agent specifics, so they can be swapped for real lookups
(a real sender-history store, a WHOIS call, an actual threat-intel API)
without touching the graph wiring — only the function bodies change.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from app.schemas import EmailPayload


def _seeded_hash(value: str) -> int:
    """Deterministic pseudo-randomness so demo runs are repeatable."""
    return int(hashlib.sha256(value.encode()).hexdigest(), 16)


def lookup_sender_history(email: EmailPayload) -> dict:
    """Stub: has this sender been seen before, and was it flagged?"""
    h = _seeded_hash(email.sender)
    seen_before = h % 3 != 0  # ~66% "known" senders
    prior_flags = h % 5  # 0-4 prior flags, deterministic per sender

    return {
        "sender": email.sender,
        "seen_before": seen_before,
        "first_seen_days_ago": (h % 400) if seen_before else None,
        "prior_flag_count": prior_flags if seen_before else 0,
    }


def lookup_domain_age(email: EmailPayload) -> dict:
    """Stub: how old is the sending domain? Freshly registered = red flag."""
    domain = email.sender.split("@")[-1] if "@" in email.sender else email.sender
    h = _seeded_hash(domain)
    age_days = h % 3650  # up to ~10 years
    registered_on = (datetime.now(timezone.utc) - timedelta(days=age_days)).date().isoformat()

    return {
        "domain": domain,
        "age_days": age_days,
        "registered_on": registered_on,
        "newly_registered": age_days < 30,
    }


def lookup_threat_intel(email: EmailPayload) -> dict:
    """Stub: fake threat-intel hit for sender domain / URLs."""
    domain = email.sender.split("@")[-1] if "@" in email.sender else email.sender
    h = _seeded_hash(domain + "intel")
    is_known_malicious = h % 4 == 0  # ~25% hit rate, deterministic

    matched_urls = []
    for url in email.urls:
        uh = _seeded_hash(url)
        if uh % 3 == 0:
            matched_urls.append(url)

    return {
        "domain_flagged": is_known_malicious,
        "source": "mock_threat_intel_feed" if is_known_malicious else None,
        "matched_malicious_urls": matched_urls,
    }