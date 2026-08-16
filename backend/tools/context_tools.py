"""
Layer 2 context tools.

lookup_sender_history() reads real, accumulated data from a local SQLite
DB (see db.py) — every email that runs through the graph gets recorded,
so this reflects actual prior traffic, fully offline.

domain_age and threat_intel stubs were removed intentionally: they were
fake (hash-based) data that could feed the reasoning LLM misleading
signals, and a real replacement would require per-email network calls
(WHOIS, threat-intel API), which conflicts with keeping the pipeline
offline-first. Re-add here if/when real, rate-limit-aware integrations
are ready.
"""

from __future__ import annotations

from db import get_sender_history
from schemas import EmailPayload

# Kalo pake DB
# def lookup_sender_history(email: EmailPayload) -> dict:
#     """Real lookup: has this sender been seen before, based on actual recorded traffic?"""
#     return get_sender_history(email.sender)

def lookup_sender_history(email: EmailPayload) -> dict:
    """No persistence — always reports the sender as unseen."""
    return {
        "sender": email.sender,
        "seen_before": False,
        "first_seen_days_ago": None,
        "prior_flag_count": 0,
    }
 

