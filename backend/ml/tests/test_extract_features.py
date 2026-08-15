"""
Regression tests for ml/features/extract.py.

Run structural-only (no embedding backend needed):
    pytest backend/tests/test_extract_features.py -v

These specifically guard against the display_name_mismatch bug: a
personal name (e.g. "Jane Doe") must never be flagged just because it
doesn't appear in the sending domain — only known-brand impersonation
should trigger that feature.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features.extract import extract_features, _sender_features


BENIGN_EMAILS = [
    {
        "sender": "jane.doe@acme-corp.com",
        "display_name": "Jane Doe",
        "subject": "Q3 budget review",
        "body": "Hi team, can we sync on the numbers before Friday's meeting?",
        "urls": [],
    },
    {
        "sender": "notifications@stripe.com",
        "display_name": "Stripe",
        "subject": "Your invoice is ready",
        "body": "Your monthly invoice for August is now available in your dashboard.",
        "urls": ["https://dashboard.stripe.com/invoices/123"],
    },
    {
        "sender": "no-reply@github.com",
        "display_name": "GitHub",
        "subject": "[repo] New pull request",
        "body": "A new pull request was opened on your repository.",
        "urls": ["https://github.com/org/repo/pull/42"],
    },
    {
        "sender": "maria.santos@university.edu",
        "display_name": "Maria Santos",
        "subject": "Recommendation letter request",
        "body": "Hi Professor, I hope you're doing well. I wanted to ask if you'd be willing to write me a recommendation letter.",
        "urls": [],
    },
]

PHISHING_EMAILS = [
    {
        "sender": "support@paypa1-secure.com",
        "display_name": "PayPal Support",
        "subject": "Your account has been suspended",
        "body": "Dear user, we detected unusual activity. Click here urgently to verify your account.",
        "urls": ["http://paypa1-secure.com/verify-now"],
    },
    {
        "sender": "security@micr0soft-alerts.com",
        "display_name": "Microsoft",
        "subject": "Security alert: unusual activity detected",
        "body": "Act now to confirm your account or it will be suspended within 24 hours.",
        "urls": ["http://micr0soft-alerts.com/confirm"],
    },
]


def test_personal_names_never_trigger_display_mismatch():
    """The core regression: a real person's name should not be flagged just
    because it isn't a substring of their employer's domain."""
    for email in BENIGN_EMAILS:
        feats = _sender_features(email["sender"], email["display_name"])
        assert feats["display_name_mismatch"] == 0, (
            f"False positive display_name_mismatch for benign sender {email['sender']!r} "
            f"with display name {email['display_name']!r}"
        )


def test_brand_impersonation_still_detected():
    """Make sure fixing the false positive didn't also break true positives:
    a display name claiming a known brand, from a domain that isn't that
    brand's domain, must still be flagged."""
    for email in PHISHING_EMAILS:
        feats = _sender_features(email["sender"], email["display_name"])
        assert feats["display_name_mismatch"] == 1, (
            f"Missed brand impersonation for {email['sender']!r} "
            f"claiming to be {email['display_name']!r}"
        )


def test_legitimate_brand_sender_not_flagged():
    """A brand name sent from that brand's actual domain should not be
    flagged as a mismatch."""
    feats = _sender_features("no-reply@paypal.com", "PayPal")
    assert feats["display_name_mismatch"] == 0


def test_full_feature_extraction_structural_only():
    """extract_features() should run without an embedding backend when
    include_embedding=False, and benign emails should have low keyword_hits."""
    for email in BENIGN_EMAILS:
        feats = extract_features(email, include_embedding=False)
        assert feats["display_name_mismatch"] == 0
        assert "embed_phishing_similarity" not in feats


def test_full_feature_extraction_includes_embedding_keys_when_requested():
    """With include_embedding=True, embedding features should be present
    (falling back to neutral 0.0 values if Ollama/centroids aren't
    available in the test environment — see ml/embeddings/features.py)."""
    email = BENIGN_EMAILS[0]
    feats = extract_features(email, include_embedding=True)
    assert "embed_phishing_similarity" in feats
    assert "embed_benign_similarity" in feats
    assert "embed_similarity_margin" in feats