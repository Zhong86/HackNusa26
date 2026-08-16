"""
Feature engineering shared by training and serving.

Two feature families:
  1. Structural/keyword features (fast, deterministic, no model access) —
     kept dependency-free so they can run with zero setup.
  2. Embedding features (via ml.embeddings.client) — semantic signal from
     a local embedding model (Ollama / bge-m3 by default). These require
     the embedding backend to be reachable; callers that can't guarantee
     that (e.g. quick unit tests) can pass include_embedding=False.

CHANGELOG (this revision):
  - `keyword_hits` (a single blended count over one flat keyword list) is
    replaced by three separate category scores: urgency_score,
    financial_request_score, credential_request_score. The old blended
    count was diagnosed as near-dead weight (86% of phishing training
    rows scored 0 on it) because it required near-exact phrase matches
    and couldn't represent "financial pressure without urgency/credential
    language" as its own pattern (e.g. quiet invoice-fraud emails) versus
    "generic urgency" (e.g. account-suspension phishing). Splitting into
    categories lets the model learn each pattern independently instead of
    needing one blended signal to fire on all of them.
  - `display_name_mismatch` (a 12-brand hardcoded lookup) is joined by a
    new `org_identity_mismatch` feature that generalizes past any fixed
    brand list: it fires when the display name asserts an organizational
    identity (a company-ish name, "Support"/"Billing"/"Team"/"Notifications"
    suffix, etc.) that isn't reflected anywhere in the sending domain —
    without requiring that identity to be a brand we've hardcoded. The
    original brand-specific check is kept (still useful, high precision
    for the brands it knows) alongside the new broader one.
"""
import re
from urllib.parse import urlparse
from datetime import datetime

# --- Categorized keyword lists -------------------------------------------
# Previously a single flat SUSPICIOUS_KEYWORDS list produced one blended
# `keyword_hits` count. Splitting into categories lets "financial pressure
# with no urgency/credential language" (quiet invoice/BEC-style fraud)
# register as its own pattern instead of needing to match the same list
# as loud "verify your account now" phishing.

URGENCY_KEYWORDS = [
    "urgent", "act now", "immediately", "immediate attention",
    "limited time", "before it's too late", "failure to", "will be suspended",
    "avoid a delay", "avoid delay", "action needed", "action required",
    "expires soon", "final notice", "respond within", "within 24 hours",
    "your account will be", "as soon as possible",
]

FINANCIAL_REQUEST_KEYWORDS = [
    "invoice", "outstanding balance", "payment details", "billing details",
    "update your payment", "confirm your billing", "confirm your payment",
    "wire transfer", "bank details", "payment is due", "past due",
    "account balance", "processing your payment", "confirm your invoice",
]

CREDENTIAL_REQUEST_KEYWORDS = [
    "verify", "confirm your account", "confirm your identity",
    "confirm your details", "click here", "unusual activity",
    "update your information", "security alert", "suspended",
    "reset your password", "sign in to verify", "login to confirm",
    "verify your identity",
]


def _score_keywords(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


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

# Suffixes/words that signal a display name is asserting an ORGANIZATIONAL
# identity (a team/department/company acting on someone's behalf) rather
# than being a personal name. Used by org_identity_mismatch below — this
# is deliberately not a brand list, so it generalizes to companies we've
# never seen before (e.g. "Acme Billing", "Globex Support Team").
ORG_IDENTITY_MARKERS = [
    "support", "billing", "team", "notifications", "notification",
    "service", "services", "accounts", "account", "security", "admin",
    "administrator", "helpdesk", "help desk", "customer care",
    "customer service", "payments", "payment", "verification", "no-reply",
    "noreply", "alerts", "alert", "hr", "human resources", "finance",
    "accounting", "inc", "llc", "corp", "corporation", "company", "co.",
]

# Common personal-name markers that should NOT be treated as an org
# identity claim even if a marker word coincidentally appears (e.g. a
# person literally named "Grace Corp" is astronomically unlikely, but we
# guard the obvious case: a display name that is just "First Last" with
# no other punctuation/words is left alone regardless of marker overlap).
_TWO_TOKEN_NAME_RE = re.compile(r"^[A-Z][a-z'.-]+\s+[A-Z][a-z'.-]+$")


def _org_identity_features(sender: str, display_name: str) -> dict:
    """
    org_identity_mismatch: 1 if the display name asserts an organizational
    identity (via ORG_IDENTITY_MARKERS, e.g. "Acme Billing", "IT Support
    Team") whose asserted name doesn't appear anywhere in the sending
    domain. Unlike display_name_mismatch (brand-specific lookup), this
    doesn't require the org to be on a hardcoded list — it only needs the
    display name to *look* organizational and the domain to not obviously
    match it. A plain two-token personal name ("Maria Santos") never
    triggers this, even though "hr"/"team"/etc. substrings could otherwise
    coincidentally overlap with parts of a personal name.
    """
    sender = sender or ""
    display_name = display_name or ""
    domain = sender.split("@")[-1] if "@" in sender else ""

    if not display_name or not domain:
        return {"org_identity_mismatch": 0}

    if _TWO_TOKEN_NAME_RE.match(display_name.strip()):
        return {"org_identity_mismatch": 0}

    display_lower = display_name.lower()
    asserts_org_identity = any(marker in display_lower for marker in ORG_IDENTITY_MARKERS)
    if not asserts_org_identity:
        return {"org_identity_mismatch": 0}

    # Does any significant word from the display name show up in the domain?
    # e.g. "Acme Billing" -> "acme" should appear in "acme-invoices.net".
    domain_root = domain.split(".")[0].lower()
    name_words = [w.strip(".,") for w in display_lower.split() if w.strip(".,") not in ORG_IDENTITY_MARKERS]
    name_words = [w for w in name_words if len(w) >= 3]  # skip tiny/noise tokens

    if not name_words:
        # display name is ONLY generic org-marker words ("Support Team",
        # "IT Support", "Billing Team") with no identifying company name
        # at all. This is extremely common for legitimate internal senders
        # (a company's own IT/HR/billing team mailing its own employees)
        # and is not itself suspicious — there's no company name here to
        # check against the domain in the first place, so we can't claim
        # a mismatch. Leave unflagged; this case is a genuine "no signal
        # available" rather than "signal found and it's bad".
        return {"org_identity_mismatch": 0}

    matched = any(word in domain_root or word in domain.lower() for word in name_words)
    return {"org_identity_mismatch": 0 if matched else 1}


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
    return {
        "urgency_score": _score_keywords(text, URGENCY_KEYWORDS),
        "financial_request_score": _score_keywords(text, FINANCIAL_REQUEST_KEYWORDS),
        "credential_request_score": _score_keywords(text, CREDENTIAL_REQUEST_KEYWORDS),
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

    features = {
        "domain_length": len(domain),
        "has_digit_in_domain": has_digit_in_domain,
        "display_name_mismatch": display_mismatch,
    }
    features.update(_org_identity_features(sender, display_name))
    return features


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
    "org_identity_mismatch",
    "urgency_score", "financial_request_score", "credential_request_score",
    "subject_len", "body_len", "num_exclamations",
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
