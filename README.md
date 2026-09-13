# Phishing Email Detector

**HackNusa 2026** — national cybersecurity hackathon by Telkom University × Kaspersky
Track: *AI vs AI: Cyber Defense*

An agentic AI defender against AI-generated phishing. Demonstrates an arms race: an attacker AI mutates phishing emails to evade detection, a two-layer defender (fast ML filter + reasoning agent) catches them anyway, and the system improves via a retrain loop.

**Layer 1 (ML filter):** Random Forest / XGBoost on phishing features (URL structure, sender domain, keyword patterns). High-confidence phishing → auto-quarantine; high-confidence safe → pass; uncertain → escalate to Layer 2.

**Layer 2 (LangGraph reasoning agent):** Only borderline cases reach here. Gathers extra context (sender history, threat-intel lookup, similarity to known templates) and outputs quarantine / escalate / allow with a written justification. Low confidence → human-in-the-loop via `interrupt()`.

## Possible improvements

- Improve embedding values with a multi-centroid embedding system
- Give the agent tools to properly look up domains, check URL redirects, and inspect attachments
- Self-improving detection loop — emails the agent catches feed back into retraining the ML filter

# Postman Payloads
## Safe (<0.25)
```json 
{
  "email": {
    "sender": "maria.santos@university.edu",
    "display_name": "Maria Santos",
    "subject": "Quick question about the recommendation letter",
    "body": "Hi Professor, just checking in on the recommendation letter timeline whenever you have a moment. No rush at all. But pls dont take too long though, I would need it by the end of July",
    "urls": []
  }
}
```
## Uncertain (0.25-0.75)
```json
{
  "email": {
    "sender": "billing@acme-invoices.net",
    "display_name": "Acme Billing",
    "subject": "Invoice #4471 - action needed",
    "body": "Please confirm your billing details to avoid a delay processing invoice #4471. Let us know if you have questions.",
    "urls": ["http://acme-invoices.net/invoice/4471"]
  }
}
```
## Malicious (>0.75)
```json 
{
  "email": {
    "sender": "support@paypa1-secure.com",
    "display_name": "PayPal Support",
    "subject": "Your account has been suspended",
    "body": "Dear user, we detected unusual activity. Click here urgently to verify your account or it will be suspended immediately.",
    "urls": ["http://paypa1-secure.com/verify-now"]
  }
}
```
