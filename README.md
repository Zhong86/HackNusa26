# Sentinel Loop

**HackNusa 2026** — national cybersecurity hackathon by Telkom University × Kaspersky
Track: *AI vs AI: Cyber Defense*

Sentinel Loop is an agentic AI defender against AI-generated phishing. It demonstrates
an arms race: an **attacker AI** rewrites known phishing emails to slip past a filter,
a **two-layer defender** (fast ML classifier + LLM reasoning agent) catches them anyway,
and the filter **retrains** on what the agent caught — so the next mutation round gets
caught by Layer 1 alone.

```
attacker mutates email → Layer 1 (ML) filters → uncertain? → Layer 2 (agent) reasons
                                                                     ↓
                                          caught-but-missed examples feed retrain.py
                                                                     ↓
                                             Layer 1 now catches the mutation directly
```

## Why

- BEC (business email compromise) losses: **>$55B** globally, 2013–2023 (FBI IC3)
- **14x** surge in AI-generated phishing (Hoxhunt)
- **3.4B** phishing emails sent daily (Zensec)

AI is making phishing cheaper to generate and harder to fingerprint with static rules.
Sentinel Loop's bet is that a system which *learns from what it misses* — instead of a
filter that goes stale the moment attackers change their phrasing — is a better match
for that arms race than either a static classifier or a slow, fully-manual SOC review.

## Architecture

**Layer 1 — ML filter** (`backend/ml/`)
XGBoost classifier over structural + semantic features: sender/domain lookalikes,
brand and org-identity impersonation, URL structure (shorteners, IPs, suspicious TLDs),
urgency/financial/credential keyword scoring, and embedding similarity to phishing vs.
benign centroids (local Ollama `bge-m3` embeddings). High/low confidence scores decide
immediately; mid-range scores ("uncertain zone", configurable thresholds) escalate to
Layer 2.

**Layer 2 — reasoning agent** (`backend/graph/`, built on LangGraph)
Only borderline cases reach here. Gathers context (sender history) and asks an LLM
(OpenAI-compatible endpoint — Groq/Llama by default) to reason over the raw email and
Layer 1's score, returning a structured verdict (`allow` / `quarantine` / `escalate`)
with confidence, justification, evidence, and MITRE ATT&CK technique IDs. If the LLM
call fails, the graph fails safe and auto-escalates rather than silently passing the
email through.

```
START → score_node ─┬─ confident ──→ direct_decision_node ──→ END
                     └─ uncertain ──→ gather_context_node → reason_node → auto_decide_node → END
```

> Current build runs fully automated (no `interrupt()` pause) since this is a research/
> demo pipeline, not something sitting in front of real inboxes — every run goes
> straight through to a final verdict so the trace can be observed end-to-end. A
> human-in-the-loop pause on low-confidence Layer 2 output is on the roadmap (see below).

**Attacker AI** (`backend/ml/attacker/mutate.py`)
Takes a known-phishing email and asks an LLM to rewrite it — stripping obvious red-flag
phrasing ("act now", "click here") while preserving the malicious intent — to probe
whether Layer 1 still catches the rewritten version.

**Retrain loop** (`backend/ml/training/retrain.py`)
Feeds mutated emails the agent caught (but Layer 1 missed) back into the training set
and refits Layer 1, fast enough to run live during a demo.

## Roadmap

- Human-in-the-loop pause (`interrupt()`) on low-confidence Layer 2 output
- Multi-centroid embeddings (beyond a single phishing/benign centroid pair)
- Agent tools for live domain/URL-redirect/attachment inspection
- Fully closed self-improving loop: agent catches → retrain → re-attack → repeat

# Model
- Ollama bge:m3-latest

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
