"""
Attacker AI: rewrites known phishing emails to evade Layer 1 while
keeping malicious intent. Uses an LLM to strip obvious red-flag phrasing.
"""
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MUTATION_PROMPT = """You are helping test a phishing detection system's robustness (authorized red-team exercise).
Rewrite the following phishing email so it sounds more natural and professional, \
removing obvious red-flag phrasing (e.g. "act now", "urgent", "click here"), \
while keeping the same underlying malicious request (e.g. credential harvesting, fake urgency about an account).
Keep the same sender intent and target action. Only output the rewritten body text, nothing else.

Original email:
Subject: {subject}
Body: {body}
"""


def mutate_email(email: dict) -> dict:
    prompt = MUTATION_PROMPT.format(
        subject=email.get("subject", ""),
        body=email.get("body", ""),
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    mutated_body = response.content[0].text

    mutated = dict(email)
    mutated["body"] = mutated_body
    mutated["subject"] = email.get("subject", "")  # optionally mutate subject too
    return mutated


if __name__ == "__main__":
    original = {
        "sender": "support@paypa1-secure.com",
        "display_name": "PayPal Support",
        "subject": "Your account has been suspended",
        "body": "Dear user, we detected unusual activity. Click here urgently to verify your account or it will be suspended.",
        "urls": ["http://paypa1-secure.com/verify-now"],
    }
    print(mutate_email(original))