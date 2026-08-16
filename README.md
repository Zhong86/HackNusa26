# Model
- Ollama bge:m3-latest

# Postman Payloads
## Safe (<0.25)
```json
{
  "email": {
    "sender": "no-reply@github.com",
    "display_name": "GitHub",
    "subject": "[sentinel-loop] New pull request",
    "body": "A new pull request was opened on sentinel-loop by teammate. Review it when you get a chance.",
    "urls": ["https://github.com/org/sentinel-loop/pull/17"]
  }
}
```
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
```json
{
  "email": {
    "sender": "hr@acmecorp-benefits.com",
    "display_name": "Acme HR",
    "subject": "Update your benefits enrollment",
    "body": "Open enrollment closes soon. Please review and confirm your selections in the portal.",
    "urls": ["http://acmecorp-benefits.com/portal"]
  }
}
```
## Malicious (>0.75)
```json
{
  "email": {
    "sender": "security@micr0soft-alerts.com",
    "display_name": "Microsoft",
    "subject": "Security alert: unusual activity detected",
    "body": "Act now to confirm your account or it will be suspended within 24 hours. Click here urgently to verify.",
    "urls": ["http://micr0soft-alerts.com/confirm"]
  }
}
```
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
