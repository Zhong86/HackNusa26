"""
Client LLM generik untuk provider yang kompatibel dengan format OpenAI API
(base_url + api_key bisa diarahkan ke provider mana saja, bukan cuma OpenAI).

Konfigurasi lewat environment variable:
    LLM_BASE_URL   - contoh: https://api.groq.com/openai/v1
    LLM_API_KEY    - API key provider
    LLM_MODEL      - nama model, contoh: llama-3.3-70b-versatile
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Singleton client, dibuat sekali dari env var."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
    return _client


def call_structured(system_prompt: str, user_prompt: str, schema: dict) -> dict:
    """Panggil LLM dan paksa output JSON sesuai schema (JSON mode)."""
    client = get_client()
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt + "\n\nBalas HANYA dengan JSON valid sesuai schema berikut:\n" + json.dumps(schema)},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return json.loads(content)
