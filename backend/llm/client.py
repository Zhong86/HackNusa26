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

from openai import OpenAI, OpenAIError

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Singleton client, dibuat sekali dari env var. Retry otomatis untuk error 429/5xx."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            max_retries=3,
            timeout=20.0,
        )
    return _client


class LLMCallError(RuntimeError):
    """Dilempar kalau LLM gagal dipanggil atau balas JSON yang tidak valid, setelah retry habis."""


def call_structured(system_prompt: str, user_prompt: str, schema: dict) -> dict:
    """Panggil LLM dan paksa output JSON sesuai schema (JSON mode)."""
    client = get_client()
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt + "\n\nBalas HANYA dengan JSON valid sesuai schema berikut:\n" + json.dumps(schema)},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as e:
        raise LLMCallError(f"Panggilan LLM gagal: {e}") from e

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMCallError(f"LLM membalas JSON tidak valid: {content!r}") from e
