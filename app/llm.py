"""Single OpenAI-compatible client factory.

The whole app talks to one client; switching between Groq, OpenRouter,
Ollama, the Qwen-on-Kaggle zrok tunnel, or OpenAI is purely a .env change.
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import LLM_API_KEY, resolve_llm


@lru_cache(maxsize=1)
def get_client() -> tuple[OpenAI, str]:
    base_url, model = resolve_llm()
    client = OpenAI(base_url=base_url, api_key=LLM_API_KEY or "not-needed")
    return client, model


def chat(system: str, user: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    """One-shot completion; raises on transport/API errors (caller retries or fails)."""
    client, model = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
