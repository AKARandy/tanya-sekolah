"""Final answer synthesis: evidence -> natural-language reply in the user's language."""
from __future__ import annotations

from app.llm import chat

SYNTH_SYSTEM = """You answer questions about Indonesian schools (Surabaya & Sidoarjo) using \
ONLY the evidence provided. Reply in the same language the user used (usually Indonesian). \
Be concrete: cite school names and NPSN ids from the evidence, give numbers when the evidence \
has them. If the evidence is insufficient, say so and suggest what to ask instead. \
Keep answers under 150 words. Do not invent schools, numbers, or fees."""


def synthesize_with_sql(question: str, sql: str, rows: list[dict]) -> str:
    evidence = f"SQL executed: {sql}\nResult rows (JSON): {rows}" if rows else "The SQL query returned 0 rows."
    return chat(
        SYNTH_SYSTEM,
        f"Question: {question}\n\nEvidence:\n{evidence}\n\nWrite the final answer.",
        max_tokens=600,
    )


def synthesize_with_semantic(question: str, hits: list[dict]) -> str:
    cards = "\n\n".join(f"[{h['npsn']}] {h['card']}" for h in hits) or "No matching schools found."
    return chat(
        SYNTH_SYSTEM,
        f"Question: {question}\n\nRetrieved school profiles:\n{cards}\n\nWrite the final answer.",
        max_tokens=600,
    )
