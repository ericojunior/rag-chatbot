from __future__ import annotations

import os
from typing import List, Optional


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_with_openai(*, question: str, contexts: List[str]) -> str:
    """
    Gera uma resposta baseada nos trechos recuperados.
    Requer OPENAI_API_KEY.
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    joined = "\n\n".join([f"[Trecho {i+1}]\n{c}" for i, c in enumerate(contexts)])

    system = (
        "Você é um assistente que responde perguntas com base em um documento fornecido. "
        "Use APENAS as informações presentes nos trechos. "
        "Se não houver evidência suficiente, diga claramente que não encontrou no documento."
    )

    user = (
        f"Pergunta: {question}\n\n"
        f"Trechos do documento:\n{joined}\n\n"
        "Responda em português, de forma objetiva, e cite quais trechos você usou (ex.: Trecho 1, 3)."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return (resp.choices[0].message.content or "").strip()


def fallback_answer(*, question: str, contexts: List[str]) -> str:
    """
    Modo sem LLM: retorna os trechos mais relevantes para o usuário ler.
    """
    if not contexts:
        return "Não encontrei trechos relevantes no documento para essa pergunta."
    lines = [
        "Não há LLM configurado (sem OPENAI_API_KEY).",
        "Abaixo estão os trechos mais relevantes para você consultar:",
        "",
    ]
    for i, c in enumerate(contexts, start=1):
        lines.append(f"- Trecho {i}: {c[:400]}{'...' if len(c) > 400 else ''}")
    return "\n".join(lines)

