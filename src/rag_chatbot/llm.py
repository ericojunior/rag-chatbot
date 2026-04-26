from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    api_key: str
    base_url: Optional[str]
    model: str


def _get_llm_config() -> Optional[LlmConfig]:
    """
    Resolve configuração do provedor de LLM.

    Prioridade:
    1) Novo padrão (recomendado):
       - LLM_PROVIDER=deepseek|openai
       - LLM_API_KEY=...
       - LLM_BASE_URL=...
       - LLM_MODEL=...
    2) Compatibilidade (legado):
       - OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    """
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or "").strip() or None
    model = (os.getenv("LLM_MODEL") or "").strip()

    # Modo legado (OpenAI)
    if not provider and not api_key:
        legacy_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not legacy_key:
            return None
        return LlmConfig(
            provider="openai",
            api_key=legacy_key,
            base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
            model=(os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
        )

    # Novo modo configurável
    if not provider:
        provider = "openai"

    if not api_key:
        # Conveniência: aceitar DEEPSEEK_API_KEY se LLM_PROVIDER=deepseek
        if provider == "deepseek":
            api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        elif provider == "openai":
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if not api_key:
        return None

    if provider == "deepseek":
        # DeepSeek costuma ser compatível com o SDK OpenAI usando base_url custom
        if not base_url:
            base_url = "https://api.deepseek.com/v1"
        if not model:
            model = "deepseek-chat"
    else:
        if not model:
            model = "gpt-4o-mini"

    return LlmConfig(provider=provider, api_key=api_key, base_url=base_url, model=model)


def has_llm() -> bool:
    return _get_llm_config() is not None


def generate_answer(*, question: str, contexts: List[str]) -> str:
    """
    Gera uma resposta baseada nos trechos recuperados usando o provedor configurado.
    Atualmente usa o SDK `openai` apontando para `base_url` (compatível com DeepSeek).
    """
    cfg = _get_llm_config()
    if cfg is None:
        raise RuntimeError("LLM não configurado (API key ausente).")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

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
        model=cfg.model,
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
        "Não há LLM configurado (sem API key).",
        "Abaixo estão os trechos mais relevantes para você consultar:",
        "",
    ]
    for i, c in enumerate(contexts, start=1):
        lines.append(f"- Trecho {i}: {c[:400]}{'...' if len(c) > 400 else ''}")
    return "\n".join(lines)
