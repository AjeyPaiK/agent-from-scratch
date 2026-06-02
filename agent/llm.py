"""Shared Ollama LLM factory for agent nodes."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


def build_chat_llm(*, temperature: float = 0) -> ChatOllama:
    """Return a configured ``ChatOllama`` instance.

    Parameters
    ----------
    temperature : float, optional
        Sampling temperature passed to Ollama. Defaults to ``0``.

    Returns
    -------
    ChatOllama
        LangChain chat model bound to project Ollama settings.
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )
