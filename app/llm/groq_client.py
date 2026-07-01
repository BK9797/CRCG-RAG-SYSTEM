"""Groq LLM client factory.

Centralizes ChatGroq instantiation so configuration (model, temperature,
max tokens) is defined in one place and sourced from Settings.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_groq_llm(
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 600,
) -> ChatGroq:
    """Create a configured ChatGroq instance.

    Args:
        api_key: Groq API key.
        model: Groq model identifier, e.g. "llama-3.1-70b-versatile".
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the generated response.

    Returns:
        A configured ChatGroq instance.
    """
    logger.info("Initializing Groq LLM: model=%s, temperature=%.2f", model, temperature)
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
