"""
rootai/tools/llm.py

Central LLM client wrapper. Every node imports get_llm() rather than
constructing ChatGroq directly, so:
- model, temperature, and API key live in one place
- structured-output calls share one code path
- we can swap providers later by changing this module alone

Design notes:
- Two clients exposed: get_llm() for freeform prose (Writer, prompt-in-prompt-out),
  and get_structured_llm(schema) for JSON-schema-constrained outputs (Planner,
  Hypothesis Former, Router).
- Temperature 0.2, not 0.0. Analytical agents need a small amount of noise to
  produce plausible alternative hypotheses; 0.0 makes them collapse to a single
  greedy answer.
- Instructor-style structured output would be cleaner but adds a dependency.
  LangChain's with_structured_output() is good enough and already installed.
"""
from __future__ import annotations

from typing import Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import BaseModel

from rootai.config import config


T = TypeVar("T", bound=BaseModel)


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return a plain freeform ChatGroq client."""
    return ChatGroq(
        model=config.groq_model,
        api_key=config.groq_api_key,
        temperature=temperature,
        max_tokens=1024,
    )


def get_structured_llm(schema: Type[T], temperature: float = 0.2) -> BaseChatModel:
    """
    Return a ChatGroq client bound to a Pydantic schema. Its .invoke() will
    return an instance of `schema` (Pydantic model) rather than a string.

    Errors: if the LLM returns invalid JSON or a schema mismatch, LangChain
    raises pydantic.ValidationError. Nodes should catch this and fall back
    to a safe default rather than crash the investigation.
    """
    return get_llm(temperature=temperature).with_structured_output(schema)