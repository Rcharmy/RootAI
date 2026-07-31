"""
rootai/tools/llm.py

Central LLM client wrapper. Every node imports get_llm() rather than
constructing ChatGroq directly, so:
- model, temperature, and API key live in one place
- structured-output calls share one code path
- token usage and cost accumulate in one accounting layer
"""
from __future__ import annotations

from threading import Lock
from typing import Any, Type, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import BaseModel

from rootai.config import config


T = TypeVar("T", bound=BaseModel)


_MODEL_PRICING_USD_PER_1K = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
}


class _UsageAccumulator:
    """Thread-safe cumulative usage tracker."""
    def __init__(self):
        self._lock = Lock()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0

    def record(self, input_tokens: int, output_tokens: int, model: str) -> None:
        with self._lock:
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            pricing = _MODEL_PRICING_USD_PER_1K.get(
                model, _MODEL_PRICING_USD_PER_1K["llama-3.3-70b-versatile"]
            )
            call_cost = (input_tokens / 1000.0) * pricing["input"] + (output_tokens / 1000.0) * pricing["output"]
            self._total_cost_usd += call_cost
            self._call_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
                "total_tokens": self._total_input_tokens + self._total_output_tokens,
                "cost_usd": self._total_cost_usd,
                "call_count": self._call_count,
            }

    def reset(self) -> None:
        with self._lock:
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cost_usd = 0.0
            self._call_count = 0


usage = _UsageAccumulator()


class _UsageCallback(BaseCallbackHandler):
    """LangChain callback that records token usage after every LLM call."""

    def __init__(self, model: str):
        self.model = model

    def on_llm_end(self, response, **kwargs: Any) -> None:
        try:
            generations = response.generations
            if not generations or not generations[0]:
                return
            first = generations[0][0]
            message = getattr(first, "message", None)
            if message is None:
                return
            meta = getattr(message, "usage_metadata", None)
            if meta:
                usage.record(
                    input_tokens=int(meta.get("input_tokens", 0)),
                    output_tokens=int(meta.get("output_tokens", 0)),
                    model=self.model,
                )
        except Exception:
            pass


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return a plain freeform ChatGroq client with usage tracking wired in."""
    return ChatGroq(
        model=config.groq_model,
        api_key=config.groq_api_key,
        temperature=temperature,
        max_tokens=1024,
        callbacks=[_UsageCallback(model=config.groq_model)],
    )


def get_structured_llm(schema: Type[T], temperature: float = 0.2) -> BaseChatModel:
    """Return a ChatGroq client bound to a Pydantic schema."""
    return get_llm(temperature=temperature).with_structured_output(schema)


def get_current_usage() -> dict:
    """Public: snapshot of cumulative usage since last reset."""
    return usage.snapshot()


def reset_usage() -> None:
    """Public: reset accumulator, called at start of each investigation."""
    usage.reset()