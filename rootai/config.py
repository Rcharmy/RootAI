"""
rootai/config.py

Central configuration for RootAI. Reads environment variables from .env,
validates them, and exposes a single Config object every module imports.

Design note: config is loaded once at import time. If .env changes, restart
the process. This is deliberate to avoid stale-config bugs across long
investigations.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root. Path is relative to this file.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class Config(BaseModel):
    """Runtime configuration, loaded once at import."""

    # LLM
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # Budgets (must match BudgetTracker defaults in state.py)
    max_steps: int = 15
    max_cost_usd: float = 0.10
    max_wall_seconds: int = 180

    # Paths (absolute, resolved from project root)
    project_root: Path
    duckdb_path: Path
    chroma_path: Path
    traces_dir: Path
    raw_data_dir: Path


def _load_config() -> Config:
    """Read env vars and construct Config. Fails loudly on missing required vars."""
    root = Path(__file__).resolve().parent.parent

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("TODO"):
        raise RuntimeError(
            "GROQ_API_KEY not set in .env. See .env.example for the required format."
        )

    return Config(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        max_steps=int(os.getenv("ROOTAI_MAX_STEPS", "15")),
        max_cost_usd=float(os.getenv("ROOTAI_MAX_COST_USD", "0.10")),
        max_wall_seconds=int(os.getenv("ROOTAI_MAX_WALL_SECONDS", "180")),
        project_root=root,
        duckdb_path=root / os.getenv("ROOTAI_DUCKDB_PATH", "data/processed/olist.duckdb"),
        chroma_path=root / os.getenv("ROOTAI_CHROMA_PATH", "data/chroma"),
        traces_dir=root / os.getenv("ROOTAI_TRACES_DIR", "traces"),
        raw_data_dir=root / "data" / "raw",
    )


# Singleton loaded at import. Every module does `from rootai.config import config`.
config: Config = _load_config()