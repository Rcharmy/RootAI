"""
rootai/config.py

Central configuration for RootAI. Reads environment variables from .env,
falls back to os.environ, then falls back to st.secrets if running under
Streamlit. Exposes a single Config object every module imports.

Design note: config is loaded once at import time. If .env or secrets
change, restart the process.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Resolve a config value from, in order:
    1. os.environ (populated by .env locally or by process env in the cloud).
    2. st.secrets (only available inside Streamlit runtime).
    Returns default if nothing matches.
    """
    val = os.environ.get(key)
    if val is not None:
        return val

    # Fall back to Streamlit secrets if available
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return default


class Config(BaseModel):
    """Runtime configuration."""

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    max_steps: int = 15
    max_cost_usd: float = 0.10
    max_wall_seconds: int = 180

    project_root: Path
    duckdb_path: Path
    chroma_path: Path
    traces_dir: Path
    raw_data_dir: Path


def _load_config() -> Config:
    """Read env vars and construct Config. Fails loudly on missing required vars."""
    root = Path(__file__).resolve().parent.parent

    api_key = _get_setting("GROQ_API_KEY")
    if not api_key or api_key.startswith("TODO"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Set it in .env locally, or in Streamlit "
            "Community Cloud secrets."
        )

    return Config(
        groq_api_key=api_key,
        groq_model=_get_setting("GROQ_MODEL", "llama-3.3-70b-versatile"),
        max_steps=int(_get_setting("ROOTAI_MAX_STEPS", "15")),
        max_cost_usd=float(_get_setting("ROOTAI_MAX_COST_USD", "0.10")),
        max_wall_seconds=int(_get_setting("ROOTAI_MAX_WALL_SECONDS", "180")),
        project_root=root,
        duckdb_path=root / _get_setting("ROOTAI_DUCKDB_PATH", "data/processed/olist.duckdb"),
        chroma_path=root / _get_setting("ROOTAI_CHROMA_PATH", "data/chroma"),
        traces_dir=root / _get_setting("ROOTAI_TRACES_DIR", "traces"),
        raw_data_dir=root / "data" / "raw",
    )


config: Config = _load_config()