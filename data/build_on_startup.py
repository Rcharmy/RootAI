"""
data/build_on_startup.py

Ensure the DuckDB file exists at config.duckdb_path. Idempotent: if the
file is already present with rows, does nothing. Otherwise downloads the
Olist CSVs from Kaggle and rebuilds via data.setup_data.

Called from streamlit_app.py on module import so that Streamlit Community
Cloud deployments can bootstrap the data on first hit after container
start.

Local development uses `python data/setup_data.py` directly and this script
is a no-op.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb


def _log(msg: str) -> None:
    print(f"[build_on_startup] {msg}", flush=True)


def _has_data(duckdb_path: Path) -> bool:
    """Return True if the DuckDB file exists AND has the denorm view populated."""
    if not duckdb_path.exists():
        return False
    try:
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            count = con.execute("SELECT COUNT(*) FROM order_items_denorm").fetchone()[0]
            return count > 0
        finally:
            con.close()
    except Exception:
        return False


def _write_kaggle_credentials_if_present() -> bool:
    """
    Read KAGGLE_USERNAME and KAGGLE_KEY from Streamlit secrets or env vars,
    write ~/.kaggle/kaggle.json so the kaggle CLI can authenticate.
    Returns True if credentials were written.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    creds_path = kaggle_dir / "kaggle.json"

    if creds_path.exists():
        return True

    # Try st.secrets (only available inside Streamlit runtime)
    username = None
    key = None
    try:
        import streamlit as st
        if "kaggle" in st.secrets:
            username = st.secrets["kaggle"].get("username")
            key = st.secrets["kaggle"].get("key")
    except Exception:
        pass

    # Fallback: environment variables
    username = username or os.environ.get("KAGGLE_USERNAME")
    key = key or os.environ.get("KAGGLE_KEY")

    if not username or not key:
        return False

    import json
    creds_path.write_text(json.dumps({"username": username, "key": key}))
    # kaggle CLI insists on chmod 600 on the file
    try:
        os.chmod(creds_path, 0o600)
    except Exception:
        pass
    return True


def ensure_data_ready() -> None:
    """Public entry point. Called from streamlit_app.py."""
    from rootai.config import config

    duckdb_path = Path(config.duckdb_path)
    if _has_data(duckdb_path):
        _log(f"DuckDB present at {duckdb_path}. Skipping build.")
        return

    _log("DuckDB missing or empty. Building from Olist CSVs.")

    raw_dir = config.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Do we already have the raw CSVs?
    csvs = list(raw_dir.glob("*.csv"))
    if len(csvs) < 9:
        _log("Raw CSVs missing. Attempting Kaggle download.")
        creds_written = _write_kaggle_credentials_if_present()
        if not creds_written:
            raise RuntimeError(
                "Cannot download Olist dataset: no Kaggle credentials found. "
                "In Streamlit Cloud, add a [kaggle] section to secrets with "
                "username and key. Locally, ensure data/raw/ contains the 9 Olist CSVs."
            )

        subprocess.check_call([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "olistbr/brazilian-ecommerce",
            "-p", str(raw_dir),
            "--unzip",
        ])
        _log("Kaggle download complete.")

    # Run the setup script
    setup_script = Path(__file__).parent / "setup_data.py"
    _log(f"Running {setup_script}...")
    subprocess.check_call([sys.executable, str(setup_script)])
    _log("Data build complete.")


if __name__ == "__main__":
    ensure_data_ready()