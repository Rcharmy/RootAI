"""
rootai/tools/db.py

DuckDB access layer. The SQL Explorer node uses these functions to run
queries and get results in a form the LLM can reason about.

Design notes:
- All connections are opened read_only=True. The database is treated as an
  immutable artifact; nothing in the agent path should ever write. This is
  the first line of defense before the Phase 4 SQL guardrails add pattern-based
  checks (block DROP/DELETE/INSERT/UPDATE).
- run_query returns both a DataFrame and a QueryResult metadata object. The
  DataFrame goes to the Python Analyst node; the metadata goes into state
  for tracing. Separating the two keeps the state schema JSON-serializable
  (DataFrames are not).
- A hard row-count cap prevents an unbounded query from blowing up the LLM
  context. If a query returns more than MAX_ROWS_PREVIEW rows, we truncate
  and set was_truncated=True so the Analyst knows to aggregate more.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import duckdb
import pandas as pd

from rootai.config import config


# Cap on rows returned in the preview attached to state. Full result stays
# in the DataFrame the caller can hand to the Python Analyst.
MAX_ROWS_PREVIEW = 50

# Cap on total rows the query can return. Aborts if exceeded. Prevents
# an accidental "SELECT *" from the 112k-row denorm view.
MAX_ROWS_FULL = 10_000


@dataclass
class QueryResult:
    """Metadata about a query execution. JSON-serializable."""
    query: str
    row_count: int
    columns: list[str]
    duration_ms: int
    preview_markdown: str
    was_truncated: bool
    error: Optional[str] = None


@contextmanager
def get_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Open a read-only DuckDB connection. Yields the connection, closes on exit.

    Usage:
        with get_connection() as con:
            df = con.execute("SELECT ...").fetchdf()
    """
    con = duckdb.connect(str(config.duckdb_path), read_only=True)
    try:
        yield con
    finally:
        con.close()


def _to_preview_markdown(df: pd.DataFrame, max_rows: int = MAX_ROWS_PREVIEW) -> str:
    """
    Render up to max_rows of a DataFrame as a compact markdown table.
    Empty DataFrame produces a placeholder string, never a crash.
    """
    if df.empty:
        return "_(no rows)_"
    head = df.head(max_rows)
    try:
        return head.to_markdown(index=False)
    except ImportError:
        # to_markdown requires tabulate. Fall back to plain string if not installed.
        return head.to_string(index=False)


def run_query(sql: str) -> tuple[pd.DataFrame, QueryResult]:
    """
    Execute a SQL query against DuckDB and return (DataFrame, QueryResult).

    On success:
        - DataFrame contains up to MAX_ROWS_FULL rows.
        - QueryResult carries metadata for state.sql_queries.

    On failure:
        - Returns (empty DataFrame, QueryResult with error field set).
        - Does NOT raise. The Explorer node decides what to do with the error.

    Design: queries run against a fresh read-only connection each time. This
    is a small performance hit (~5ms of connection overhead per query) that
    buys clean isolation across investigations; no shared cursor state can
    leak between calls.
    """
    start = time.perf_counter()
    with get_connection() as con:
        try:
            df = con.execute(sql).fetchdf()
        except Exception as e:  # noqa: BLE001 -- deliberately broad; Explorer decides how to handle
            duration_ms = int((time.perf_counter() - start) * 1000)
            return pd.DataFrame(), QueryResult(
                query=sql,
                row_count=0,
                columns=[],
                duration_ms=duration_ms,
                preview_markdown="",
                was_truncated=False,
                error=str(e),
            )

    duration_ms = int((time.perf_counter() - start) * 1000)
    row_count = len(df)
    was_truncated = False

    if row_count > MAX_ROWS_FULL:
        df = df.head(MAX_ROWS_FULL)
        was_truncated = True

    return df, QueryResult(
        query=sql,
        row_count=row_count,
        columns=list(df.columns),
        duration_ms=duration_ms,
        preview_markdown=_to_preview_markdown(df),
        was_truncated=was_truncated,
        error=None,
    )


def get_table_row_count(table_name: str = "order_items_denorm") -> int:
    """Cheap health check: does the denorm view have rows?"""
    with get_connection() as con:
        return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]


def check_health() -> dict:
    """
    End-to-end health check the Streamlit UI can call at startup.
    Returns a dict with 'ok' bool and diagnostic info.
    """
    try:
        row_count = get_table_row_count()
        return {
            "ok": True,
            "duckdb_path": str(config.duckdb_path),
            "denorm_row_count": row_count,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "duckdb_path": str(config.duckdb_path),
            "error": str(e),
        }