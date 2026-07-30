"""
rootai/tools/dataset_context.py

Introspects the DuckDB database and builds a DatasetContext object that the
Planner node injects into prompts.

Design notes:
- Introspection at investigation start means schema changes flow into prompts
  automatically. No hardcoded column names in the LLM prompt layer.
- Distinguishes 'dimensions' (columns you slice by) from 'metrics' (columns
  you aggregate). This mirrors how a real BI tool models the same tables and
  gives the Planner useful structure without teaching it what every column is.
- The classification uses column names and DuckDB types. It is deliberate,
  not clever: numeric columns whose name suggests a measure become metrics,
  categorical or ID columns become dimensions. Hand-tuned overrides handle
  edge cases (order_id is numeric-looking but is an ID, not a metric).
"""
from __future__ import annotations

from typing import Iterable

import duckdb

from rootai.config import config
from rootai.state import DatasetContext


# The denormalized view is the ONLY table the agent queries. Raw tables exist
# but are hidden from the agent to keep the surface area small.
AGENT_VISIBLE_TABLES = ["order_items_denorm"]


# Columns whose *name* signals "this is an identifier, not a measure", even
# when DuckDB reports them as numeric. Prevents order_item_id being classified
# as a metric.
_ID_COLUMN_PATTERNS = (
    "_id",
    "_prefix",
)


# Columns whose name signals "this is a temporal anchor". Not dimensions in
# the sliceable sense; the agent uses them for windowing.
_TIME_COLUMN_PATTERNS = (
    "_date",
    "_timestamp",
    "_at",
)


# Columns that ARE metrics (real numbers you aggregate). Named explicitly
# because pattern-matching numerics would produce false positives.
_METRIC_COLUMNS = {
    "price",
    "freight_value",
    "line_value",
    "payment_value_total",
    "payment_installments_max",
    "review_score",
    "review_count",
    "delivery_days",
    "estimated_delivery_days",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
}


def _classify_column(name: str, duckdb_type: str) -> str:
    """Return 'dimension', 'metric', 'time', or 'skip'."""
    name_lower = name.lower()

    # Time first: order_purchase_timestamp is a datetime, we surface it separately
    if any(p in name_lower for p in _TIME_COLUMN_PATTERNS):
        return "time"

    # Explicit metric list wins over any pattern
    if name_lower in _METRIC_COLUMNS:
        return "metric"

    # IDs are dimensions the agent can group by but should not aggregate
    if any(name_lower.endswith(p) for p in _ID_COLUMN_PATTERNS):
        return "dimension"

    # Everything else numeric that isn't an ID or explicit metric: skip.
    # We prefer to omit an ambiguous column rather than misclassify it.
    type_upper = duckdb_type.upper()
    if any(t in type_upper for t in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "NUMERIC")):
        return "skip"

    # Text/varchar becomes a dimension
    if any(t in type_upper for t in ("VARCHAR", "TEXT", "CHAR", "BLOB")):
        return "dimension"

    # Fallback: skip unrecognized types (dates already handled above)
    return "skip"


def _get_view_columns(con: duckdb.DuckDBPyConnection, view_name: str) -> list[tuple[str, str]]:
    """Return list of (column_name, duckdb_type) for the given view."""
    rows = con.execute(f"DESCRIBE {view_name}").fetchall()
    # DESCRIBE returns (column_name, column_type, null, key, default, extra)
    return [(r[0], r[1]) for r in rows]


def build_dataset_context(
    duckdb_path: str | None = None,
    notes: str | None = None,
) -> DatasetContext:
    """
    Connect to DuckDB, introspect the denormalized view, and return a
    populated DatasetContext.

    Args:
        duckdb_path: Override path. Defaults to config.duckdb_path.
        notes: Optional freeform notes to embed in the context (e.g. known
            data quirks the agent should be aware of).

    Returns:
        DatasetContext with tables, dimensions, metrics, time_column, and grain set.

    Raises:
        RuntimeError if the denorm view is missing or empty.
    """
    db_path = duckdb_path or str(config.duckdb_path)
    con = duckdb.connect(db_path, read_only=True)

    try:
        tables: dict[str, list[str]] = {}
        dimensions: list[str] = []
        metrics: list[str] = []
        time_columns: list[str] = []

        for view_name in AGENT_VISIBLE_TABLES:
            cols = _get_view_columns(con, view_name)
            if not cols:
                raise RuntimeError(f"View {view_name} exists but has no columns")

            tables[view_name] = [c[0] for c in cols]

            for col_name, col_type in cols:
                kind = _classify_column(col_name, col_type)
                if kind == "dimension":
                    dimensions.append(col_name)
                elif kind == "metric":
                    metrics.append(col_name)
                elif kind == "time":
                    time_columns.append(col_name)

        # Sanity check: view must have rows
        row_count = con.execute(
            f"SELECT COUNT(*) FROM {AGENT_VISIBLE_TABLES[0]}"
        ).fetchone()[0]
        if row_count == 0:
            raise RuntimeError(f"{AGENT_VISIBLE_TABLES[0]} exists but is empty")

        # Pick the canonical time column: prefer order_purchase_timestamp if present
        time_column = "order_purchase_timestamp"
        if time_column not in time_columns:
            # Fallback: use the first time column we found, else keep the default
            time_column = time_columns[0] if time_columns else "order_purchase_timestamp"

        # Deterministic ordering makes prompts stable and diff-friendly
        dimensions.sort()
        metrics.sort()

        default_notes = (
            f"Denormalized view at order_item grain. {row_count:,} rows. "
            f"Data spans 2016-09 through 2018-10; September and October 2018 are "
            f"truncated, so like-for-like comparisons should end no later than "
            f"August 2018."
        )

        return DatasetContext(
            name="olist_ecommerce",
            tables=tables,
            dimensions=dimensions,
            metrics=metrics,
            time_column=time_column,
            grain="order_item",
            notes=notes or default_notes,
        )
    finally:
        con.close()