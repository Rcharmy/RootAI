"""
rootai/nodes/sql_explorer.py

SQL Explorer node stub. In Phase 3 this becomes the node that generates a
SQL query based on the current plan, executes it against DuckDB via
rootai.tools.db.run_query, and appends the result to state.

For Phase 2 the stub emits a canned SQL string and a fake result preview.
"""
from __future__ import annotations

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    NodeName,
    SQLQuery,
)


def sql_explorer_node(state: InvestigationState) -> dict:
    """Stub: emit a canned SQL query record."""
    print(f"STUB: sql_explorer node called (step {state.current_step + 1})")

    stub_query = SQLQuery(
        step=state.current_step + 1,
        query=(
            "SELECT product_category_english, SUM(price) AS revenue "
            "FROM order_items_denorm "
            "WHERE order_purchase_timestamp BETWEEN '2018-04-01' AND '2018-06-30' "
            "GROUP BY 1 ORDER BY revenue DESC LIMIT 5"
        ),
        rationale="STUB: slicing by product_category_english to find top movers",
        row_count=5,
        columns=["product_category_english", "revenue"],
        result_preview="| product_category_english | revenue |\n|---|---|\n| STUB | 100.0 |",
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.SQL_EXPLORER,
        action="stub_sql_query",
        input_summary="plan: category slice",
        output_summary=f"stub query returning {stub_query.row_count} rows",
    )

    return {
        "sql_queries": [stub_query],
        "current_step": state.current_step + 1,
        "current_node": NodeName.SQL_EXPLORER,
        "action_log": [log_entry],
    }