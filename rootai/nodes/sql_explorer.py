"""
rootai/nodes/sql_explorer.py

SQL Explorer node. Generates one targeted SQL query per hop, executes
it against the read-only DuckDB connection, and records the result
in state.sql_queries.

Design:
- Structured output: LLM returns {sql, rationale, dimension_being_tested}.
  Rationale is required and shows in the trace and final brief.
- Only order_items_denorm is queryable. Prompt lists the exact columns.
- SQL safety check (rootai/guardrails/sql_safety) runs before execution.
  Rejection paths land in state.sql_queries with passed_guardrails=False
  so the agent can see what was tried and course-correct.
- On query error, we return the SQLQuery record with error field set;
  the graph continues. The Router will notice repeated failures.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from rootai.guardrails.sql_safety import check_sql_safety
from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    NodeName,
    SQLQuery,
)
from rootai.tools.db import run_query
from rootai.tools.llm import get_structured_llm


class ExplorerOutput(BaseModel):
    """LLM-generated SQL query with reasoning."""
    sql: str = Field(description="A single SELECT or WITH...SELECT statement against order_items_denorm. Must end with a LIMIT clause if not already grouped/aggregated.")
    rationale: str = Field(description="1-2 sentence explanation of what this query tests and why now.")
    dimension_being_tested: str = Field(description="The primary dimension this query is slicing by (e.g. 'product_category_english', 'customer_state'). Use 'none' if the query is a whole-population aggregate.")


SYSTEM_PROMPT = (
    "You are the SQL Explorer for RootAI. Your job is to write ONE targeted SQL query per turn against DuckDB.\n\n"
    "The only queryable view is `order_items_denorm` at order_item grain (one row per line item in an order).\n\n"
    "HARD RULES:\n"
    "- Write ONLY a single SELECT statement (CTEs via WITH are fine). Never UNION, DROP, DELETE, INSERT, UPDATE, CREATE, ALTER.\n"
    "- Use only columns that exist in the view (listed in the user message).\n"
    "- Time filter must use `order_purchase_timestamp`.\n\n"
    "COMPARISON PATTERN (CRITICAL): When comparing a baseline window and a comparison window (which is nearly every query in this project), you MUST produce a single result set with one row per dimension value and separate columns per window. Use conditional aggregation, NOT UNION.\n\n"
    "CORRECT pattern:\n"
    "  SELECT dim,\n"
    "         SUM(CASE WHEN order_purchase_timestamp BETWEEN 'BASE_START' AND 'BASE_END' THEN metric END) AS baseline,\n"
    "         SUM(CASE WHEN order_purchase_timestamp BETWEEN 'COMP_START' AND 'COMP_END' THEN metric END) AS comparison\n"
    "  FROM order_items_denorm\n"
    "  WHERE order_purchase_timestamp BETWEEN 'BASE_START' AND 'COMP_END'\n"
    "  GROUP BY dim\n"
    "  ORDER BY (comparison - baseline) DESC\n"
    "  LIMIT 20\n\n"
    "INCORRECT pattern (do not use): SELECT dim, SUM(metric) FROM view WHERE ts BETWEEN base UNION ALL SELECT dim, SUM(metric) FROM view WHERE ts BETWEEN comp. This loses which window each row came from.\n\n"
    "OTHER RULES:\n"
    "- Always end with LIMIT 50 unless already aggregated to a small result set.\n"
    "- Return rationale in 1-2 sentences naming the dimension you're slicing and the hypothesis it tests.\n"
    "- Do NOT repeat a previous query verbatim. If a previous query returned zero rows or errored, choose a different dimension or a broader filter."
)

USER_TEMPLATE = (
    "Investigation question: {question}\n"
    "KPI: {kpi_name} ({direction}), magnitude: {magnitude}\n"
    "Comparison window: {comp_start} to {comp_end}\n"
    "Baseline window: {base_start} to {base_end}\n\n"
    "Plan: {plan}\n\n"
    "Available columns in order_items_denorm:\n{columns}\n\n"
    "Previous queries this investigation ({n_prev}):\n{prev_queries}\n\n"
    "Existing hypotheses ({n_hyp}):\n{hyp_list}\n\n"
    "Dead ends (dimensions already ruled out): {dead_ends}\n\n"
    "Write the next SQL query to advance the investigation. Return the SQL, rationale, and dimension_being_tested."
)


def _summarize_prev_queries(state: InvestigationState) -> str:
    if not state.sql_queries:
        return "(none)"
    lines = []
    for q in state.sql_queries[-5:]:  # last 5 max
        status = "ERROR" if q.error else f"{q.row_count} rows"
        preview = q.query.replace("\n", " ")[:100]
        lines.append(f"- [{status}] {preview}")
    return "\n".join(lines)


def _summarize_hypotheses(state: InvestigationState) -> str:
    if not state.hypotheses:
        return "(none)"
    lines = []
    for h in state.hypotheses:
        lines.append(f"- {h.id} [{h.status.value}, conf={h.confidence:.2f}]: {h.statement[:120]}")
    return "\n".join(lines)


def _fallback_query(state: InvestigationState, error_msg: str) -> ExplorerOutput:
    """Safe fallback if the LLM call fails: query by product category as a baseline slice."""
    sq = state.structured_question
    comp_end = sq.time_window.get("end", "2018-06-30") if sq else "2018-06-30"
    comp_start = sq.time_window.get("start", "2018-04-01") if sq else "2018-04-01"
    fallback_sql = (
        f"SELECT product_category_english, SUM(price) AS revenue, COUNT(*) AS line_count "
        f"FROM order_items_denorm "
        f"WHERE order_purchase_timestamp BETWEEN '{comp_start}' AND '{comp_end}' "
        f"GROUP BY 1 ORDER BY revenue DESC LIMIT 20"
    )
    return ExplorerOutput(
        sql=fallback_sql,
        rationale=f"FALLBACK (LLM failed: {error_msg[:80]}). Baseline slice by product_category_english.",
        dimension_being_tested="product_category_english",
    )


def sql_explorer_node(state: InvestigationState) -> dict:
    """Generate and execute one SQL query."""
    step = state.current_step + 1
    print(f"sql_explorer: generating query (step {step})")

    sq = state.structured_question
    columns_str = ", ".join(state.dataset.tables.get("order_items_denorm", []))
    magnitude_str = f"{sq.magnitude_pct}%" if sq and sq.magnitude_pct is not None else "unknown"

    user_msg = USER_TEMPLATE.format(
        question=state.original_question,
        kpi_name=sq.kpi_name if sq else "unknown",
        direction=sq.direction if sq else "unknown",
        magnitude=magnitude_str,
        comp_start=sq.time_window.get("start", "?") if sq else "?",
        comp_end=sq.time_window.get("end", "?") if sq else "?",
        base_start=sq.comparison_window.get("start", "?") if sq else "?",
        base_end=sq.comparison_window.get("end", "?") if sq else "?",
        plan=state.plan or "(no plan set)",
        columns=columns_str,
        n_prev=len(state.sql_queries),
        prev_queries=_summarize_prev_queries(state),
        n_hyp=len(state.hypotheses),
        hyp_list=_summarize_hypotheses(state),
        dead_ends=", ".join(state.dead_ends) if state.dead_ends else "(none)",
    )

    llm = get_structured_llm(ExplorerOutput)

    try:
        output: ExplorerOutput = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except (ValidationError, Exception) as e:
        print(f"  sql_explorer LLM call failed: {e}. Using fallback.")
        output = _fallback_query(state, str(e))

    # Safety check before execution
    safety = check_sql_safety(output.sql)

    if not safety.passed:
        print(f"  sql rejected by safety check: {safety.reason}")
        rejected_query = SQLQuery(
            step=step,
            query=output.sql,
            rationale=f"REJECTED: {safety.reason}. Original rationale: {output.rationale}",
            row_count=0,
            columns=[],
            error=f"safety check failed: {safety.reason}",
            passed_guardrails=False,
        )
        log_entry = ActionLogEntry(
            step=step,
            node=NodeName.SQL_EXPLORER,
            action="sql_rejected",
            input_summary=output.sql[:120],
            output_summary=f"rejected: {safety.reason}",
        )
        return {
            "sql_queries": [rejected_query],
            "current_step": step,
            "current_node": NodeName.SQL_EXPLORER,
            "action_log": [log_entry],
        }

    # Execute
    df, qr = run_query(output.sql)
    print(f"  query returned {qr.row_count} rows in {qr.duration_ms}ms" + (f", error: {qr.error}" if qr.error else ""))

    executed_query = SQLQuery(
        step=step,
        query=output.sql,
        rationale=output.rationale,
        row_count=qr.row_count,
        columns=qr.columns,
        result_preview=qr.preview_markdown,
        error=qr.error,
        duration_ms=qr.duration_ms,
        passed_guardrails=True,
    )

    log_entry = ActionLogEntry(
        step=step,
        node=NodeName.SQL_EXPLORER,
        action="sql_executed",
        input_summary=f"dimension={output.dimension_being_tested}",
        output_summary=(qr.error[:120] if qr.error else f"{qr.row_count} rows, {qr.columns}"),
    )

    return {
        "sql_queries": [executed_query],
        "current_step": step,
        "current_node": NodeName.SQL_EXPLORER,
        "action_log": [log_entry],
    }