"""
rootai/nodes/python_analyst.py

Python Analyst node. Consumes the most recent SQL result and runs one of
the restricted analytical tools on the DataFrame.

Design:
- LLM picks the tool NAME and arguments via structured output. It does
  NOT write Python code. This is the "restricted" mode of python_tool_mode.
- The last SQL query must have executed successfully (passed_guardrails,
  no error, row_count > 0). If not, this node emits an evidence-free
  no-op analysis and lets the Router decide what to do next.
- The chosen tool runs against the DataFrame we re-fetch from the SQL
  Explorer's stored query. Yes, re-executing the same SQL is wasteful.
  Phase 5+ optimization if we care.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    NodeName,
    PythonAnalysis,
)
from rootai.tools.analysis import TOOLS, TOOL_DESCRIPTIONS
from rootai.tools.db import run_query
from rootai.tools.llm import get_structured_llm


class AnalystChoice(BaseModel):
    """LLM's choice of tool and args."""
    tool_name: str = Field(description="One of: contribution_analysis, top_k_by_dimension, pct_change_summary.")
    tool_args: dict = Field(description="Args dict for the chosen tool. Keys must match the tool's signature exactly. Column names must match the SQL result columns exactly.")
    rationale: str = Field(description="1-2 sentence explanation of why this tool for this query.")


SYSTEM_PROMPT = (
    "You are the Python Analyst for RootAI. You pick ONE analytical tool per turn and run it on the most recent SQL result.\n\n"
    "Available tools:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
    + "\n\nRules:\n"
    + "- tool_name MUST be one of the three listed above. Do not invent tools.\n"
    + "- tool_args keys and value types MUST match the tool's signature exactly.\n"
    + "- Column names in tool_args MUST match the SQL result's actual column names exactly (case-sensitive).\n"
    + "- Pick contribution_analysis when the SQL result has baseline_* and comparison_* columns (a two-window comparison).\n"
    + "- Pick top_k_by_dimension when the SQL result is a single-window aggregate.\n"
    + "- Pick pct_change_summary when you want to find outliers by percent change rather than absolute delta."
)


USER_TEMPLATE = (
    "Investigation question: {question}\n"
    "Plan: {plan}\n\n"
    "Most recent SQL query:\n"
    "  Rationale: {sql_rationale}\n"
    "  Columns: {sql_columns}\n"
    "  Row count: {sql_row_count}\n"
    "  Preview:\n{sql_preview}\n\n"
    "Existing hypotheses ({n_hyp}):\n{hyp_list}\n\n"
    "Pick the tool and args to advance the investigation."
)


def _summarize_hypotheses(state: InvestigationState) -> str:
    if not state.hypotheses:
        return "(none)"
    return "\n".join(
        f"- {h.id} [{h.status.value}, conf={h.confidence:.2f}]: {h.statement[:120]}"
        for h in state.hypotheses
    )


def _no_op(state: InvestigationState, step: int, reason: str) -> dict:
    """Skip this hop if there is no usable SQL result."""
    print(f"  python_analyst: no-op ({reason})")
    analysis = PythonAnalysis(
        step=step,
        tool_name="skipped",
        tool_args={},
        rationale=f"skipped: {reason}",
        result_summary=None,
        error=reason,
    )
    log_entry = ActionLogEntry(
        step=step,
        node=NodeName.PYTHON_ANALYST,
        action="analysis_skipped",
        input_summary=reason[:120],
        output_summary="no-op",
    )
    return {
        "python_analyses": [analysis],
        "current_step": step,
        "current_node": NodeName.PYTHON_ANALYST,
        "action_log": [log_entry],
    }


def python_analyst_node(state: InvestigationState) -> dict:
    """Choose and run one analytical tool on the most recent SQL result."""
    step = state.current_step + 1
    print(f"python_analyst: choosing tool (step {step})")

    # Precondition: we need a successful, non-empty SQL result
    if not state.sql_queries:
        return _no_op(state, step, "no SQL query has been executed yet")

    last_sql = state.sql_queries[-1]
    if last_sql.error or last_sql.row_count == 0 or not last_sql.passed_guardrails:
        return _no_op(state, step, f"last SQL unusable: error={last_sql.error}, rows={last_sql.row_count}, passed={last_sql.passed_guardrails}")

    # Re-execute to get the DataFrame (state doesn't carry DFs; they aren't JSON-serializable)
    df, _ = run_query(last_sql.query)
    if df.empty:
        return _no_op(state, step, "re-executed SQL returned empty DataFrame")

    user_msg = USER_TEMPLATE.format(
        question=state.original_question,
        plan=state.plan or "(no plan set)",
        sql_rationale=last_sql.rationale,
        sql_columns=last_sql.columns,
        sql_row_count=last_sql.row_count,
        sql_preview=last_sql.result_preview or "(no preview)",
        n_hyp=len(state.hypotheses),
        hyp_list=_summarize_hypotheses(state),
    )

    llm = get_structured_llm(AnalystChoice)

    try:
        choice: AnalystChoice = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except (ValidationError, Exception) as e:
        print(f"  python_analyst LLM call failed: {e}")
        return _no_op(state, step, f"LLM choice failed: {e}")

    if choice.tool_name not in TOOLS:
        return _no_op(state, step, f"LLM picked unknown tool '{choice.tool_name}'. Available: {list(TOOLS.keys())}")

    tool_fn = TOOLS[choice.tool_name]

    try:
        result = tool_fn(df=df, **choice.tool_args)
    except TypeError as e:
        # Bad args for the tool
        print(f"  tool arg mismatch: {e}")
        analysis = PythonAnalysis(
            step=step,
            tool_name=choice.tool_name,
            tool_args=choice.tool_args,
            rationale=choice.rationale,
            error=f"arg mismatch: {e}",
        )
        log_entry = ActionLogEntry(
            step=step,
            node=NodeName.PYTHON_ANALYST,
            action="tool_arg_mismatch",
            input_summary=f"{choice.tool_name}({choice.tool_args})",
            output_summary=str(e)[:120],
        )
        return {
            "python_analyses": [analysis],
            "current_step": step,
            "current_node": NodeName.PYTHON_ANALYST,
            "action_log": [log_entry],
        }

    print(f"  {choice.tool_name} -> {result.summary[:120]}")

    analysis = PythonAnalysis(
        step=step,
        tool_name=choice.tool_name,
        tool_args=choice.tool_args,
        rationale=choice.rationale,
        result_summary=result.summary,
        error=result.error,
    )

    log_entry = ActionLogEntry(
        step=step,
        node=NodeName.PYTHON_ANALYST,
        action="analysis_run",
        input_summary=f"{choice.tool_name}({choice.tool_args})",
        output_summary=result.summary[:120] if result.summary else "(no summary)",
    )

    # We also stash the compact findings dict on the analysis so downstream
    # nodes (Hypothesis Former, Writer) can reason about specific numbers.
    # Pydantic PythonAnalysis has a dict-shaped result field via result_summary;
    # for structured findings we JSON-serialize them into the same field.
    # Cleaner: extend PythonAnalysis with a findings dict. Doing that inline.
    if result.findings:
        # Append compact findings JSON to result_summary so the LLM downstream can see them
        import json
        analysis = analysis.model_copy(update={
            "result_summary": (analysis.result_summary or "") + "\n\nFindings:\n" + json.dumps(result.findings, indent=2, default=str)[:1500]
        })

    return {
        "python_analyses": [analysis],
        "current_step": step,
        "current_node": NodeName.PYTHON_ANALYST,
        "action_log": [log_entry],
    }