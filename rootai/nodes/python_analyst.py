"""
rootai/nodes/python_analyst.py

Python Analyst node stub. In Phase 3 this becomes the node that runs
contribution analysis, top-K decomposition, or cohort compare on the
DataFrames the SQL Explorer produced.

For Phase 2 the stub emits a canned PythonAnalysis record.
"""
from __future__ import annotations

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    NodeName,
    PythonAnalysis,
)


def python_analyst_node(state: InvestigationState) -> dict:
    """Stub: emit a canned contribution-analysis record."""
    print(f"STUB: python_analyst node called (step {state.current_step + 1})")

    stub_analysis = PythonAnalysis(
        step=state.current_step + 1,
        tool_name="contribution_analysis",
        tool_args={"dimension": "product_category_english", "top_k": 5},
        rationale="STUB: computing category contribution to revenue delta",
        result_summary="STUB: top category contributes 45% of decline",
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.PYTHON_ANALYST,
        action="stub_contribution_analysis",
        input_summary="dataframe of category revenue",
        output_summary=stub_analysis.result_summary,
    )

    return {
        "python_analyses": [stub_analysis],
        "current_step": state.current_step + 1,
        "current_node": NodeName.PYTHON_ANALYST,
        "action_log": [log_entry],
    }