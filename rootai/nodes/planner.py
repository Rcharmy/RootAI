"""
rootai/nodes/planner.py

Planner node stub. In Phase 3 this becomes the LLM call that parses the
user's raw question into a structured KPIQuestion and produces the initial
investigation plan.

For Phase 2 (skeleton), it writes a deterministic stub plan and hardcoded
KPIQuestion so downstream nodes have realistic-looking inputs to work with.
"""
from __future__ import annotations

from datetime import datetime

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    InvestigationStatus,
    KPIQuestion,
    NodeName,
)


def planner_node(state: InvestigationState) -> dict:
    """Stub: emit a deterministic plan and structured question."""
    print(f"STUB: planner node called (step {state.current_step + 1})")

    stub_question = KPIQuestion(
        kpi_name="revenue",
        direction="down",
        magnitude_pct=-12.0,
        time_window={"start": "2018-04-01", "end": "2018-06-30"},
        comparison_window={"start": "2017-04-01", "end": "2017-06-30"},
        grain="quarterly",
        raw_question=state.original_question,
    )

    stub_plan = (
        "STUB PLAN: 1) slice by product_category_english, "
        "2) slice by customer_state, "
        "3) form hypotheses on top movers, "
        "4) test with contribution analysis, "
        "5) write brief."
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.PLANNER,
        action="stub_planning",
        input_summary=state.original_question[:120],
        output_summary=stub_plan[:120],
    )

    return {
        "status": InvestigationStatus.RUNNING,
        "structured_question": stub_question,
        "plan": stub_plan,
        "current_step": state.current_step + 1,
        "current_node": NodeName.PLANNER,
        "action_log": [log_entry],
    }