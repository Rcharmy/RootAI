"""
rootai/nodes/planner.py

Planner node. First node in every investigation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    InvestigationStatus,
    KPIQuestion,
    NodeName,
)
from rootai.tools.llm import get_structured_llm


class PlannerOutput(BaseModel):
    """Schema the LLM produces."""
    kpi_name: str = Field(description="The KPI being investigated, e.g. 'revenue', 'aov'.")
    direction: str = Field(description="'up', 'down', or 'unknown'.")
    magnitude_pct: float | None = Field(default=None, description="Signed percent, e.g. -12.0 for a 12% drop. Null if not stated.")
    time_window_start: str = Field(description="ISO date, comparison period start.")
    time_window_end: str = Field(description="ISO date, comparison period end.")
    baseline_window_start: str = Field(description="ISO date, baseline period start.")
    baseline_window_end: str = Field(description="ISO date, baseline period end.")
    grain: str = Field(description="'quarterly', 'monthly', 'weekly', or 'daily'.")
    plan: str = Field(description="2-4 sentence plan naming specific dimensions from the dataset context to slice first.")
    needs_clarification: bool = Field(default=False, description="True only if the question is genuinely ambiguous.")
    clarification_question: str | None = Field(default=None, description="If needs_clarification, a single specific question to ask.")


SYSTEM_PROMPT = (
    "You are the Planner for RootAI, an autonomous analytics agent that investigates why business KPIs moved.\n\n"
    "Your job on step 0:\n"
    "1. Parse the user's question into a structured KPIQuestion.\n"
    "2. Produce a 2-4 sentence plan naming dimensions from the dataset context you will slice first.\n\n"
    "Rules:\n"
    "- Only use dimensions that appear in the provided dataset context. Do not invent column names.\n"
    "- If the question does not specify a time window, INFER a sensible one. Only set needs_clarification=true if truly ambiguous.\n"
    "- For like-for-like comparisons, prefer H1 vs H1, Q2 vs Q2. Olist data ends August 2018 cleanly; do not use windows ending in September or October 2018.\n"
    "- Keep the plan concrete and dimension-first."
)


USER_TEMPLATE = (
    "Dataset context:\n"
    "- Name: {dataset_name}\n"
    "- Grain: {grain}\n"
    "- Dimensions available: {dimensions}\n"
    "- Metrics available: {metrics}\n"
    "- Time column: {time_column}\n"
    "- Notes: {notes}\n\n"
    "User question: {question}\n\n"
    "Parse the question and produce your plan."
)


def _fallback_output(question: str, error_msg: str) -> PlannerOutput:
    return PlannerOutput(
        kpi_name="unknown",
        direction="unknown",
        magnitude_pct=None,
        time_window_start="2018-01-01",
        time_window_end="2018-06-30",
        baseline_window_start="2017-01-01",
        baseline_window_end="2017-06-30",
        grain="quarterly",
        plan=f"FALLBACK PLAN (planner LLM failed: {error_msg[:80]}). Slice by product_category_english, then customer_state.",
        needs_clarification=False,
    )


def planner_node(state: InvestigationState) -> dict:
    """Real Planner: LLM structured output, grounded in DatasetContext."""
    print(f"planner: parsing question and planning investigation (step {state.current_step + 1})")

    user_msg = USER_TEMPLATE.format(
        dataset_name=state.dataset.name,
        grain=state.dataset.grain,
        dimensions=", ".join(state.dataset.dimensions),
        metrics=", ".join(state.dataset.metrics),
        time_column=state.dataset.time_column,
        notes=state.dataset.notes or "(none)",
        question=state.original_question,
    )

    llm = get_structured_llm(PlannerOutput)

    try:
        output: PlannerOutput = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except (ValidationError, Exception) as e:
        print(f"  planner LLM call failed: {e}. Using fallback.")
        output = _fallback_output(state.original_question, str(e))

    structured_question = KPIQuestion(
        kpi_name=output.kpi_name,
        direction=output.direction if output.direction in ("up", "down", "unknown") else "unknown",
        magnitude_pct=output.magnitude_pct,
        time_window={"start": output.time_window_start, "end": output.time_window_end},
        comparison_window={"start": output.baseline_window_start, "end": output.baseline_window_end},
        grain=output.grain,
        raw_question=state.original_question,
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.PLANNER,
        action="parse_and_plan",
        input_summary=state.original_question[:120],
        output_summary=f"{output.kpi_name} {output.direction} {output.magnitude_pct}%; plan: {output.plan[:80]}",
    )

    new_status = (
        InvestigationStatus.NEEDS_CLARIFICATION
        if output.needs_clarification
        else InvestigationStatus.RUNNING
    )

    return {
        "status": new_status,
        "structured_question": structured_question,
        "plan": output.plan,
        "needs_clarification": output.needs_clarification,
        "clarification_question": output.clarification_question,
        "current_step": state.current_step + 1,
        "current_node": NodeName.PLANNER,
        "action_log": [log_entry],
    }