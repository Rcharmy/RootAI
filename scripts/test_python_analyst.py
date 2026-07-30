"""
scripts/test_python_analyst.py

Exercise the real Python Analyst node in isolation. Feed it a synthetic
state with a real SQL query already run, so it has something to analyze.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.nodes.python_analyst import python_analyst_node
from rootai.nodes.sql_explorer import sql_explorer_node
from rootai.state import (
    InvestigationState,
    InvestigationStatus,
    KPIQuestion,
)
from rootai.tools.dataset_context import build_dataset_context


def main():
    dataset = build_dataset_context()

    sq = KPIQuestion(
        kpi_name="revenue",
        direction="up",
        magnitude_pct=None,
        time_window={"start": "2018-04-01", "end": "2018-06-30"},
        comparison_window={"start": "2017-04-01", "end": "2017-06-30"},
        grain="quarterly",
        raw_question="Revenue in Q2 2018 was higher than Q2 2017. What drove the growth?",
    )

    state = InvestigationState(
        original_question=sq.raw_question,
        structured_question=sq,
        plan="Slice by product_category_english to find categories with the largest revenue growth.",
        dataset=dataset,
        status=InvestigationStatus.RUNNING,
    )

    # Run SQL Explorer first to produce a real query result in state
    explorer_result = sql_explorer_node(state)
    state = state.model_copy(update={
        "sql_queries": explorer_result["sql_queries"],
        "current_step": explorer_result["current_step"],
    })

    if state.sql_queries[-1].error:
        print(f"SQL Explorer errored, cannot test analyst: {state.sql_queries[-1].error}")
        return

    # Now run the Python Analyst
    result = python_analyst_node(state)

    print()
    print("=== Python Analyst result ===")
    pa = result["python_analyses"][0]
    print(f"tool_name: {pa.tool_name}")
    print(f"tool_args: {pa.tool_args}")
    print(f"error: {pa.error}")
    print()
    print("Rationale:")
    print(f"  {pa.rationale}")
    print()
    print("Result summary + findings:")
    print(pa.result_summary or "(no summary)")


if __name__ == "__main__":
    main()