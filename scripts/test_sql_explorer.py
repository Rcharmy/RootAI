"""
scripts/test_sql_explorer.py

Exercise the real SQL Explorer node in isolation. Builds a state where
the Planner output has already been set, then runs the Explorer once
and prints what happened.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        direction="down",
        magnitude_pct=-12.0,
        time_window={"start": "2018-04-01", "end": "2018-06-30"},
        comparison_window={"start": "2017-04-01", "end": "2017-06-30"},
        grain="quarterly",
        raw_question="Revenue in Q2 2018 was lower than Q2 2017, why?",
    )

    state = InvestigationState(
        original_question=sq.raw_question,
        structured_question=sq,
        plan="First slice by product_category_english to find categories with the largest revenue drop between Q2 2017 and Q2 2018. Then examine customer_state.",
        dataset=dataset,
        status=InvestigationStatus.RUNNING,
    )

    result = sql_explorer_node(state)

    print()
    print("=== SQL Explorer result ===")
    q = result["sql_queries"][0]
    print(f"passed_guardrails: {q.passed_guardrails}")
    print(f"row_count: {q.row_count}")
    print(f"columns: {q.columns}")
    print(f"duration_ms: {q.duration_ms}")
    print(f"error: {q.error}")
    print()
    print("Rationale:")
    print(f"  {q.rationale}")
    print()
    print("SQL:")
    print(q.query)
    print()
    if q.result_preview:
        print("Result preview:")
        print(q.result_preview)


if __name__ == "__main__":
    main()