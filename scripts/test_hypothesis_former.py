"""
scripts/test_hypothesis_former.py

Exercise the real Hypothesis Former by chaining Explorer -> Analyst ->
Former so it has real findings to reason about.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.nodes.hypothesis_former import hypothesis_former_node
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
        raw_question="Revenue in Q2 2018 was much higher than Q2 2017. What drove the growth?",
    )

    state = InvestigationState(
        original_question=sq.raw_question,
        structured_question=sq,
        plan="Slice by product_category_english first to identify categories driving revenue growth. Then examine customer_state for geographic concentration.",
        dataset=dataset,
        status=InvestigationStatus.RUNNING,
    )

    # SQL Explorer
    r1 = sql_explorer_node(state)
    state = state.model_copy(update={
        "sql_queries": r1["sql_queries"],
        "current_step": r1["current_step"],
    })
    if state.sql_queries[-1].error:
        print(f"SQL errored: {state.sql_queries[-1].error}")
        return

    # Python Analyst
    r2 = python_analyst_node(state)
    state = state.model_copy(update={
        "python_analyses": r2["python_analyses"],
        "current_step": r2["current_step"],
    })

    # Hypothesis Former
    r3 = hypothesis_former_node(state)

    print()
    print("=== Hypothesis Former result ===")
    hyps = r3.get("hypotheses", [])
    evs = r3.get("evidence", [])
    print(f"hypotheses emitted: {len(hyps)}")
    for h in hyps:
        print(f"  id={h.id}")
        print(f"    statement: {h.statement}")
        print(f"    status: {h.status.value}, confidence: {h.confidence:.2f}")
        print(f"    supporting_evidence_ids: {h.supporting_evidence_ids}")
        print(f"    refuting_evidence_ids: {h.refuting_evidence_ids}")
    print()
    print(f"evidence emitted: {len(evs)}")
    for e in evs:
        print(f"  id={e.id}")
        print(f"    finding: {e.finding}")
        print(f"    supports: {e.supports_hypothesis_ids}")
        print(f"    refutes: {e.refutes_hypothesis_ids}")
        print(f"    magnitude: {e.magnitude}")


if __name__ == "__main__":
    main()