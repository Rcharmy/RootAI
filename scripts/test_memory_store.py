"""
scripts/test_memory_store.py

Sanity test for rootai.memory.store. Creates a fake completed investigation,
stores it, queries for something similar, prints results.

Zero Groq token cost: ChromaDB uses local embeddings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.memory.store import (
    clear_memory,
    get_collection_stats,
    query_similar,
    store_investigation,
)
from rootai.state import (
    DatasetContext,
    ExecutiveBrief,
    InvestigationState,
    InvestigationStatus,
    KPIQuestion,
    RankedCause,
)


def make_fake_investigation(inv_id: str, question: str, kpi: str, direction: str, cause: str, confidence: float) -> InvestigationState:
    """Build a synthetic completed state for testing."""
    sq = KPIQuestion(
        kpi_name=kpi,
        direction=direction,
        magnitude_pct=None,
        time_window={"start": "2018-04-01", "end": "2018-06-30"},
        comparison_window={"start": "2017-04-01", "end": "2017-06-30"},
        grain="quarterly",
        raw_question=question,
    )
    brief = ExecutiveBrief(
        tl_dr=f"Q2 2018 {kpi} {direction} versus Q2 2017. {cause}",
        ranked_causes=[
            RankedCause(rank=1, cause=cause, confidence=confidence, evidence_ids=[])
        ],
        chart_refs=[],
        caveats=[],
        recommended_next_actions=[],
    )
    state = InvestigationState(
        investigation_id=inv_id,
        original_question=question,
        structured_question=sq,
        status=InvestigationStatus.CONCLUDED,
        dataset=DatasetContext(),
        final_brief=brief,
    )
    return state


def main():
    print("Clearing memory for clean test...")
    clear_memory()
    print("Stats after clear:", get_collection_stats())
    print()

    # Store three fake investigations
    print("Storing 3 fake investigations...")
    fake1 = make_fake_investigation(
        inv_id="inv_test001",
        question="Why did Q2 2018 revenue grow?",
        kpi="revenue",
        direction="up",
        cause="Growth concentrated in health_beauty and watches_gifts categories",
        confidence=0.75,
    )
    fake2 = make_fake_investigation(
        inv_id="inv_test002",
        question="What caused the AOV decline in H1 2018?",
        kpi="aov",
        direction="down",
        cause="Category mix shift toward lower-priced items",
        confidence=0.60,
    )
    fake3 = make_fake_investigation(
        inv_id="inv_test003",
        question="Why did delivery times get worse in 2018?",
        kpi="delivery_days",
        direction="up",
        cause="Geographic expansion to more distant states",
        confidence=0.80,
    )

    for f in (fake1, fake2, fake3):
        ok = store_investigation(f)
        print(f"  stored {f.investigation_id}: {ok}")
    print()
    print("Stats after storage:", get_collection_stats())
    print()

    # Query for something similar to inv_test001
    print("Query: 'Revenue in Q2 2018 higher than Q2 2017. What drove growth?'")
    priors = query_similar("Revenue in Q2 2018 higher than Q2 2017. What drove growth?", n_results=3)
    for p in priors:
        print(f"  - {p.id} (sim={p.similarity_score:.2f}): {p.question}")
        print(f"    top cause: {p.key_causes}")
    print()

    # Query for something unrelated
    print("Query: 'What are the top selling seller cities?'")
    priors = query_similar("What are the top selling seller cities?", n_results=3)
    for p in priors:
        print(f"  - {p.id} (sim={p.similarity_score:.2f}): {p.question}")
    print()

    # Cleanup
    print("Clearing test data...")
    clear_memory()
    print("Stats after clear:", get_collection_stats())


if __name__ == "__main__":
    main()