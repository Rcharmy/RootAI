"""
scripts/seed_memory.py

One-off script to populate ChromaDB with canonical past investigations.
Run once before demos so the very first user query has something to
retrieve. Idempotent: re-running upserts the same records.

Zero Groq token cost.

Usage:
    python scripts/seed_memory.py
    python scripts/seed_memory.py --clear  # wipe and re-seed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.memory.store import (
    clear_memory,
    get_collection_stats,
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


def _make_investigation(
    inv_id: str,
    question: str,
    kpi: str,
    direction: str,
    comp_window: tuple[str, str],
    base_window: tuple[str, str],
    tl_dr: str,
    causes: list[tuple[str, float]],  # (cause_text, confidence)
    caveats: list[str] | None = None,
) -> InvestigationState:
    """Build a synthetic completed InvestigationState for seeding."""
    sq = KPIQuestion(
        kpi_name=kpi,
        direction=direction,
        magnitude_pct=None,
        time_window={"start": comp_window[0], "end": comp_window[1]},
        comparison_window={"start": base_window[0], "end": base_window[1]},
        grain="quarterly",
        raw_question=question,
    )
    brief = ExecutiveBrief(
        tl_dr=tl_dr,
        ranked_causes=[
            RankedCause(
                rank=i + 1,
                cause=cause_text,
                confidence=conf,
                evidence_ids=[],
            )
            for i, (cause_text, conf) in enumerate(causes)
        ],
        chart_refs=[],
        caveats=caveats or [],
        recommended_next_actions=[],
    )
    return InvestigationState(
        investigation_id=inv_id,
        original_question=question,
        structured_question=sq,
        status=InvestigationStatus.CONCLUDED,
        dataset=DatasetContext(),
        final_brief=brief,
    )


# Five canonical investigations covering the three main eval case archetypes:
# - single_cause growth (2)
# - single_cause decline (1)
# - null / no clear cause (1)
# - multi_cause (1)
# The IDs are prefixed 'seed_' so they are visually distinguishable from real runs.
SEEDS = [
    _make_investigation(
        inv_id="seed_q2_revenue_growth",
        question="Why did Q2 2018 revenue grow substantially versus Q2 2017?",
        kpi="revenue",
        direction="up",
        comp_window=("2018-04-01", "2018-06-30"),
        base_window=("2017-04-01", "2017-06-30"),
        tl_dr=(
            "Q2 2018 revenue grew roughly 138% versus Q2 2017. Growth was broadly distributed "
            "across categories with the top 5 (health_beauty, watches_gifts, bed_bath_table, "
            "housewares, furniture_decor) contributing 47% of the total delta. Growth reflects "
            "platform scaling rather than a single category driver."
        ),
        causes=[
            (
                "Platform-wide expansion across categories; growth in health_beauty and "
                "watches_gifts is largest but concentration is moderate (47% top-5), "
                "consistent with broad marketplace scaling rather than a single driver.",
                0.75,
            ),
        ],
        caveats=[
            "Top-5 concentration of 47% is moderate; a stronger single-driver hypothesis would require concentration > 60%."
        ],
    ),
    _make_investigation(
        inv_id="seed_h1_revenue_growth",
        question="What drove H1 2018 revenue growth compared to H1 2017?",
        kpi="revenue",
        direction="up",
        comp_window=("2018-01-01", "2018-06-30"),
        base_window=("2017-01-01", "2017-06-30"),
        tl_dr=(
            "H1 2018 revenue nearly doubled versus H1 2017. Growth was driven primarily by "
            "order volume (customer acquisition) rather than average order value, which was "
            "roughly flat. Consistent with marketplace-platform growth patterns in scaling phase."
        ),
        causes=[
            (
                "Order volume growth (customer acquisition), driven by increased unique orders "
                "and unique customers. AOV remained roughly flat, meaning revenue growth was "
                "volume-led, not basket-led.",
                0.80,
            ),
        ],
    ),
    _make_investigation(
        inv_id="seed_aov_decline",
        question="Why did average order value decline in 2018 versus 2017?",
        kpi="aov",
        direction="down",
        comp_window=("2018-01-01", "2018-06-30"),
        base_window=("2017-01-01", "2017-06-30"),
        tl_dr=(
            "AOV declined modestly. The driver is category mix shift: higher-frequency, "
            "lower-price categories (health_beauty, housewares) grew as a share of the mix, "
            "while higher-ticket categories held steady in absolute terms but lost share. "
            "AOV within each category was roughly stable."
        ),
        causes=[
            (
                "Category mix shift toward lower-price categories. Within-category AOV was "
                "stable; the average dropped because the mix rebalanced.",
                0.70,
            ),
        ],
        caveats=[
            "The mix shift is a positive signal for marketplace breadth (more high-frequency purchases), even though headline AOV moved down."
        ],
    ),
    _make_investigation(
        inv_id="seed_no_single_cause",
        question="AOV in Q2 2018 was about 2% lower than Q2 2017. What caused the decline?",
        kpi="aov",
        direction="down",
        comp_window=("2018-04-01", "2018-06-30"),
        base_window=("2017-04-01", "2017-06-30"),
        tl_dr=(
            "No isolable driver. The 2% change is broadly distributed across categories, "
            "states, and sellers, and is within normal variation for a KPI at this granularity. "
            "Recommend not treating this movement as actionable."
        ),
        causes=[],  # empty ranked_causes for null case
        caveats=[
            "Top single-dimension contribution below 20% of total variance; no dimension emerges as a driver.",
            "Small drift on high-cardinality dimensions like AOV typically reflects noise, not signal.",
        ],
    ),
    _make_investigation(
        inv_id="seed_multi_cause_growth",
        question="Give a full accounting of what drove H1 2018 revenue growth.",
        kpi="revenue",
        direction="up",
        comp_window=("2018-01-01", "2018-06-30"),
        base_window=("2017-01-01", "2017-06-30"),
        tl_dr=(
            "Growth had two roughly-equal drivers: São Paulo state expansion in existing "
            "categories (~45%), and new category emergence (particularly computers_accessories) "
            "distributed across states (~40%). These effects interact: SP was also the largest "
            "state for the new categories, so state and category contributions compound rather "
            "than substitute."
        ),
        causes=[
            (
                "São Paulo state expansion in existing categories (bed_bath_table, health_beauty). "
                "SP grew disproportionately in both order volume and category-level revenue.",
                0.65,
            ),
            (
                "New category emergence, particularly computers_accessories, driven by new "
                "seller onboarding rather than existing-seller expansion. Distributed across states.",
                0.60,
            ),
        ],
        caveats=[
            "Real revenue growth rarely has one cause. Attributing 100% to either state or category expansion would miss the interaction structure."
        ],
    ),
]


def main():
    args = sys.argv[1:]
    if "--clear" in args:
        print("Clearing existing memory before seeding...")
        clear_memory()

    print(f"Seeding {len(SEEDS)} canonical investigations...")
    for inv in SEEDS:
        ok = store_investigation(inv)
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {inv.investigation_id}: {inv.original_question[:70]}")

    print()
    print("Final memory stats:", get_collection_stats())


if __name__ == "__main__":
    main()