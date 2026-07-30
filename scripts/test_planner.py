"""
scripts/test_planner.py

Exercise the real Planner node in isolation with a synthetic state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.nodes.planner import planner_node
from rootai.state import InvestigationState, InvestigationStatus
from rootai.tools.dataset_context import build_dataset_context


def main():
    dataset = build_dataset_context()
    state = InvestigationState(
        original_question="Revenue in Q2 2018 was lower than Q2 2017, why?",
        dataset=dataset,
        status=InvestigationStatus.PENDING,
    )
    result = planner_node(state)

    print()
    print("=== Planner result ===")
    print("status:", result["status"])
    print("plan:", result["plan"])
    print()
    sq = result["structured_question"]
    print("KPIQuestion:")
    print(f"  kpi_name: {sq.kpi_name}")
    print(f"  direction: {sq.direction}")
    print(f"  magnitude_pct: {sq.magnitude_pct}")
    print(f"  time_window: {sq.time_window}")
    print(f"  comparison_window: {sq.comparison_window}")
    print(f"  grain: {sq.grain}")
    print(f"  needs_clarification: {result['needs_clarification']}")


if __name__ == "__main__":
    main()