"""
app.py

Command-line entry point for RootAI. Runs a single investigation end-to-end.

Usage:
    python app.py "Revenue dropped 12% in Q2 2018 vs Q2 2017, why?"

Behavior:
    1. Builds an initial InvestigationState with the user's question and
       a live DatasetContext introspected from the DuckDB.
    2. Invokes the compiled LangGraph.
    3. Prints the final ExecutiveBrief.
    4. Dumps the full state (including action_log) to traces/inv_<id>.json.

In Phase 2 every node is a stub; the run produces a placeholder brief but
proves the graph plumbing works end-to-end.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from rootai.config import config
from rootai.graph import compiled_graph
from rootai.state import InvestigationState, InvestigationStatus
from rootai.tools.dataset_context import build_dataset_context


def _serialize_state(state: dict) -> dict:
    """
    Convert a LangGraph state dict into JSON-safe form.

    LangGraph returns state as a dict of field name -> value. Pydantic models
    inside need model_dump(), datetimes need isoformat().
    """
    def convert(x):
        if hasattr(x, "model_dump"):
            return x.model_dump(mode="json")
        if isinstance(x, list):
            return [convert(i) for i in x]
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, datetime):
            return x.isoformat()
        return x

    return {k: convert(v) for k, v in state.items()}


def run_investigation(question: str) -> dict:
    """
    Run a single investigation end-to-end. Returns the final state dict.
    """
    print("=" * 70)
    print(f"RootAI investigation")
    print(f"Question: {question}")
    print("=" * 70)

    dataset = build_dataset_context()

    initial_state = InvestigationState(
        original_question=question,
        dataset=dataset,
        status=InvestigationStatus.PENDING,
    )

    print(f"investigation_id: {initial_state.investigation_id}")
    print(f"dataset: {dataset.name}, grain={dataset.grain}, "
          f"{len(dataset.dimensions)} dims / {len(dataset.metrics)} metrics")
    print("-" * 70)

    # LangGraph accepts either a state model or its dict form as input.
    # We pass model_dump() so the reducers see plain-Python values.
    final_state = compiled_graph.invoke(initial_state.model_dump())

    print("-" * 70)
    print("Investigation complete.")
    print(f"Status: {final_state.get('status')}")
    print(f"Steps taken: {final_state.get('current_step')}")
    print()

    brief = final_state.get("final_brief")
    if brief is None:
        print("No final brief produced.")
    else:
        # brief may already be a dict (from serialization) or a Pydantic model
        if hasattr(brief, "model_dump"):
            brief = brief.model_dump()
        print("TL;DR:")
        print(f"  {brief['tl_dr']}")
        print()
        print("Ranked causes:")
        for cause in brief.get("ranked_causes", []):
            print(f"  {cause['rank']}. {cause['cause']}")
            print(f"     confidence: {cause['confidence']}")
        if brief.get("caveats"):
            print()
            print("Caveats:")
            for c in brief["caveats"]:
                print(f"  - {c}")

    # Dump trace
    trace_dir = Path(config.traces_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    inv_id = final_state.get("investigation_id", "unknown")
    trace_path = trace_dir / f"{inv_id}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_state(final_state), f, indent=2, default=str)
    print()
    print(f"Trace saved: {trace_path}")

    return final_state


def main() -> None:
    if len(sys.argv) < 2:
        # Default question so `python app.py` still works for a smoke test
        question = "Revenue dropped 12% in Q2 2018 vs Q2 2017, why?"
        print(f"(no question provided, using default)")
    else:
        question = sys.argv[1]

    run_investigation(question)


if __name__ == "__main__":
    main()