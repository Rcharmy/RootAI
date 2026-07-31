"""
app.py

Command-line entry point for RootAI. Runs a single investigation end-to-end.

Usage:
    python app.py "Revenue dropped 12% in Q2 2018 vs Q2 2017, why?"

Behavior:
    1. Resets the LLM usage accumulator so cost tracking is per-investigation.
    2. Builds an initial InvestigationState with the user's question and
       a live DatasetContext introspected from the DuckDB.
    3. Invokes the compiled LangGraph with a recursion limit.
    4. Prints the final ExecutiveBrief and total token/cost usage.
    5. Stores the completed investigation into ChromaDB for future retrieval.
    6. Dumps the full state to traces/inv_<id>.json.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

import json
import sys
from datetime import datetime
from pathlib import Path

from rootai.config import config
from rootai.graph import compiled_graph
from rootai.memory.store import store_investigation
from rootai.state import (
    ExecutiveBrief,
    InvestigationState,
    InvestigationStatus,
    KPIQuestion,
)
from rootai.tools.dataset_context import build_dataset_context
from rootai.tools.llm import get_current_usage, reset_usage


def _serialize_state(state: dict) -> dict:
    """Convert a LangGraph state dict into JSON-safe form."""
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


def _rebuild_state_for_memory(final_state: dict) -> InvestigationState | None:
    """
    LangGraph returns state as a dict of possibly-serialized components.
    Rebuild a proper InvestigationState just for the memory store call,
    so store_investigation() gets typed access to structured_question and
    final_brief.

    Returns None if the state is not concluded or the brief is missing.
    """
    if final_state.get("status") not in (
        InvestigationStatus.CONCLUDED,
        InvestigationStatus.CONCLUDED.value,
        "concluded",
    ):
        return None

    brief = final_state.get("final_brief")
    if brief is None:
        return None

    # Normalize the brief and structured_question back into pydantic if they
    # came back as dicts (LangGraph serialization is inconsistent across versions).
    if isinstance(brief, dict):
        brief = ExecutiveBrief.model_validate(brief)

    sq = final_state.get("structured_question")
    if isinstance(sq, dict):
        sq = KPIQuestion.model_validate(sq)

    # Minimum viable state for the store call
    dataset = build_dataset_context()
    return InvestigationState(
        investigation_id=str(final_state.get("investigation_id", "unknown")),
        original_question=str(final_state.get("original_question", "")),
        structured_question=sq,
        status=InvestigationStatus.CONCLUDED,
        dataset=dataset,
        final_brief=brief,
    )


def run_investigation(question: str) -> dict:
    """Run a single investigation end-to-end. Returns the final state dict."""
    print("=" * 70)
    print(f"RootAI investigation")
    print(f"Question: {question}")
    print("=" * 70)

    reset_usage()

    dataset = build_dataset_context()

    initial_state = InvestigationState(
        original_question=question,
        dataset=dataset,
        status=InvestigationStatus.PENDING,
    )

    print(f"investigation_id: {initial_state.investigation_id}")
    print(f"model: {config.groq_model}")
    print(f"dataset: {dataset.name}, grain={dataset.grain}, "
          f"{len(dataset.dimensions)} dims / {len(dataset.metrics)} metrics")
    print("-" * 70)

    final_state = compiled_graph.invoke(
        initial_state.model_dump(),
        config={"recursion_limit": 30},
    )

    print("-" * 70)
    print("Investigation complete.")
    print(f"Status: {final_state.get('status')}")
    print(f"Steps taken: {final_state.get('current_step')}")

    final_usage = get_current_usage()
    print(f"Total LLM calls: {final_usage['call_count']}")
    print(f"Total tokens: {final_usage['total_tokens']:,} ({final_usage['input_tokens']:,} in / {final_usage['output_tokens']:,} out)")
    print(f"Estimated cost: ${final_usage['cost_usd']:.4f}")
    print()

    brief = final_state.get("final_brief")
    if brief is None:
        print("No final brief produced.")
    else:
        if hasattr(brief, "model_dump"):
            brief_dict = brief.model_dump()
        else:
            brief_dict = brief
        print("TL;DR:")
        print(f"  {brief_dict['tl_dr']}")
        print()
        print("Ranked causes:")
        for cause in brief_dict.get("ranked_causes", []):
            print(f"  {cause['rank']}. {cause['cause']}")
            print(f"     confidence: {cause['confidence']}")
        if brief_dict.get("caveats"):
            print()
            print("Caveats:")
            for c in brief_dict["caveats"]:
                print(f"  - {c}")

    # Phase 5: store the completed investigation into ChromaDB for future retrieval
    reconstructed = _rebuild_state_for_memory(final_state)
    if reconstructed is not None:
        stored = store_investigation(reconstructed)
        print()
        print(f"Memory: stored={stored} (investigation_id={reconstructed.investigation_id})")
    else:
        print()
        print("Memory: not stored (investigation did not conclude with a brief)")

    trace_dir = Path(config.traces_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    inv_id = final_state.get("investigation_id", "unknown")
    trace_path = trace_dir / f"{inv_id}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_state(final_state), f, indent=2, default=str)
    print(f"Trace saved: {trace_path}")

    return final_state


def main() -> None:
    if len(sys.argv) < 2:
        question = "Revenue dropped 12% in Q2 2018 vs Q2 2017, why?"
        print(f"(no question provided, using default)")
    else:
        question = sys.argv[1]

    run_investigation(question)


if __name__ == "__main__":
    main()