"""
rootai/nodes/hypothesis_former.py

Hypothesis Former node stub. In Phase 3 this becomes the LLM call that
proposes new hypotheses from the latest evidence, and updates status /
confidence on existing ones. Phase 5 wires the ChromaDB RAG memory in.

For Phase 2 the stub emits one canned Hypothesis and one supporting Evidence.
"""
from __future__ import annotations

from rootai.state import (
    ActionLogEntry,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    InvestigationState,
    NodeName,
)


def hypothesis_former_node(state: InvestigationState) -> dict:
    """Stub: propose one hypothesis and one supporting evidence record."""
    print(f"STUB: hypothesis_former node called (step {state.current_step + 1})")

    stub_hypothesis = Hypothesis(
        statement="STUB HYPOTHESIS: revenue decline concentrated in bed_bath_table category",
        rationale="STUB: top category shows outsized drop vs baseline",
        status=HypothesisStatus.PROPOSED,
        confidence=0.6,
        dimensions_to_test=["product_category_english", "seller_id"],
        created_at_step=state.current_step + 1,
        updated_at_step=state.current_step + 1,
    )

    stub_evidence = Evidence(
        step=state.current_step + 1,
        source_node=NodeName.HYPOTHESIS_FORMER,
        description="STUB: category-level revenue comparison",
        finding="STUB: bed_bath_table down 18% vs Q2 2017, largest single-category delta",
        supports_hypothesis_ids=[stub_hypothesis.id],
        magnitude=0.45,
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.HYPOTHESIS_FORMER,
        action="stub_form_hypothesis",
        input_summary="1 python analysis, 1 sql query",
        output_summary=f"proposed h={stub_hypothesis.id} with e={stub_evidence.id}",
    )

    return {
        "hypotheses": [stub_hypothesis],
        "evidence": [stub_evidence],
        "current_step": state.current_step + 1,
        "current_node": NodeName.HYPOTHESIS_FORMER,
        "action_log": [log_entry],
    }