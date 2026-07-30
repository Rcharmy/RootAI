"""
rootai/nodes/writer.py

Writer node stub. In Phase 3 this becomes the LLM call that synthesizes
the final ExecutiveBrief from hypotheses, evidence, and analyses.

For Phase 2 the stub composes a brief from the state's existing hypotheses
so you see the plumbing works end-to-end.
"""
from __future__ import annotations

from datetime import datetime

from rootai.state import (
    ActionLogEntry,
    ExecutiveBrief,
    HypothesisStatus,
    InvestigationState,
    InvestigationStatus,
    NodeName,
    RankedCause,
)


def writer_node(state: InvestigationState) -> dict:
    """Stub: synthesize a final ExecutiveBrief from hypotheses in state."""
    print(f"STUB: writer node called (step {state.current_step + 1})")

    # Rank hypotheses by confidence, use as ranked causes
    sorted_hyp = sorted(state.hypotheses, key=lambda h: h.confidence, reverse=True)
    ranked_causes = [
        RankedCause(
            rank=i + 1,
            cause=h.statement,
            confidence=h.confidence,
            evidence_ids=h.supporting_evidence_ids,
            contribution_estimate="STUB: not computed",
            recommended_action="STUB: recommendation pending Phase 3",
        )
        for i, h in enumerate(sorted_hyp)
        if h.status != HypothesisStatus.REFUTED
    ]

    brief = ExecutiveBrief(
        tl_dr=(
            f"STUB BRIEF: Investigated '{state.original_question}'. "
            f"Found {len(ranked_causes)} candidate cause(s) across "
            f"{state.current_step} exploration steps."
        ),
        ranked_causes=ranked_causes,
        chart_refs=[],
        caveats=["STUB: this is a Phase 2 skeleton output, not a real investigation."],
        recommended_next_actions=["Advance to Phase 3 to replace stubs with real logic."],
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.WRITER,
        action="stub_write_brief",
        input_summary=f"{len(state.hypotheses)} hyp, {len(state.evidence)} evi",
        output_summary=f"{len(ranked_causes)} ranked causes",
    )

    return {
        "final_brief": brief,
        "status": InvestigationStatus.CONCLUDED,
        "current_step": state.current_step + 1,
        "current_node": NodeName.WRITER,
        "action_log": [log_entry],
        "completed_at": datetime.utcnow(),
    }