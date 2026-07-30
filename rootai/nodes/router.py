"""
rootai/nodes/router.py

Router node stub. In Phase 3 this becomes the LLM call that reads state
and decides CONTINUE / REFINE / CONCLUDE / ABORT. The design_decisions.md
justifies this being a node (auditable) rather than a conditional edge
(cheaper).

For Phase 2 the stub uses a simple deterministic rule:
- If we have hypotheses and have taken >= 3 steps, CONCLUDE.
- Otherwise CONTINUE.

This gives the skeleton a real loop-and-exit structure without needing an
LLM call.
"""
from __future__ import annotations

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    NodeName,
    RouterDecision,
)


# Deterministic step threshold for the stub. In Phase 3 the LLM decides.
STUB_MIN_STEPS_BEFORE_CONCLUDE = 4


def router_node(state: InvestigationState) -> dict:
    """Stub: deterministic router based on step count and budget."""
    print(f"STUB: router node called (step {state.current_step + 1})")

    # Budget-first: if we blew past the budget, ABORT
    if state.budget.exceeded():
        decision = RouterDecision.ABORT
        rationale = f"STUB: budget exceeded ({state.budget.steps_used}/{state.budget.max_steps})"
    elif len(state.hypotheses) >= 1 and state.current_step >= STUB_MIN_STEPS_BEFORE_CONCLUDE:
        decision = RouterDecision.CONCLUDE
        rationale = f"STUB: {len(state.hypotheses)} hypothesis, {state.current_step} steps, concluding"
    else:
        decision = RouterDecision.CONTINUE
        rationale = f"STUB: {state.current_step} steps taken, continuing exploration"

    # Bump the budget's step counter so the guard triggers eventually
    new_budget = state.budget.model_copy(
        update={"steps_used": state.budget.steps_used + 1}
    )

    log_entry = ActionLogEntry(
        step=state.current_step + 1,
        node=NodeName.ROUTER,
        action="stub_route",
        input_summary=f"{len(state.hypotheses)} hyp, {len(state.evidence)} evi",
        output_summary=f"{decision.value}: {rationale[:80]}",
    )

    return {
        "router_decision": decision,
        "router_rationale": rationale,
        "budget": new_budget,
        "current_step": state.current_step + 1,
        "current_node": NodeName.ROUTER,
        "action_log": [log_entry],
    }