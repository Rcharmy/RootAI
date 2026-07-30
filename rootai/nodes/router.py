"""
rootai/nodes/router.py

Router node. Called after every Hypothesis Former hop. Decides:
- CONTINUE: keep exploring, back to SQL Explorer
- REFINE: dig deeper on a specific hypothesis (Phase 5+; for now routes like CONTINUE)
- CONCLUDE: enough evidence, hand off to Writer
- ABORT: something is broken, stop with an ABORTED status

Design:
- Budget check (steps and cost) runs BEFORE any LLM call. If exceeded,
  ABORT deterministically. Interview-defensible: "Guardrails should not
  cost API budget to enforce."
- LLM sees a compact state summary, not raw data. Cost per Router call
  should be under 500 tokens in / 100 out.
- The Router increments budget.steps_used to represent that a routing
  decision was made, even though the "step" concept is fuzzy. This is
  what triggers eventual auto-abort even if the LLM keeps saying CONTINUE.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from rootai.state import (
    ActionLogEntry,
    InvestigationState,
    NodeName,
    RouterDecision,
)
from rootai.tools.llm import get_structured_llm


class RouterOutput(BaseModel):
    """LLM's routing decision."""
    decision: str = Field(description="One of: continue, refine, conclude, abort.")
    rationale: str = Field(description="1-2 sentence explanation.")


SYSTEM_PROMPT = (
    "You are the Router for RootAI. After each investigation step you decide whether to keep exploring or stop.\n\n"
    "Choose ONE decision:\n"
    "- continue: at least one hypothesis is worth testing further with a new SQL slice. Use when: fewer than 2 hypotheses exist, or the highest-confidence hypothesis has confidence < 0.7 and dimensions_to_test are non-empty.\n"
    "- refine: an existing hypothesis needs a targeted deeper look (specific seller, specific state within a category). Use sparingly, only when a clear hypothesis has emerged but needs one more evidence angle.\n"
    "- conclude: the investigation is done. Use when: (a) at least one hypothesis has confidence >= 0.7 with evidence, OR (b) all hypotheses are INCONCLUSIVE and further slicing is unlikely to yield a driver, OR (c) 5+ SQL queries have been run and no new hypotheses are forming.\n"
    "- abort: something is broken (repeated SQL errors, contradictory evidence, no progress). Rare.\n\n"
    "Rules:\n"
    "- Choose conclude EARLIER, not later. A brief with 60% confidence is more useful than 10 more steps for no gain.\n"
    "- If the investigation has taken 4+ steps and no hypothesis has moved above 0.5 confidence, prefer conclude (write an INCONCLUSIVE-flavored brief) over continue.\n"
    "- Do NOT return decision='continue' if the highest-confidence hypothesis is already >= 0.8; that means you have your answer."
)


USER_TEMPLATE = (
    "Investigation question: {question}\n"
    "Steps used: {steps_used}/{max_steps}, cost used: ${cost_used:.4f}/${max_cost:.4f}\n\n"
    "Hypotheses ({n_hyp}):\n{hyp_summary}\n\n"
    "Evidence count: {n_evi}\n"
    "SQL queries: {n_sql} (last error: {last_sql_error})\n"
    "Python analyses: {n_pa}\n"
    "Dead ends: {dead_ends}\n\n"
    "Decide: continue, refine, conclude, or abort. Give a 1-2 sentence rationale."
)


_DECISION_MAP = {
    "continue": RouterDecision.CONTINUE,
    "refine": RouterDecision.REFINE,
    "conclude": RouterDecision.CONCLUDE,
    "abort": RouterDecision.ABORT,
}


def _summarize_hypotheses(state: InvestigationState) -> str:
    if not state.hypotheses:
        return "(none)"
    lines = []
    for h in state.hypotheses:
        n_supp = len(h.supporting_evidence_ids)
        n_ref = len(h.refuting_evidence_ids)
        lines.append(
            f"- {h.id} [status={h.status.value}, conf={h.confidence:.2f}, supp={n_supp}, ref={n_ref}]: {h.statement[:150]}"
        )
    return "\n".join(lines)


def router_node(state: InvestigationState) -> dict:
    """Deterministic budget check first, then LLM decision."""
    step = state.current_step + 1
    print(f"router: deciding next step (step {step})")

    # Budget first, no LLM call
    if state.budget.exceeded():
        decision = RouterDecision.ABORT
        rationale = f"budget exceeded (steps {state.budget.steps_used}/{state.budget.max_steps}, cost ${state.budget.cost_usd:.4f}/${state.budget.max_cost_usd:.4f})"
        print(f"  {decision.value}: {rationale}")
        new_budget = state.budget.model_copy(update={
            "steps_used": state.budget.steps_used + 1,
            "reason_stopped": rationale,
        })
        log_entry = ActionLogEntry(
            step=step,
            node=NodeName.ROUTER,
            action="budget_abort",
            input_summary="",
            output_summary=rationale[:120],
        )
        return {
            "router_decision": decision,
            "router_rationale": rationale,
            "budget": new_budget,
            "current_step": step,
            "current_node": NodeName.ROUTER,
            "action_log": [log_entry],
        }

    last_sql_error = "-"
    if state.sql_queries and state.sql_queries[-1].error:
        last_sql_error = state.sql_queries[-1].error[:80]

    user_msg = USER_TEMPLATE.format(
        question=state.original_question,
        steps_used=state.budget.steps_used,
        max_steps=state.budget.max_steps,
        cost_used=state.budget.cost_usd,
        max_cost=state.budget.max_cost_usd,
        n_hyp=len(state.hypotheses),
        hyp_summary=_summarize_hypotheses(state),
        n_evi=len(state.evidence),
        n_sql=len(state.sql_queries),
        last_sql_error=last_sql_error,
        n_pa=len(state.python_analyses),
        dead_ends=", ".join(state.dead_ends) if state.dead_ends else "(none)",
    )

    llm = get_structured_llm(RouterOutput)

    try:
        output: RouterOutput = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        decision_str = output.decision.strip().lower()
        decision = _DECISION_MAP.get(decision_str, RouterDecision.CONTINUE)
        rationale = output.rationale
    except (ValidationError, Exception) as e:
        print(f"  router LLM call failed: {e}. Defaulting to CONTINUE.")
        decision = RouterDecision.CONTINUE
        rationale = f"LLM failed ({str(e)[:80]}), defaulting to CONTINUE"

    print(f"  {decision.value}: {rationale[:120]}")

    new_budget = state.budget.model_copy(update={
        "steps_used": state.budget.steps_used + 1,
    })

    log_entry = ActionLogEntry(
        step=step,
        node=NodeName.ROUTER,
        action="route",
        input_summary=f"{len(state.hypotheses)} hyp, {len(state.evidence)} evi, {len(state.sql_queries)} sql",
        output_summary=f"{decision.value}: {rationale[:100]}",
    )

    return {
        "router_decision": decision,
        "router_rationale": rationale,
        "budget": new_budget,
        "current_step": step,
        "current_node": NodeName.ROUTER,
        "action_log": [log_entry],
    }