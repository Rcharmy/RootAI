"""
rootai/nodes/writer.py

Writer node. Terminal node in every investigation. Synthesizes the
accumulated hypotheses, evidence, SQL queries, and Python analyses
into a final ExecutiveBrief.

Design:
- Ranked causes come only from hypotheses that are SUPPORTED or PROPOSED
  with confidence >= 0.5. Refuted or low-confidence hypotheses drop from
  ranked_causes but may appear in caveats.
- When no hypothesis has confidence >= 0.5, the Writer emits an
  explicit "no clear cause" tl_dr and empty ranked_causes list. It does
  NOT fabricate. This is the property the null eval cases test.
- The Writer prompt permits the LLM to say "the user's premise appears
  incorrect" if a hypothesis argued that. This surfaces contradictions
  rather than papering over them.
- ranked_causes carry evidence_ids from the hypotheses they were built
  from, keeping the brief auditable.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from rootai.state import (
    ActionLogEntry,
    ExecutiveBrief,
    HypothesisStatus,
    InvestigationState,
    InvestigationStatus,
    NodeName,
    RankedCause,
)
from rootai.tools.llm import get_structured_llm


class RankedCauseDraft(BaseModel):
    """One entry in the ranked list of causes. LLM emits these."""
    rank: int = Field(description="1 = strongest cause. Sequential from 1.")
    cause: str = Field(description="1-2 sentence description of the cause.")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0. Copy from the underlying hypothesis's confidence.")
    hypothesis_id: str = Field(description="ID of the hypothesis this ranked cause is based on (h_XXXXXXXX).")
    contribution_estimate: str | None = Field(default=None, description="Estimated contribution as a phrase (e.g. '~13% of total delta') if the evidence supports one.")
    recommended_action: str | None = Field(default=None, description="One-sentence actionable recommendation, or null if none is warranted.")


class WriterOutput(BaseModel):
    """LLM structured output for the Writer node."""
    tl_dr: str = Field(description="2-3 sentence executive summary. If no clear cause, say so explicitly.")
    ranked_causes: list[RankedCauseDraft] = Field(description="Ranked causes with confidence >= 0.5. Empty list if none qualify.")
    caveats: list[str] = Field(description="Important caveats: refuted hypotheses worth mentioning, data quality issues, small sample sizes, or window truncation.")
    recommended_next_actions: list[str] = Field(description="Concrete next steps a human analyst could take. Empty list if none.")


SYSTEM_PROMPT = (
    "You are the Writer for RootAI. Your job is to synthesize an investigation into a concise executive brief.\n\n"
    "Rules:\n"
    "- ranked_causes MUST come from hypotheses that are SUPPORTED or PROPOSED with confidence >= 0.5.\n"
    "- Do NOT include REFUTED or INCONCLUSIVE hypotheses in ranked_causes. They may appear in caveats.\n"
    "- If NO hypothesis has confidence >= 0.5, the tl_dr must explicitly say no clear cause was found, and ranked_causes MUST be an empty list. Do not fabricate causes to fill the ranked list.\n"
    "- Rank causes by confidence (1 = highest confidence).\n"
    "- Each ranked cause MUST reference the underlying hypothesis_id.\n"
    "- If a hypothesis argued the user's stated premise was incorrect (e.g. 'revenue actually rose, not dropped'), lead the tl_dr with that.\n"
    "- Caveats should mention: hypotheses that were refuted, hypotheses that were inconclusive with brief reasoning, window truncation issues, and any repeated tool errors.\n"
    "- recommended_next_actions should be concrete (e.g. 'Investigate whether the SP concentration is durable by looking at Q3'). Not vague ('do more analysis')."
)


USER_TEMPLATE = (
    "Investigation question: {question}\n"
    "Structured question: kpi={kpi}, direction={direction}, magnitude={magnitude}, windows: comp={comp}, base={base}\n\n"
    "Final hypotheses ({n_hyp}):\n{hyp_summary}\n\n"
    "Final evidence ({n_evi}):\n{evi_summary}\n\n"
    "Investigation stats:\n"
    "- SQL queries run: {n_sql}\n"
    "- Python analyses run: {n_pa}\n"
    "- Steps used: {steps}\n"
    "- Errors along the way: {errors_summary}\n\n"
    "Router's final decision and rationale: {router_decision} - {router_rationale}\n\n"
    "Produce the ExecutiveBrief. Remember: do not fabricate causes if no hypothesis has confidence >= 0.5."
)


def _summarize_hypotheses(state: InvestigationState) -> str:
    if not state.hypotheses:
        return "(none)"
    lines = []
    for h in state.hypotheses:
        n_supp = len(h.supporting_evidence_ids)
        n_ref = len(h.refuting_evidence_ids)
        lines.append(
            f"- {h.id} [status={h.status.value}, conf={h.confidence:.2f}, supp={n_supp}, ref={n_ref}]: {h.statement}"
        )
    return "\n".join(lines)


def _summarize_evidence(state: InvestigationState) -> str:
    if not state.evidence:
        return "(none)"
    lines = []
    for e in state.evidence[-10:]:  # cap at last 10 to keep prompt small
        supp = ",".join(e.supports_hypothesis_ids) or "-"
        ref = ",".join(e.refutes_hypothesis_ids) or "-"
        mag = f" (magnitude={e.magnitude})" if e.magnitude is not None else ""
        lines.append(f"- {e.id} (supports=[{supp}], refutes=[{ref}]){mag}: {e.finding}")
    return "\n".join(lines)


def _errors_summary(state: InvestigationState) -> str:
    """Cheap check: were there systematic errors during the investigation?"""
    py_errors = sum(1 for a in state.python_analyses if a.error)
    sql_errors = sum(1 for q in state.sql_queries if q.error)
    rejected_sql = sum(1 for q in state.sql_queries if not q.passed_guardrails)
    parts = []
    if py_errors:
        parts.append(f"{py_errors} python analysis error(s)")
    if sql_errors:
        parts.append(f"{sql_errors} SQL error(s)")
    if rejected_sql:
        parts.append(f"{rejected_sql} SQL rejected by guardrails")
    return ", ".join(parts) if parts else "none"


def _fallback_brief(state: InvestigationState) -> ExecutiveBrief:
    """Safe default if the LLM Writer call fails."""
    # Try to salvage anything we can from state.hypotheses
    salvageable = [
        h for h in state.hypotheses
        if h.confidence >= 0.5 and h.status != HypothesisStatus.REFUTED
    ]
    ranked = [
        RankedCause(
            rank=i + 1,
            cause=h.statement,
            confidence=h.confidence,
            evidence_ids=h.supporting_evidence_ids,
            contribution_estimate=None,
            recommended_action=None,
        )
        for i, h in enumerate(sorted(salvageable, key=lambda x: x.confidence, reverse=True))
    ]
    return ExecutiveBrief(
        tl_dr=(
            f"WRITER FALLBACK: LLM synthesis failed. Preserved {len(ranked)} hypotheses "
            f"from state at confidence >= 0.5."
        ),
        ranked_causes=ranked,
        chart_refs=[],
        caveats=["Writer node LLM call failed; brief is a mechanical fallback, not a synthesis."],
        recommended_next_actions=["Re-run the investigation."],
    )


def writer_node(state: InvestigationState) -> dict:
    """Real Writer: LLM synthesizes ExecutiveBrief from accumulated state."""
    step = state.current_step + 1
    print(f"writer: synthesizing executive brief (step {step})")

    sq = state.structured_question
    magnitude_str = f"{sq.magnitude_pct}%" if sq and sq.magnitude_pct is not None else "unspecified"

    user_msg = USER_TEMPLATE.format(
        question=state.original_question,
        kpi=sq.kpi_name if sq else "?",
        direction=sq.direction if sq else "?",
        magnitude=magnitude_str,
        comp=f"{sq.time_window.get('start', '?')}..{sq.time_window.get('end', '?')}" if sq else "?",
        base=f"{sq.comparison_window.get('start', '?')}..{sq.comparison_window.get('end', '?')}" if sq else "?",
        n_hyp=len(state.hypotheses),
        hyp_summary=_summarize_hypotheses(state),
        n_evi=len(state.evidence),
        evi_summary=_summarize_evidence(state),
        n_sql=len(state.sql_queries),
        n_pa=len(state.python_analyses),
        steps=state.current_step,
        errors_summary=_errors_summary(state),
        router_decision=state.router_decision.value if state.router_decision else "?",
        router_rationale=state.router_rationale or "?",
    )

    llm = get_structured_llm(WriterOutput)

    try:
        output: WriterOutput = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except (ValidationError, Exception) as e:
        print(f"  writer LLM call failed: {e}. Using fallback.")
        brief = _fallback_brief(state)
        log_entry = ActionLogEntry(
            step=step,
            node=NodeName.WRITER,
            action="writer_llm_failed_fallback",
            input_summary=str(e)[:120],
            output_summary=f"{len(brief.ranked_causes)} salvaged causes",
        )
        return {
            "final_brief": brief,
            "status": InvestigationStatus.CONCLUDED,
            "current_step": step,
            "current_node": NodeName.WRITER,
            "action_log": [log_entry],
            "completed_at": datetime.utcnow(),
        }

    # Convert drafts to final RankedCause objects, pulling evidence_ids
    # from the referenced hypothesis. This is what makes the brief
    # auditable back to specific findings.
    hyp_by_id = {h.id: h for h in state.hypotheses}
    ranked_causes: list[RankedCause] = []
    for draft in output.ranked_causes:
        source_hyp = hyp_by_id.get(draft.hypothesis_id)
        evidence_ids = list(source_hyp.supporting_evidence_ids) if source_hyp else []
        ranked_causes.append(
            RankedCause(
                rank=draft.rank,
                cause=draft.cause,
                confidence=draft.confidence,
                evidence_ids=evidence_ids,
                contribution_estimate=draft.contribution_estimate,
                recommended_action=draft.recommended_action,
            )
        )

    # Ensure rank ordering by confidence, break ties by rank
    ranked_causes.sort(key=lambda r: (-r.confidence, r.rank))
    for i, rc in enumerate(ranked_causes):
        rc.rank = i + 1

    brief = ExecutiveBrief(
        tl_dr=output.tl_dr,
        ranked_causes=ranked_causes,
        chart_refs=[],
        caveats=output.caveats,
        recommended_next_actions=output.recommended_next_actions,
    )

    print(f"  brief: {len(ranked_causes)} ranked cause(s), {len(output.caveats)} caveat(s)")

    log_entry = ActionLogEntry(
        step=step,
        node=NodeName.WRITER,
        action="write_brief",
        input_summary=f"{len(state.hypotheses)} hyp, {len(state.evidence)} evi",
        output_summary=f"{len(ranked_causes)} causes: {output.tl_dr[:100]}",
    )

    return {
        "final_brief": brief,
        "status": InvestigationStatus.CONCLUDED,
        "current_step": step,
        "current_node": NodeName.WRITER,
        "action_log": [log_entry],
        "completed_at": datetime.utcnow(),
    }