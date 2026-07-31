"""
rootai/nodes/hypothesis_former.py

Hypothesis Former node. Reads accumulated SQL results, Python analyses,
and existing hypotheses; produces new hypotheses, updates to existing
ones, and evidence records that link them together.

Design:
- ONE structured LLM call returns both hypotheses and evidence. The
  Evidence.supports_hypothesis_ids and Evidence.refutes_hypothesis_ids
  fields carry the causal linkage.
- Hypothesis updates use the SAME id as the existing hypothesis. The
  state.upsert_by_id reducer in state.py replaces on id match. New
  hypotheses get a fresh id.
- Prompt explicitly permits INCONCLUSIVE verdicts.
- Prompt requires numeric support for confidence changes.
- Summaries are aggressively truncated to control token cost on Groq
  free tier (100k tokens per day).
- Confidence cap guardrail (rootai/guardrails/confidence_cap) is applied
  after evidence linking. LLM can set confidence anywhere in [0, 1] but
  we clip the upper bound based on how much supporting evidence exists.
"""
from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from rootai.guardrails.confidence_cap import apply_confidence_cap
from rootai.state import (
    ActionLogEntry,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    InvestigationState,
    NodeName,
)
from rootai.tools.llm import get_structured_llm


class HypothesisDraft(BaseModel):
    """A hypothesis the LLM is emitting or updating.

    If `id` matches an existing hypothesis, state's upsert_by_id reducer
    overwrites. If new, it appends.
    """
    id: str = Field(description="Existing hypothesis id (h_XXXXXXXX) to update, OR 'new' to create a fresh one.")
    statement: str = Field(description="1-2 sentence statement of the candidate cause.")
    rationale: str = Field(description="Why this hypothesis is plausible given the evidence so far.")
    status: str = Field(description="'proposed', 'supported', 'refuted', or 'inconclusive'.")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0. Move confidence only when specific findings numerically support or refute.")
    dimensions_to_test: list[str] = Field(default_factory=list, description="Dimensions worth testing next to further support or refute this hypothesis.")


class EvidenceDraft(BaseModel):
    """A finding the LLM is attaching to hypotheses."""
    description: str = Field(description="What was looked at (e.g. 'category-level contribution to Q2 revenue growth').")
    finding: str = Field(description="What was observed (e.g. 'watches_gifts contributed 13% of total growth').")
    supports_hypothesis_ids: list[str] = Field(default_factory=list, description="IDs of hypotheses this evidence supports.")
    refutes_hypothesis_ids: list[str] = Field(default_factory=list, description="IDs of hypotheses this evidence refutes.")
    magnitude: float | None = Field(default=None, description="Optional: estimated contribution or effect size as a decimal (0.13 = 13%).")


class FormerOutput(BaseModel):
    """LLM structured output for one Hypothesis Former call."""
    hypotheses: list[HypothesisDraft] = Field(description="New hypotheses to propose OR existing ones to update.")
    evidence: list[EvidenceDraft] = Field(description="Findings linked to one or more hypotheses.")
    reasoning: str = Field(description="1-3 sentence summary of what changed and why.")


SYSTEM_PROMPT = (
    "You are the Hypothesis Former for RootAI. Your job is to reason about causes given the evidence gathered so far.\n\n"
    "On each turn you may:\n"
    "1. PROPOSE new hypotheses about what caused the KPI movement.\n"
    "2. UPDATE existing hypotheses: raise confidence when a finding specifically supports it, lower confidence or mark REFUTED when a finding specifically contradicts it, or mark INCONCLUSIVE when evidence is broadly distributed with no clear driver.\n"
    "3. ATTACH evidence records to hypotheses via supports_hypothesis_ids / refutes_hypothesis_ids.\n\n"
    "Rules:\n"
    "- To UPDATE an existing hypothesis, use its exact id (h_XXXXXXXX) in the id field. To CREATE a new one, use 'new'.\n"
    "- Move confidence ONLY when a specific number in the findings supports or contradicts. Do not raise confidence for general 'activity' or 'exploration'.\n"
    "- INCONCLUSIVE is a valid, valuable verdict. If findings are broadly distributed (e.g. top-K concentration under 30%, or largest single-dimension contribution under 20% of total), prefer INCONCLUSIVE over inventing a cause.\n"
    "- Do not propose more than 3 new hypotheses per turn. Fewer, sharper hypotheses beat many vague ones.\n"
    "- Every hypothesis update MUST have at least one evidence record supporting or refuting it (or explaining the INCONCLUSIVE verdict).\n"
    "- If the user's question presupposes a direction (e.g. 'why did revenue drop?') but the data shows the opposite, propose a hypothesis stating the premise appears incorrect. Do NOT manufacture a cause that fits the false premise."
)


USER_TEMPLATE = (
    "Investigation question: {question}\n"
    "Structured question: kpi={kpi}, direction={direction}, magnitude={magnitude}, windows: comp={comp}, base={base}\n"
    "Plan: {plan}\n\n"
    "Recent SQL queries (last 2 of {n_sql}):\n{sql_summary}\n\n"
    "Recent Python analyses (last 2 of {n_pa}):\n{pa_summary}\n\n"
    "Existing hypotheses ({n_hyp}):\n{hyp_summary}\n\n"
    "Recent evidence (last 3 of {n_evi}):\n{evi_summary}\n\n"
    "Dead ends: {dead_ends}\n\n"
    "Produce your hypotheses (new and/or updates), evidence linking them to findings, and a brief reasoning summary."
)


def _summarize_sql(state: InvestigationState) -> str:
    if not state.sql_queries:
        return "(none)"
    lines = []
    for q in state.sql_queries[-2:]:
        status = "ERROR" if q.error else f"{q.row_count} rows"
        lines.append(f"- [{status}] rationale: {q.rationale[:150]}")
        if q.result_preview and not q.error:
            preview_lines = q.result_preview.split("\n")[:4]
            lines.append("  preview: " + " | ".join(preview_lines))
    return "\n".join(lines)


def _summarize_analyses(state: InvestigationState) -> str:
    if not state.python_analyses:
        return "(none)"
    lines = []
    for a in state.python_analyses[-2:]:
        marker = "ERROR" if a.error else "OK"
        lines.append(f"- [{marker}] {a.tool_name}: {(a.result_summary or a.error or '(no summary)')[:500]}")
    return "\n".join(lines)


def _summarize_hypotheses(state: InvestigationState) -> str:
    if not state.hypotheses:
        return "(none yet)"
    lines = []
    for h in state.hypotheses:
        lines.append(
            f"- {h.id} [status={h.status.value}, conf={h.confidence:.2f}]: {h.statement}"
        )
    return "\n".join(lines)


def _summarize_evidence(state: InvestigationState) -> str:
    if not state.evidence:
        return "(none yet)"
    lines = []
    for e in state.evidence[-3:]:
        supp = ",".join(e.supports_hypothesis_ids) or "-"
        ref = ",".join(e.refutes_hypothesis_ids) or "-"
        lines.append(f"- {e.id} (supports=[{supp}], refutes=[{ref}]): {e.finding[:200]}")
    return "\n".join(lines)


_STATUS_MAP = {
    "proposed": HypothesisStatus.PROPOSED,
    "supported": HypothesisStatus.SUPPORTED,
    "refuted": HypothesisStatus.REFUTED,
    "inconclusive": HypothesisStatus.INCONCLUSIVE,
}


def hypothesis_former_node(state: InvestigationState) -> dict:
    """Real Hypothesis Former: LLM proposes/updates hypotheses with linked evidence, then applies confidence cap."""
    step = state.current_step + 1
    print(f"hypothesis_former: reasoning about causes (step {step})")

    sq = state.structured_question
    magnitude_str = f"{sq.magnitude_pct}%" if sq and sq.magnitude_pct is not None else "unspecified"

    user_msg = USER_TEMPLATE.format(
        question=state.original_question,
        kpi=sq.kpi_name if sq else "?",
        direction=sq.direction if sq else "?",
        magnitude=magnitude_str,
        comp=f"{sq.time_window.get('start', '?')}..{sq.time_window.get('end', '?')}" if sq else "?",
        base=f"{sq.comparison_window.get('start', '?')}..{sq.comparison_window.get('end', '?')}" if sq else "?",
        plan=state.plan or "(no plan)",
        n_sql=len(state.sql_queries),
        sql_summary=_summarize_sql(state),
        n_pa=len(state.python_analyses),
        pa_summary=_summarize_analyses(state),
        n_hyp=len(state.hypotheses),
        hyp_summary=_summarize_hypotheses(state),
        n_evi=len(state.evidence),
        evi_summary=_summarize_evidence(state),
        dead_ends=", ".join(state.dead_ends) if state.dead_ends else "(none)",
    )

    llm = get_structured_llm(FormerOutput)

    try:
        output: FormerOutput = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
    except (ValidationError, Exception) as e:
        print(f"  hypothesis_former LLM call failed: {e}")
        log_entry = ActionLogEntry(
            step=step,
            node=NodeName.HYPOTHESIS_FORMER,
            action="former_llm_failed",
            input_summary=str(e)[:120],
            output_summary="no hypotheses formed this turn",
        )
        return {
            "current_step": step,
            "current_node": NodeName.HYPOTHESIS_FORMER,
            "action_log": [log_entry],
            "errors": [f"hypothesis_former: {e}"],
        }

    existing_ids = {h.id for h in state.hypotheses}
    new_id_map: dict[int, str] = {}
    processed_hypotheses: list[Hypothesis] = []

    for i, draft in enumerate(output.hypotheses):
        status_str = draft.status.strip().lower()
        status = _STATUS_MAP.get(status_str, HypothesisStatus.PROPOSED)

        if draft.id in existing_ids:
            original = next(h for h in state.hypotheses if h.id == draft.id)
            hyp = Hypothesis(
                id=draft.id,
                statement=draft.statement,
                rationale=draft.rationale,
                status=status,
                confidence=draft.confidence,
                supporting_evidence_ids=list(original.supporting_evidence_ids),
                refuting_evidence_ids=list(original.refuting_evidence_ids),
                dimensions_to_test=draft.dimensions_to_test,
                created_at_step=original.created_at_step,
                updated_at_step=step,
            )
        else:
            new_id = f"h_{uuid4().hex[:8]}"
            new_id_map[i] = new_id
            hyp = Hypothesis(
                id=new_id,
                statement=draft.statement,
                rationale=draft.rationale,
                status=status,
                confidence=draft.confidence,
                dimensions_to_test=draft.dimensions_to_test,
                created_at_step=step,
                updated_at_step=step,
            )

        processed_hypotheses.append(hyp)

    def resolve_ids(id_list: list[str]) -> list[str]:
        resolved: list[str] = []
        first_new_id = next(iter(new_id_map.values()), None)
        for x in id_list:
            xl = x.strip().lower()
            if xl in existing_ids:
                resolved.append(xl)
            elif xl == "new" and first_new_id:
                resolved.append(first_new_id)
            elif xl.isdigit():
                idx = int(xl)
                if idx in new_id_map:
                    resolved.append(new_id_map[idx])
                elif 0 <= idx < len(processed_hypotheses):
                    resolved.append(processed_hypotheses[idx].id)
            elif x in {h.id for h in processed_hypotheses}:
                resolved.append(x)
        return resolved

    processed_evidence: list[Evidence] = []
    for e in output.evidence:
        ev = Evidence(
            step=step,
            source_node=NodeName.HYPOTHESIS_FORMER,
            description=e.description,
            finding=e.finding,
            supports_hypothesis_ids=resolve_ids(e.supports_hypothesis_ids),
            refutes_hypothesis_ids=resolve_ids(e.refutes_hypothesis_ids),
            magnitude=e.magnitude,
        )
        processed_evidence.append(ev)

    hyp_by_id = {h.id: h for h in processed_hypotheses}
    for ev in processed_evidence:
        for hid in ev.supports_hypothesis_ids:
            if hid in hyp_by_id:
                if ev.id not in hyp_by_id[hid].supporting_evidence_ids:
                    hyp_by_id[hid].supporting_evidence_ids.append(ev.id)
        for hid in ev.refutes_hypothesis_ids:
            if hid in hyp_by_id:
                if ev.id not in hyp_by_id[hid].refuting_evidence_ids:
                    hyp_by_id[hid].refuting_evidence_ids.append(ev.id)

    # Apply the confidence cap based on final supporting evidence counts.
    # LLM can set confidence anywhere in [0, 1]; we clip the upper bound
    # based on how much evidence actually supports each hypothesis.
    capped_hypotheses = []
    cap_events = []
    for h in processed_hypotheses:
        capped, was_capped = apply_confidence_cap(h)
        capped_hypotheses.append(capped)
        if was_capped:
            cap_events.append(
                f"{h.id}: {h.confidence:.2f} -> {capped.confidence:.2f} "
                f"(supp={len(h.supporting_evidence_ids)})"
            )
    processed_hypotheses = capped_hypotheses

    if cap_events:
        print(f"  confidence cap applied to {len(cap_events)} hypothesis(es): {'; '.join(cap_events)}")

    print(f"  emitted {len(processed_hypotheses)} hypothesis draft(s), {len(processed_evidence)} evidence record(s)")

    log_entry = ActionLogEntry(
        step=step,
        node=NodeName.HYPOTHESIS_FORMER,
        action="form_hypotheses",
        input_summary=f"{len(state.sql_queries)} sql, {len(state.python_analyses)} pa, {len(state.hypotheses)} hyp",
        output_summary=f"{len(processed_hypotheses)} hyp / {len(processed_evidence)} evi. Reasoning: {output.reasoning[:120]}",
    )

    return {
        "hypotheses": processed_hypotheses,
        "evidence": processed_evidence,
        "current_step": step,
        "current_node": NodeName.HYPOTHESIS_FORMER,
        "action_log": [log_entry],
    }