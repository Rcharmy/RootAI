"""
evals/metrics.py

Deterministic scoring functions for the 20 labeled cases in
evals/labeled_investigations.json.

Design:
- No LLM-as-judge. Scoring is string-containment + confidence-band checks.
  Auditable, reproducible, defensible in interviews.
- Three functions match the three case_type values in the labeled JSON:
  single_cause, multi_cause, null_case.
- Each function returns a ScoreResult with a numeric score in [0, 1] and
  a string 'reason' that goes into the report so a reader can see WHY the
  score was assigned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ScoreResult:
    """One case's score. Serializable via dataclasses.asdict()."""
    case_id: str
    case_type: str
    score: float
    reason: str
    primary_cause_hit: Optional[bool] = None
    contributing_hits: list[str] = field(default_factory=list)
    confidence_match: Optional[bool] = None


def _cause_matches(agent_cause_text: str, ground_truth_keywords: list[str]) -> bool:
    """Case-insensitive substring match on any keyword."""
    agent_lower = (agent_cause_text or "").lower()
    for kw in ground_truth_keywords:
        if kw and kw.lower() in agent_lower:
            return True
    return False


def _get_ranked_causes(agent_output: dict) -> list[dict]:
    brief = agent_output.get("final_brief")
    if brief is None:
        return []
    if hasattr(brief, "model_dump"):
        brief = brief.model_dump()
    return brief.get("ranked_causes", []) or []


def _get_tl_dr(agent_output: dict) -> str:
    brief = agent_output.get("final_brief")
    if brief is None:
        return ""
    if hasattr(brief, "model_dump"):
        brief = brief.model_dump()
    return brief.get("tl_dr", "") or ""


def _extract_keywords(cause_field: dict) -> list[str]:
    """Lift keywords from primary_cause dict: dimension, supporting_dimensions, top content words in statement."""
    kws: list[str] = []

    dim = cause_field.get("dimension") or ""
    if dim:
        for part in dim.replace(" x ", " ").replace(",", " ").split():
            if part.strip():
                kws.append(part.strip())

    for sd in cause_field.get("supporting_dimensions", []) or []:
        for part in str(sd).replace(" x ", " ").replace(",", " ").split():
            if part.strip():
                kws.append(part.strip())

    stmt = str(cause_field.get("statement", "")).lower()
    STOP = {"the", "a", "an", "in", "on", "of", "and", "or", "to", "for", "with", "at", "by", "from", "up", "down", "is", "was", "are", "were", "be", "been"}
    words = [w.strip(".,;:()") for w in stmt.split()]
    content_words = [w for w in words if len(w) > 4 and w not in STOP][:5]
    kws.extend(content_words)

    seen = set()
    deduped = []
    for k in kws:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            deduped.append(k)
    return deduped


def score_single_cause(agent_output: dict, gt: dict) -> ScoreResult:
    """Score single_cause. 1.0 = cause + confidence hit, 0.5 = one of two, 0 = neither."""
    case_id = gt.get("id", "?")
    ground_truth = gt.get("ground_truth", {})
    primary = ground_truth.get("primary_cause", {})
    expected_conf = float(ground_truth.get("expected_confidence", 0.7))
    conf_floor = expected_conf * 0.6

    ranked = _get_ranked_causes(agent_output)
    if not ranked:
        return ScoreResult(
            case_id=case_id, case_type="single_cause", score=0.0,
            reason="agent returned no ranked causes",
            primary_cause_hit=False, confidence_match=False,
        )

    top = ranked[0]
    top_cause_text = str(top.get("cause", ""))
    top_conf = float(top.get("confidence", 0.0))

    keywords = _extract_keywords(primary)
    keyword_hit = _cause_matches(top_cause_text, keywords)
    confidence_hit = top_conf >= conf_floor

    if keyword_hit and confidence_hit:
        return ScoreResult(
            case_id=case_id, case_type="single_cause", score=1.0,
            reason=f"top cause matches primary dimension ({keywords[:3]}), confidence {top_conf:.2f} >= {conf_floor:.2f}",
            primary_cause_hit=True, confidence_match=True,
        )
    if keyword_hit:
        return ScoreResult(
            case_id=case_id, case_type="single_cause", score=0.5,
            reason=f"top cause matches but confidence {top_conf:.2f} below {conf_floor:.2f}",
            primary_cause_hit=True, confidence_match=False,
        )
    if confidence_hit:
        return ScoreResult(
            case_id=case_id, case_type="single_cause", score=0.5,
            reason=f"confidence {top_conf:.2f} in band but top cause '{top_cause_text[:80]}' does not match keywords {keywords[:3]}",
            primary_cause_hit=False, confidence_match=True,
        )
    return ScoreResult(
        case_id=case_id, case_type="single_cause", score=0.0,
        reason=f"top cause '{top_cause_text[:80]}' does not match keywords {keywords[:3]}, confidence {top_conf:.2f} below {conf_floor:.2f}",
        primary_cause_hit=False, confidence_match=False,
    )


def score_multi_cause(agent_output: dict, gt: dict) -> ScoreResult:
    """Score multi_cause. 1.0 = all required causes identified, partial = subset."""
    case_id = gt.get("id", "?")
    ground_truth = gt.get("ground_truth", {})
    primary_causes = ground_truth.get("primary_causes", []) or []
    eval_rules = ground_truth.get("eval_rules", {}) or {}
    partial_credit = float(eval_rules.get("credit_partial_if_one_hit", 0.4))

    ranked = _get_ranked_causes(agent_output)
    if not ranked:
        return ScoreResult(
            case_id=case_id, case_type="multi_cause", score=0.0,
            reason="agent returned no ranked causes",
        )

    agent_texts = " ; ".join(str(c.get("cause", "")) for c in ranked)

    hits: list[str] = []
    for pc in primary_causes:
        kws = _extract_keywords(pc)
        if _cause_matches(agent_texts, kws):
            hits.append(pc.get("dimension", "unknown"))

    if len(hits) >= len(primary_causes):
        return ScoreResult(
            case_id=case_id, case_type="multi_cause", score=1.0,
            reason=f"identified all {len(primary_causes)} required causes: {hits}",
            contributing_hits=hits,
        )
    if len(hits) >= 1:
        return ScoreResult(
            case_id=case_id, case_type="multi_cause", score=partial_credit,
            reason=f"identified {len(hits)}/{len(primary_causes)} required causes: {hits} (partial credit {partial_credit})",
            contributing_hits=hits,
        )
    return ScoreResult(
        case_id=case_id, case_type="multi_cause", score=0.0,
        reason=f"identified 0/{len(primary_causes)} required causes. Agent ranked_causes: {agent_texts[:200]}",
        contributing_hits=[],
    )


def score_null_case(agent_output: dict, gt: dict) -> ScoreResult:
    """Score null_case. 1.0 = correctly declined (empty ranked_causes or all conf<0.5). 0 = forbidden cause claimed."""
    case_id = gt.get("id", "?")
    ground_truth = gt.get("ground_truth", {})
    must_not_claim = ground_truth.get("must_not_claim", []) or []

    ranked = _get_ranked_causes(agent_output)

    if not ranked:
        return ScoreResult(
            case_id=case_id, case_type="null_case", score=1.0,
            reason="agent correctly returned no ranked causes for a null case",
        )

    max_conf = max((float(c.get("confidence", 0.0)) for c in ranked), default=0.0)
    if max_conf < 0.5:
        return ScoreResult(
            case_id=case_id, case_type="null_case", score=1.0,
            reason=f"all ranked causes below 0.5 confidence (max: {max_conf:.2f})",
        )

    agent_texts = " ; ".join(str(c.get("cause", "")) for c in ranked)
    forbidden_hits = []
    for phrase in must_not_claim:
        if phrase and _cause_matches(agent_texts, [phrase]):
            forbidden_hits.append(phrase)

    if forbidden_hits:
        return ScoreResult(
            case_id=case_id, case_type="null_case", score=0.0,
            reason=f"agent confidently claimed forbidden cause(s): {forbidden_hits}. Max conf: {max_conf:.2f}",
        )

    return ScoreResult(
        case_id=case_id, case_type="null_case", score=0.5,
        reason=f"agent claimed a non-forbidden cause at confidence {max_conf:.2f} on a null case. Overconfident but not a specific failure.",
    )


def score_case(agent_output: dict, gt: dict) -> ScoreResult:
    """Dispatch to the right scorer based on gt['case_type']."""
    ct = gt.get("case_type", "single_cause")
    if ct == "multi_cause":
        return score_multi_cause(agent_output, gt)
    if ct == "null_case":
        return score_null_case(agent_output, gt)
    return score_single_cause(agent_output, gt)


def summarize_results(results: list[ScoreResult]) -> dict:
    """Aggregate ScoreResults into overall + by-type breakdown."""
    if not results:
        return {"n_cases": 0, "overall_score": 0.0, "by_type": {}}

    total = sum(r.score for r in results)
    overall = total / len(results)

    by_type: dict[str, dict] = {}
    for r in results:
        entry = by_type.setdefault(r.case_type, {"n": 0, "total_score": 0.0, "case_ids": []})
        entry["n"] += 1
        entry["total_score"] += r.score
        entry["case_ids"].append(r.case_id)
    for entry in by_type.values():
        entry["avg_score"] = entry["total_score"] / entry["n"]

    return {
        "n_cases": len(results),
        "overall_score": overall,
        "by_type": by_type,
    }