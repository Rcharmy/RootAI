"""
rootai/guardrails/confidence_cap.py

Confidence cap by evidence count. Applied after the Hypothesis Former
runs, before hypotheses are written to state.

Design:
- Cap: 1 evidence -> 0.60, 2 -> 0.75, 3 -> 0.85, 4+ -> 0.95.
- The LLM can set confidence anywhere in [0, 1]; the cap only reduces
  the value, never raises it. So if the LLM was already cautious
  (e.g. 0.4 with 3 evidence records), the cap does nothing.
- Refuting evidence does NOT count toward the cap. Only supporting.
  This is deliberate: a refutation is negative signal, not positive
  weight.
"""
from __future__ import annotations

from rootai.state import Hypothesis


# Ordered list of (min_supporting_count, max_confidence)
_CAP_CURVE: list[tuple[int, float]] = [
    (0, 0.35),  # 0 evidence: max 0.35 (should not happen but defensive)
    (1, 0.60),  # 1 evidence: max 0.60
    (2, 0.75),  # 2 evidence: max 0.75
    (3, 0.85),  # 3 evidence: max 0.85
    (4, 0.95),  # 4+ evidence: max 0.95
]


def max_confidence_for(supporting_count: int) -> float:
    """Return the maximum confidence allowed for a given supporting-evidence count."""
    cap = 0.35
    for threshold, allowed in _CAP_CURVE:
        if supporting_count >= threshold:
            cap = allowed
    return cap


def apply_confidence_cap(hypothesis: Hypothesis) -> tuple[Hypothesis, bool]:
    """
    If the hypothesis's confidence exceeds the cap for its evidence count,
    return (capped_copy, True). Otherwise return (original, False).
    """
    n_supporting = len(hypothesis.supporting_evidence_ids)
    cap = max_confidence_for(n_supporting)
    if hypothesis.confidence > cap:
        return hypothesis.model_copy(update={"confidence": cap}), True
    return hypothesis, False