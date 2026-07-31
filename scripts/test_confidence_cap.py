"""Quick unit test for the confidence cap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rootai.guardrails.confidence_cap import apply_confidence_cap, max_confidence_for
from rootai.state import Hypothesis, HypothesisStatus


CASES = [
    # (evidence_ids_count, llm_confidence, expected_final_confidence, description)
    (0, 0.90, 0.35, "0 evidence, LLM says 0.90 -> capped to 0.35"),
    (1, 0.80, 0.60, "1 evidence, LLM says 0.80 -> capped to 0.60"),
    (1, 0.40, 0.40, "1 evidence, LLM says 0.40 -> no cap (already low)"),
    (2, 0.90, 0.75, "2 evidence, LLM says 0.90 -> capped to 0.75"),
    (3, 0.99, 0.85, "3 evidence, LLM says 0.99 -> capped to 0.85"),
    (4, 0.99, 0.95, "4 evidence, LLM says 0.99 -> capped to 0.95"),
    (10, 0.99, 0.95, "10 evidence, LLM says 0.99 -> capped to 0.95 (max)"),
    (2, 0.50, 0.50, "2 evidence, LLM says 0.50 -> no cap"),
]

failed = 0
for n_evi, llm_conf, expected, desc in CASES:
    hyp = Hypothesis(
        statement="test",
        rationale="test",
        status=HypothesisStatus.PROPOSED,
        confidence=llm_conf,
        supporting_evidence_ids=[f"e_{i}" for i in range(n_evi)],
    )
    capped, was_capped = apply_confidence_cap(hyp)
    ok = abs(capped.confidence - expected) < 0.001
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {desc} (got {capped.confidence:.2f}, capped={was_capped})")
    if not ok:
        failed += 1

print(f"\n{len(CASES) - failed}/{len(CASES)} checks passed")