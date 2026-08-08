"""
scripts/test_metrics.py

Unit tests for evals/metrics.py scoring functions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import score_single_cause, score_multi_cause, score_null_case


def _fake_output(causes: list[tuple[str, float]], tl_dr: str = "") -> dict:
    return {
        "final_brief": {
            "tl_dr": tl_dr,
            "ranked_causes": [
                {"rank": i + 1, "cause": text, "confidence": conf, "evidence_ids": []}
                for i, (text, conf) in enumerate(causes)
            ],
        }
    }


def test_single_cause_full_credit():
    gt = {
        "id": "olist_001",
        "case_type": "single_cause",
        "ground_truth": {
            "primary_cause": {
                "dimension": "product_category",
                "statement": "Growth driven by health_beauty and watches_gifts categories",
                "supporting_dimensions": ["product_category_english"],
            },
            "expected_confidence": 0.7,
        },
    }
    agent = _fake_output([
        ("Revenue growth driven by product_category expansion in health_beauty", 0.75),
    ])
    r = score_single_cause(agent, gt)
    assert r.score == 1.0, f"Expected 1.0, got {r.score}: {r.reason}"
    print(f"[PASS] single_cause full credit: {r.score}")


def test_single_cause_wrong_dimension():
    gt = {
        "id": "olist_002",
        "case_type": "single_cause",
        "ground_truth": {
            "primary_cause": {
                "dimension": "customer_state",
                "statement": "SP state expansion",
                "supporting_dimensions": [],
            },
            "expected_confidence": 0.85,
        },
    }
    agent = _fake_output([
        ("Revenue growth driven by product category shifts", 0.80),
    ])
    r = score_single_cause(agent, gt)
    assert r.score == 0.5, f"Expected 0.5 (conf hit, dim miss), got {r.score}: {r.reason}"
    print(f"[PASS] single_cause wrong dim, right conf: {r.score}")


def test_single_cause_low_confidence():
    gt = {
        "id": "olist_003",
        "case_type": "single_cause",
        "ground_truth": {
            "primary_cause": {
                "dimension": "product_category",
                "statement": "Bed bath table category",
                "supporting_dimensions": [],
            },
            "expected_confidence": 0.8,
        },
    }
    agent = _fake_output([
        ("Product_category shift toward bed_bath_table", 0.35),
    ])
    r = score_single_cause(agent, gt)
    assert r.score == 0.5, f"Expected 0.5 (dim hit, conf miss), got {r.score}: {r.reason}"
    print(f"[PASS] single_cause right dim, low conf: {r.score}")


def test_single_cause_empty():
    gt = {
        "id": "olist_004",
        "case_type": "single_cause",
        "ground_truth": {
            "primary_cause": {"dimension": "seller_id", "statement": "", "supporting_dimensions": []},
            "expected_confidence": 0.7,
        },
    }
    agent = _fake_output([])
    r = score_single_cause(agent, gt)
    assert r.score == 0.0
    print(f"[PASS] single_cause empty ranked_causes: {r.score}")


def test_multi_cause_both_hit():
    gt = {
        "id": "olist_015",
        "case_type": "multi_cause",
        "ground_truth": {
            "primary_causes": [
                {"dimension": "customer_state x product_category", "statement": "SP expansion"},
                {"dimension": "product_category x seller_id", "statement": "New category emergence"},
            ],
            "eval_rules": {"credit_partial_if_one_hit": 0.4},
        },
    }
    agent = _fake_output([
        ("Growth driven by customer_state expansion in São Paulo", 0.65),
        ("New product_category emergence from seller_id onboarding", 0.60),
    ])
    r = score_multi_cause(agent, gt)
    assert r.score == 1.0, f"Expected 1.0, got {r.score}: {r.reason}"
    print(f"[PASS] multi_cause both hit: {r.score}")


def test_multi_cause_one_hit():
    gt = {
        "id": "olist_015",
        "case_type": "multi_cause",
        "ground_truth": {
            "primary_causes": [
                {"dimension": "customer_state x product_category", "statement": "SP expansion"},
                {"dimension": "product_category x seller_id", "statement": "New category emergence"},
            ],
            "eval_rules": {"credit_partial_if_one_hit": 0.4},
        },
    }
    agent = _fake_output([
        ("Growth from customer_state expansion in SP", 0.65),
        ("Overall marketplace growth", 0.55),
    ])
    r = score_multi_cause(agent, gt)
    assert r.score == 0.4, f"Expected 0.4 (partial), got {r.score}: {r.reason}"
    print(f"[PASS] multi_cause one hit (partial credit): {r.score}")


def test_null_case_empty_perfect():
    gt = {
        "id": "olist_017",
        "case_type": "null_case",
        "ground_truth": {
            "must_not_claim": ["SP shift as primary cause", "any specific category as primary cause"],
            "expected_confidence": 0.2,
        },
    }
    agent = _fake_output([])
    r = score_null_case(agent, gt)
    assert r.score == 1.0
    print(f"[PASS] null_case empty ranked_causes: {r.score}")


def test_null_case_low_confidence_ok():
    gt = {
        "id": "olist_017",
        "case_type": "null_case",
        "ground_truth": {
            "must_not_claim": ["SP shift"],
            "expected_confidence": 0.2,
        },
    }
    agent = _fake_output([
        ("Possible category mix drift", 0.35),
    ])
    r = score_null_case(agent, gt)
    assert r.score == 1.0
    print(f"[PASS] null_case low confidence stays right: {r.score}")


def test_null_case_forbidden_cause():
    gt = {
        "id": "olist_017",
        "case_type": "null_case",
        "ground_truth": {
            "must_not_claim": ["SP shift", "category mix"],
            "expected_confidence": 0.2,
        },
    }
    agent = _fake_output([
        ("Revenue decline driven by SP shift toward lower-price sellers", 0.75),
    ])
    r = score_null_case(agent, gt)
    assert r.score == 0.0
    print(f"[PASS] null_case forbidden cause claimed: {r.score}")


def test_null_case_overconfident_but_ok():
    gt = {
        "id": "olist_018",
        "case_type": "null_case",
        "ground_truth": {
            "must_not_claim": ["geographic expansion as sole cause"],
            "expected_confidence": 0.25,
        },
    }
    agent = _fake_output([
        ("Slight drift in seller processing time", 0.60),
    ])
    r = score_null_case(agent, gt)
    assert r.score == 0.5
    print(f"[PASS] null_case overconfident but not forbidden: {r.score}")


if __name__ == "__main__":
    tests = [
        test_single_cause_full_credit,
        test_single_cause_wrong_dimension,
        test_single_cause_low_confidence,
        test_single_cause_empty,
        test_multi_cause_both_hit,
        test_multi_cause_one_hit,
        test_null_case_empty_perfect,
        test_null_case_low_confidence_ok,
        test_null_case_forbidden_cause,
        test_null_case_overconfident_but_ok,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{len(tests)} tests passed")