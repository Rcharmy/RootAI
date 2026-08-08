"""
evals/run_evals.py

Batch runner for the 20 labeled cases in evals/labeled_investigations.json.

Behavior:
- Reads labeled cases.
- For each case in the requested range: runs the agent, scores the result,
  appends one JSONL record to evals/results/results.jsonl.
- Resumable: appending is idempotent per case_id if you delete existing
  matching lines. Simpler in practice: use --start/--end to advance in
  small batches without repeating work.

Usage:
    python evals/run_evals.py                 # runs all cases
    python evals/run_evals.py --start 0 --end 5   # runs cases 0-4 (inclusive-start, exclusive-end)
    python evals/run_evals.py --case olist_001    # runs just one case
    python evals/run_evals.py --dry-run           # prints the plan without invoking the agent

Cost expectation: ~15-25K tokens per case on Llama 3.3 70B. Groq free tier
gives 100K tokens/day, so budget 3-5 cases per session.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import ScoreResult, score_case
from rootai.config import config
from rootai.tools.llm import get_current_usage


# Paths
LABELED_PATH = Path(__file__).resolve().parent / "labeled_investigations.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_DIR / "results.jsonl"


def load_cases() -> list[dict]:
    """Read the labeled cases JSON. Returns the list of case dicts."""
    if not LABELED_PATH.exists():
        raise FileNotFoundError(f"Labeled cases not found at {LABELED_PATH}")
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", []) or []
    if not cases:
        raise RuntimeError(f"No cases found in {LABELED_PATH}")
    return cases


def load_completed_case_ids() -> set[str]:
    """
    Return the set of case_ids already in results.jsonl. Used to skip
    cases that have already been scored. If results file does not exist,
    returns empty set.
    """
    if not RESULTS_PATH.exists():
        return set()
    ids: set[str] = set()
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("case_id")
                if cid:
                    ids.add(cid)
            except json.JSONDecodeError:
                continue
    return ids


def run_one_case(case: dict) -> dict:
    """
    Run the agent against one labeled case and score the result.
    Returns a JSONL-ready record.

    Import app inside the function so a --dry-run does not initialize the LLM.
    """
    # Deferred import so --dry-run does not construct the compiled graph
    from app import run_investigation

    case_id = case.get("id", "?")
    case_type = case.get("case_type", "?")
    question = case.get("question", "")
    difficulty = case.get("difficulty", "?")

    print(f"\n{'=' * 70}")
    print(f"CASE {case_id} ({case_type}, {difficulty})")
    print(f"Q: {question}")
    print("=" * 70)

    started = datetime.utcnow()
    error_str: str | None = None
    final_state: dict = {}

    try:
        final_state = run_investigation(question)
    except Exception as e:
        error_str = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    completed = datetime.utcnow()
    usage = get_current_usage()

    # Score the case
    if error_str:
        score = ScoreResult(
            case_id=case_id,
            case_type=case_type,
            score=0.0,
            reason=f"agent raised exception: {error_str}",
        )
    else:
        score = score_case(final_state, case)

    # Extract just the brief and metadata from final_state so the JSONL
    # record is compact. The full trace is already in traces/.
    brief = final_state.get("final_brief")
    if brief is not None and hasattr(brief, "model_dump"):
        brief = brief.model_dump()

    record = {
        "case_id": case_id,
        "case_type": case_type,
        "difficulty": difficulty,
        "question": question,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_sec": (completed - started).total_seconds(),
        "agent_status": str(final_state.get("status", "unknown")),
        "agent_steps": final_state.get("current_step", 0),
        "agent_investigation_id": final_state.get("investigation_id", "unknown"),
        "tokens_used": usage["total_tokens"],
        "cost_usd": usage["cost_usd"],
        "score": score.score,
        "score_reason": score.reason,
        "score_details": asdict(score),
        "agent_brief": brief,
        "error": error_str,
    }

    # Console summary
    print()
    print(f"SCORE: {score.score:.2f} ({score.reason})")
    print(f"tokens: {usage['total_tokens']:,}, cost: ${usage['cost_usd']:.4f}")
    return record


def append_record(record: dict) -> None:
    """Append one JSONL record to results.jsonl."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run RootAI evaluation against labeled cases.")
    parser.add_argument("--start", type=int, default=0, help="Starting index (inclusive). Default 0.")
    parser.add_argument("--end", type=int, default=None, help="Ending index (exclusive). Default: all remaining.")
    parser.add_argument("--case", type=str, default=None, help="Run only this case id (e.g. olist_003).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running the agent.")
    parser.add_argument("--force", action="store_true", help="Re-run cases already in results.jsonl.")
    args = parser.parse_args()

    cases = load_cases()
    completed_ids = load_completed_case_ids()

    # Filter cases per args
    if args.case:
        cases_to_run = [c for c in cases if c.get("id") == args.case]
        if not cases_to_run:
            print(f"No case with id={args.case} found. Available: {[c.get('id') for c in cases]}")
            return
    else:
        end = args.end if args.end is not None else len(cases)
        cases_to_run = cases[args.start:end]

    if not args.force:
        skipped = [c for c in cases_to_run if c.get("id") in completed_ids]
        cases_to_run = [c for c in cases_to_run if c.get("id") not in completed_ids]
        if skipped:
            print(f"Skipping {len(skipped)} already-completed cases: {[c.get('id') for c in skipped]}")
            print("Use --force to re-run.")

    if not cases_to_run:
        print("No cases to run. Exiting.")
        return

    print(f"Model: {config.groq_model}")
    print(f"Plan: {len(cases_to_run)} case(s) to run:")
    for c in cases_to_run:
        print(f"  - {c.get('id')} ({c.get('case_type')}, {c.get('difficulty')})")
    print()

    if args.dry_run:
        print("--dry-run: exiting without running.")
        return

    for case in cases_to_run:
        try:
            record = run_one_case(case)
            append_record(record)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Partial results saved.")
            break
        except Exception as e:
            print(f"UNEXPECTED ERROR on case {case.get('id')}: {e}")
            traceback.print_exc()
            error_record = {
                "case_id": case.get("id"),
                "case_type": case.get("case_type"),
                "difficulty": case.get("difficulty"),
                "question": case.get("question"),
                "error": f"runner error: {e}",
                "score": 0.0,
                "score_reason": f"runner error: {e}",
            }
            append_record(error_record)

    print("\n" + "=" * 70)
    print("Batch complete.")
    print(f"Results appended to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()