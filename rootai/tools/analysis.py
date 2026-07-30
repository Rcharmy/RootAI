"""
rootai/tools/analysis.py

Restricted analytical toolbox. The Python Analyst node picks one of these
functions per hop, based on the current SQL result. The LLM chooses
which tool and what arguments; it does NOT write arbitrary Python.

Design rationale:
- Full LLM-generated Python via exec() would be more flexible but is a
  standard interview red flag. A whitelisted toolbox is auditable, safe,
  and forces the LLM to reason in terms of analytical primitives rather
  than raw code.
- Each function takes a pandas DataFrame and returns a compact
  AnalysisResult (dict of numbers plus a short summary string). Compact
  return values keep the trace and downstream LLM context small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AnalysisResult:
    """Return type from every analysis function. JSON-serializable."""
    tool_name: str
    summary: str
    findings: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def contribution_analysis(
    df: pd.DataFrame,
    baseline_col: str,
    comparison_col: str,
    dimension_col: str,
    top_k: int = 5,
) -> AnalysisResult:
    """
    Decompose the change from baseline to comparison into per-dimension
    contributions. Returns the top-K contributors (positive and negative).

    Requires df to have at least dimension_col, baseline_col, comparison_col.
    """
    tool = "contribution_analysis"
    try:
        for col in (dimension_col, baseline_col, comparison_col):
            if col not in df.columns:
                return AnalysisResult(
                    tool_name=tool,
                    summary=f"missing column: {col}",
                    error=f"df must contain column {col}",
                )

        work = df[[dimension_col, baseline_col, comparison_col]].copy()
        work[baseline_col] = pd.to_numeric(work[baseline_col], errors="coerce").fillna(0)
        work[comparison_col] = pd.to_numeric(work[comparison_col], errors="coerce").fillna(0)
        work["delta"] = work[comparison_col] - work[baseline_col]

        total_baseline = float(work[baseline_col].sum())
        total_comparison = float(work[comparison_col].sum())
        total_delta = total_comparison - total_baseline
        pct_change = (total_delta / total_baseline * 100.0) if total_baseline else None

        work_sorted = work.sort_values("delta", key=lambda s: s.abs(), ascending=False)
        top = work_sorted.head(top_k)

        contributors = []
        for _, row in top.iterrows():
            share = (row["delta"] / total_delta * 100.0) if total_delta else None
            contributors.append({
                "dimension_value": str(row[dimension_col]),
                "baseline": float(row[baseline_col]),
                "comparison": float(row[comparison_col]),
                "delta": float(row["delta"]),
                "share_of_total_delta_pct": (round(share, 1) if share is not None else None),
            })

        # Concentration signal: how much of the total delta is explained by top-K
        top_delta_sum = float(top["delta"].sum())
        concentration = (top_delta_sum / total_delta * 100.0) if total_delta else None

        summary_lines = [
            f"total baseline={total_baseline:,.0f}, comparison={total_comparison:,.0f}, delta={total_delta:+,.0f}"
            + (f" ({pct_change:+.1f}%)" if pct_change is not None else ""),
            f"top {top_k} dimension values explain "
            + (f"{concentration:.0f}%" if concentration is not None else "?%")
            + " of the total delta",
        ]

        return AnalysisResult(
            tool_name=tool,
            summary=". ".join(summary_lines),
            findings={
                "total_baseline": total_baseline,
                "total_comparison": total_comparison,
                "total_delta": total_delta,
                "pct_change": pct_change,
                "top_k_concentration_pct": concentration,
                "top_contributors": contributors,
            },
        )
    except Exception as e:
        return AnalysisResult(tool_name=tool, summary=f"error: {e}", error=str(e))


def top_k_by_dimension(
    df: pd.DataFrame,
    dimension_col: str,
    metric_col: str,
    top_k: int = 10,
) -> AnalysisResult:
    """
    Return the top-K dimension values by a single metric. Useful when the
    SQL Explorer produced a one-window aggregate and we want to identify
    the biggest contributors.
    """
    tool = "top_k_by_dimension"
    try:
        for col in (dimension_col, metric_col):
            if col not in df.columns:
                return AnalysisResult(
                    tool_name=tool,
                    summary=f"missing column: {col}",
                    error=f"df must contain column {col}",
                )

        work = df[[dimension_col, metric_col]].copy()
        work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce").fillna(0)
        total = float(work[metric_col].sum())
        work_sorted = work.sort_values(metric_col, ascending=False).head(top_k)

        rows = []
        for _, row in work_sorted.iterrows():
            share = (row[metric_col] / total * 100.0) if total else None
            rows.append({
                "dimension_value": str(row[dimension_col]),
                "metric": float(row[metric_col]),
                "share_of_total_pct": (round(share, 1) if share is not None else None),
            })

        top_share = float(work_sorted[metric_col].sum()) / total * 100.0 if total else None

        return AnalysisResult(
            tool_name=tool,
            summary=f"top {top_k} of {dimension_col} account for "
                    + (f"{top_share:.0f}%" if top_share is not None else "?%")
                    + f" of total {metric_col} ({total:,.0f})",
            findings={
                "total": total,
                "top_k_share_pct": top_share,
                "top_rows": rows,
            },
        )
    except Exception as e:
        return AnalysisResult(tool_name=tool, summary=f"error: {e}", error=str(e))


def pct_change_summary(
    df: pd.DataFrame,
    baseline_col: str,
    comparison_col: str,
    dimension_col: str,
) -> AnalysisResult:
    """
    Cross-dimension percent change summary. For each dimension value,
    compute pct change. Useful for identifying which dimensions moved
    disproportionately even if their absolute delta is small.
    """
    tool = "pct_change_summary"
    try:
        for col in (dimension_col, baseline_col, comparison_col):
            if col not in df.columns:
                return AnalysisResult(
                    tool_name=tool,
                    summary=f"missing column: {col}",
                    error=f"df must contain column {col}",
                )

        work = df[[dimension_col, baseline_col, comparison_col]].copy()
        work[baseline_col] = pd.to_numeric(work[baseline_col], errors="coerce").fillna(0)
        work[comparison_col] = pd.to_numeric(work[comparison_col], errors="coerce").fillna(0)
        # Avoid divide-by-zero: only compute pct where baseline > 0
        work["pct_change"] = ((work[comparison_col] - work[baseline_col]) / work[baseline_col].replace(0, pd.NA)) * 100.0

        # Sort by absolute pct change, ignoring NaN
        work_sorted = work.dropna(subset=["pct_change"]).sort_values(
            "pct_change", key=lambda s: s.abs(), ascending=False
        ).head(10)

        rows = []
        for _, row in work_sorted.iterrows():
            rows.append({
                "dimension_value": str(row[dimension_col]),
                "baseline": float(row[baseline_col]),
                "comparison": float(row[comparison_col]),
                "pct_change": round(float(row["pct_change"]), 1),
            })

        mean_change = float(work["pct_change"].mean()) if len(work) else None

        return AnalysisResult(
            tool_name=tool,
            summary=(
                f"mean pct change across {dimension_col}: "
                + (f"{mean_change:+.1f}%" if mean_change is not None else "N/A")
                + f". Largest outliers by |pct change| returned in findings."
            ),
            findings={
                "mean_pct_change": mean_change,
                "outliers": rows,
            },
        )
    except Exception as e:
        return AnalysisResult(tool_name=tool, summary=f"error: {e}", error=str(e))


# Public registry the Python Analyst node reads to know what's available
TOOLS: dict[str, callable] = {
    "contribution_analysis": contribution_analysis,
    "top_k_by_dimension": top_k_by_dimension,
    "pct_change_summary": pct_change_summary,
}


TOOL_DESCRIPTIONS = {
    "contribution_analysis": (
        "Best for two-window comparisons (baseline vs comparison columns). "
        "Returns which dimension values contributed most to the total delta. "
        "Args: baseline_col (str), comparison_col (str), dimension_col (str), top_k (int, default 5). "
        "Use when the SQL result has a 'baseline_*' and 'comparison_*' column pair."
    ),
    "top_k_by_dimension": (
        "Best for single-window aggregates. Returns top-K dimension values by a metric. "
        "Args: dimension_col (str), metric_col (str), top_k (int, default 10). "
        "Use when the SQL result has one row per dimension value with a single metric column."
    ),
    "pct_change_summary": (
        "Best for finding disproportionate movers across dimensions. "
        "Returns per-dimension percent change plus the largest outliers by absolute pct change. "
        "Args: baseline_col (str), comparison_col (str), dimension_col (str). "
        "Use when small values with big pct swings are relevant (e.g. cancellation rates)."
    ),
}