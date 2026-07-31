"""
streamlit_app.py

Browser UI for RootAI.

"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

# Ensure DuckDB is built on Streamlit Cloud first-load. Local runs no-op.
from data.build_on_startup import ensure_data_ready
ensure_data_ready()

from typing import Any, Optional

import streamlit as st

from rootai.config import config
from rootai.graph import compiled_graph
from rootai.memory.store import (
    get_collection_stats,
    store_investigation,
)
from rootai.state import (
    ExecutiveBrief,
    InvestigationState,
    InvestigationStatus,
    KPIQuestion,
)
from rootai.tools.dataset_context import build_dataset_context
from rootai.tools.llm import get_current_usage, reset_usage


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RootAI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS. Kept in one place so the visual identity is easy to tune.
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.rootai-hero {
    background: linear-gradient(120deg, #1e3a8a 0%, #7c3aed 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(30, 58, 138, 0.15);
}
.rootai-hero h1 {
    color: white !important;
    margin: 0 0 0.4rem 0;
    font-size: 2.1rem;
    font-weight: 700;
}
.rootai-hero p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
    line-height: 1.5;
}

.cause-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #7c3aed;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.cause-card.high {
    border-left-color: #16a34a;
}
.cause-card.medium {
    border-left-color: #f59e0b;
}
.cause-card.low {
    border-left-color: #9ca3af;
}
.cause-title {
    font-weight: 600;
    font-size: 1.02rem;
    color: #111827;
    margin-bottom: 0.5rem;
}
.cause-meta {
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 0.4rem;
}

.confidence-bar-wrap {
    background: #f3f4f6;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed 0%, #4f46e5 100%);
}
.confidence-bar-fill.high {
    background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%);
}
.confidence-bar-fill.medium {
    background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
}
.confidence-bar-fill.low {
    background: linear-gradient(90deg, #9ca3af 0%, #d1d5db 100%);
}

.memory-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.memory-card-title {
    font-weight: 600;
    color: #4f46e5;
    font-size: 0.9rem;
}
.memory-card-sim {
    display: inline-block;
    background: #ede9fe;
    color: #6d28d9;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

.stat-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}
.stat-label {
    color: #6b7280;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.stat-value {
    color: #111827;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 0.15rem;
}

.node-progress-line {
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.35rem;
    background: #f9fafb;
    border-left: 3px solid #7c3aed;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #374151;
}

.section-header {
    font-weight: 700;
    color: #111827;
    font-size: 1.15rem;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #f3f4f6;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_total_tokens" not in st.session_state:
    st.session_state.session_total_tokens = 0
if "session_total_cost" not in st.session_state:
    st.session_state.session_total_cost = 0.0
if "session_investigations" not in st.session_state:
    st.session_state.session_investigations = 0


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_dataset_context():
    """Cache the DatasetContext across reruns."""
    return build_dataset_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_ICONS = {
    "planner": "🧭",
    "sql_explorer": "🗄️",
    "python_analyst": "🧮",
    "hypothesis_former": "💡",
    "router": "🚦",
    "writer": "📝",
}

NODE_LABELS = {
    "planner": "Planner",
    "sql_explorer": "SQL Explorer",
    "python_analyst": "Python Analyst",
    "hypothesis_former": "Hypothesis Former",
    "router": "Router",
    "writer": "Writer",
}

NODE_ACTIONS = {
    "planner": "Parsing the question and retrieving prior investigations",
    "sql_explorer": "Generating and executing SQL against DuckDB",
    "python_analyst": "Running an analytical tool on the query result",
    "hypothesis_former": "Reasoning about causes and updating hypotheses",
    "router": "Deciding whether to continue or conclude",
    "writer": "Synthesising the executive brief",
}


def _confidence_class(conf: float) -> str:
    if conf >= 0.7:
        return "high"
    if conf >= 0.5:
        return "medium"
    return "low"


def _brief_to_dict(brief: Any) -> Optional[dict]:
    if brief is None:
        return None
    if hasattr(brief, "model_dump"):
        return brief.model_dump()
    return brief


def _rebuild_state_for_memory(final_state: dict) -> Optional[InvestigationState]:
    if final_state.get("status") not in (
        InvestigationStatus.CONCLUDED,
        InvestigationStatus.CONCLUDED.value,
        "concluded",
    ):
        return None
    brief = final_state.get("final_brief")
    if brief is None:
        return None
    if isinstance(brief, dict):
        brief = ExecutiveBrief.model_validate(brief)
    sq = final_state.get("structured_question")
    if isinstance(sq, dict):
        sq = KPIQuestion.model_validate(sq)
    return InvestigationState(
        investigation_id=str(final_state.get("investigation_id", "unknown")),
        original_question=str(final_state.get("original_question", "")),
        structured_question=sq,
        status=InvestigationStatus.CONCLUDED,
        dataset=get_dataset_context(),
        final_brief=brief,
    )


def stream_investigation(question: str, progress_container) -> dict:
    """Streaming graph invocation with live per-node display."""
    reset_usage()
    dataset = get_dataset_context()

    initial_state = InvestigationState(
        original_question=question,
        dataset=dataset,
        status=InvestigationStatus.PENDING,
    )

    accumulated: dict = initial_state.model_dump()
    hops_completed = 0
    lines_shown: list[str] = []

    for update in compiled_graph.stream(
        initial_state.model_dump(),
        config={"recursion_limit": 30},
        stream_mode="updates",
    ):
        for node_name, node_return in update.items():
            hops_completed += 1
            label = NODE_LABELS.get(node_name, node_name)
            action = NODE_ACTIONS.get(node_name, "processing")
            icon = NODE_ICONS.get(node_name, "⚙️")

            for key, val in node_return.items():
                if key in ("sql_queries", "python_analyses", "action_log", "errors", "dead_ends", "hypotheses", "evidence"):
                    existing = accumulated.get(key) or []
                    if not isinstance(existing, list):
                        existing = []
                    if isinstance(val, list):
                        existing = list(existing) + list(val)
                    accumulated[key] = existing
                else:
                    accumulated[key] = val

            usage_now = get_current_usage()
            new_line = (
                f"<div class='node-progress-line'>"
                f"<b>{icon} Step {hops_completed}. {label}</b><br>"
                f"<span style='color:#6b7280'>{action}</span><br>"
                f"<span style='color:#6b7280;font-size:0.82rem'>"
                f"tokens: {usage_now['total_tokens']:,} &middot; cost: ${usage_now['cost_usd']:.4f}"
                f"</span></div>"
            )
            lines_shown.append(new_line)

            progress_container.markdown(
                "".join(lines_shown),
                unsafe_allow_html=True,
            )

    return accumulated


def render_stat_row(status: str, steps: int, tokens: int, cost: float) -> None:
    """Render the four-metric row using custom stat cards."""
    cols = st.columns(4)
    display_status = status.split(".")[-1] if status else "?"
    stats = [
        ("Status", display_status),
        ("Steps", str(steps)),
        ("Tokens", f"{tokens:,}"),
        ("Cost", f"${cost:.4f}"),
    ]
    for col, (label, val) in zip(cols, stats):
        with col:
            st.markdown(
                f"<div class='stat-card'>"
                f"<div class='stat-label'>{label}</div>"
                f"<div class='stat-value'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_cause_card(cause: dict) -> None:
    """Render one ranked cause as a styled card."""
    conf = float(cause.get("confidence", 0.0))
    cls = _confidence_class(conf)
    bar_pct = int(conf * 100)

    contribution = cause.get("contribution_estimate")
    action = cause.get("recommended_action")

    meta_parts = []
    if contribution:
        meta_parts.append(f"Contribution: {contribution}")
    if action:
        meta_parts.append(f"Recommended: {action}")
    meta_html = "<br>".join(meta_parts) if meta_parts else ""

    st.markdown(
        f"<div class='cause-card {cls}'>"
        f"<div class='cause-title'>{cause['rank']}. {cause['cause']}</div>"
        f"<div style='font-size:0.85rem;color:#6b7280'>Confidence {conf:.2f}</div>"
        f"<div class='confidence-bar-wrap'>"
        f"<div class='confidence-bar-fill {cls}' style='width:{bar_pct}%'></div>"
        f"</div>"
        + (f"<div class='cause-meta'>{meta_html}</div>" if meta_html else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def render_memory_card(p: Any) -> None:
    """Render one retrieved-from-memory record."""
    if isinstance(p, dict):
        p_id = p.get("id", "?")
        p_q = p.get("question", "?")
        p_sim = float(p.get("similarity_score", 0))
        p_causes = p.get("key_causes", []) or []
    else:
        p_id = p.id
        p_q = p.question
        p_sim = float(p.similarity_score)
        p_causes = p.key_causes or []

    top_cause = p_causes[0][:220] if p_causes else "(no top cause recorded)"
    st.markdown(
        f"<div class='memory-card'>"
        f"<span class='memory-card-title'>{p_id}</span>"
        f"<span class='memory-card-sim'>sim {p_sim:.2f}</span>"
        f"<div style='margin-top:0.4rem;color:#374151;font-size:0.88rem'>{p_q}</div>"
        f"<div style='margin-top:0.35rem;color:#6b7280;font-size:0.82rem'>{top_cause}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='rootai-hero'>"
    "<h1>🔎 RootAI</h1>"
    "<p>Autonomous KPI investigation agent. Ask why a metric moved. "
    "The agent queries DuckDB, forms hypotheses, and returns an executive "
    "brief with auditable evidence and cost accounting.</p>"
    "</div>",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Configuration")
    st.markdown(f"**Model:** `{config.groq_model}`")
    st.markdown("**Grain:** `order_item`")
    dataset = get_dataset_context()
    st.markdown(f"**Dimensions:** {len(dataset.dimensions)}")
    st.markdown(f"**Metrics:** {len(dataset.metrics)}")

    st.markdown("---")
    st.markdown("### Memory")
    mem_stats = get_collection_stats()
    st.markdown(f"**Stored investigations:** {mem_stats.get('count', 0)}")

    st.markdown("---")
    st.markdown("### This session")
    st.markdown(f"**Investigations run:** {st.session_state.session_investigations}")
    st.markdown(f"**Tokens used:** {st.session_state.session_total_tokens:,}")
    st.markdown(f"**Cost:** ${st.session_state.session_total_cost:.4f}")

    st.markdown("---")
    with st.expander("Example questions"):
        st.markdown(
            "- Revenue in Q2 2018 was much higher than Q2 2017. What drove the growth?\n"
            "- Why did average order value drop between 2017 and 2018?\n"
            "- What are the top revenue drivers in São Paulo state?\n"
            "- Which product categories grew the most in H1 2018?"
        )


question = st.text_area(
    "**What do you want to investigate?**",
    placeholder="e.g. Revenue in Q2 2018 was much higher than Q2 2017. What drove the growth?",
    height=120,
)

run_col, hint_col = st.columns([1, 4])
with run_col:
    run_clicked = st.button(
        "🚀 Run investigation",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )
with hint_col:
    if not question.strip():
        st.caption("Enter a question to enable the Run button.")


if run_clicked:
    st.markdown("<div class='section-header'>Live progress</div>", unsafe_allow_html=True)
    progress_container = st.empty()

    try:
        final_state = stream_investigation(question.strip(), progress_container)
    except Exception as e:
        st.error(f"Investigation failed: {e}")
        st.stop()

    reconstructed = _rebuild_state_for_memory(final_state)
    if reconstructed is not None:
        store_investigation(reconstructed)

    usage = get_current_usage()
    st.session_state.session_total_tokens += usage["total_tokens"]
    st.session_state.session_total_cost += usage["cost_usd"]
    st.session_state.session_investigations += 1

    status = final_state.get("status")
    steps = final_state.get("current_step", 0)
    inv_id = final_state.get("investigation_id", "unknown")

    st.markdown("<div class='section-header'>Results</div>", unsafe_allow_html=True)
    render_stat_row(str(status), steps, usage["total_tokens"], usage["cost_usd"])
    st.caption(f"Investigation ID: `{inv_id}`")

    brief = _brief_to_dict(final_state.get("final_brief"))
    if brief is None:
        st.warning("No final brief produced.")
    else:
        tab_brief, tab_sql, tab_python, tab_log = st.tabs([
            "📋 Executive brief",
            "🗄️ SQL queries",
            "🧮 Python analyses",
            "🔍 Action log",
        ])

        with tab_brief:
            st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)
            st.info(brief["tl_dr"])

            ranked = brief.get("ranked_causes", [])
            if ranked:
                st.markdown("<div class='section-header'>Ranked causes</div>", unsafe_allow_html=True)
                for cause in ranked:
                    render_cause_card(cause)
            else:
                st.info(
                    "No causes met the ranked-cause threshold (confidence 0.5 or higher). "
                    "This is often the correct answer for questions where the change is "
                    "broadly distributed."
                )

            if brief.get("caveats"):
                st.markdown("<div class='section-header'>Caveats</div>", unsafe_allow_html=True)
                for c in brief["caveats"]:
                    st.markdown(f"- {c}")

            if brief.get("recommended_next_actions"):
                st.markdown("<div class='section-header'>Recommended next actions</div>", unsafe_allow_html=True)
                for a in brief["recommended_next_actions"]:
                    st.markdown(f"- {a}")

            priors = final_state.get("similar_prior_investigations") or []
            if priors:
                st.markdown("<div class='section-header'>Retrieved from memory</div>", unsafe_allow_html=True)
                st.caption("Prior investigations the agent consulted for context.")
                for p in priors:
                    render_memory_card(p)

        with tab_sql:
            sql_queries = final_state.get("sql_queries") or []
            if not sql_queries:
                st.info("No SQL queries ran.")
            for i, q in enumerate(sql_queries, start=1):
                if isinstance(q, dict):
                    query_text = q.get("query", "")
                    rationale = q.get("rationale", "")
                    row_count = q.get("row_count")
                    error = q.get("error")
                    duration = q.get("duration_ms")
                else:
                    query_text = q.query
                    rationale = q.rationale
                    row_count = q.row_count
                    error = q.error
                    duration = q.duration_ms
                header = f"Query {i}"
                if error:
                    header += f" (error)"
                elif row_count is not None:
                    header += f" ({row_count} rows, {duration or 0} ms)"
                with st.expander(header, expanded=(i == 1)):
                    st.markdown(f"**Rationale:** {rationale}")
                    st.code(query_text, language="sql")
                    if error:
                        st.error(error)

        with tab_python:
            analyses = final_state.get("python_analyses") or []
            if not analyses:
                st.info("No Python analyses ran.")
            for i, a in enumerate(analyses, start=1):
                if isinstance(a, dict):
                    tool = a.get("tool_name", "?")
                    args = a.get("tool_args", {})
                    summary = a.get("result_summary", "") or ""
                    error = a.get("error")
                else:
                    tool = a.tool_name
                    args = a.tool_args
                    summary = a.result_summary or ""
                    error = a.error
                header = f"Analysis {i}: {tool}"
                if error:
                    header += " (error)"
                with st.expander(header, expanded=(i == 1)):
                    st.markdown(f"**Args:** `{args}`")
                    if summary:
                        st.text(summary[:2000])
                    if error:
                        st.error(error)

        with tab_log:
            action_log = final_state.get("action_log", []) or []
            if not action_log:
                st.info("No entries.")
            for entry in action_log:
                if isinstance(entry, dict):
                    step = entry.get("step")
                    node = entry.get("node")
                    action = entry.get("action")
                    out = entry.get("output_summary")
                else:
                    step = entry.step
                    node = entry.node
                    action = entry.action
                    out = entry.output_summary
                st.markdown(f"`[{step}]` **{node}** — {action}: {out}")