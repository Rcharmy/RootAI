"""
rootai/graph.py

LangGraph state machine wiring. Assembles the 6-node investigation flow
and exposes a compiled graph for the app.

Flow:
    START
      -> Planner
          -> if needs_clarification: END (writer produces a clarification brief)
          -> else: SQL Explorer
      -> Python Analyst
      -> Hypothesis Former
      -> Router (decides CONTINUE / CONCLUDE / ABORT)
          -> if CONTINUE: back to SQL Explorer
          -> if CONCLUDE or ABORT: Writer
      -> END

Design notes:
- Router as a NODE, not a conditional edge. Trades one extra LLM call
  per hop for auditability of stopping decisions in the trace log.
- Planner has its own conditional edge for ambiguity handling. This is
  the Phase 4 ambiguity guardrail: if the question is truly ambiguous
  (no KPI mentioned, no window inferable), route directly to Writer
  which produces a clarification brief instead of running the full flow.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rootai.nodes.hypothesis_former import hypothesis_former_node
from rootai.nodes.planner import planner_node
from rootai.nodes.python_analyst import python_analyst_node
from rootai.nodes.router import router_node
from rootai.nodes.sql_explorer import sql_explorer_node
from rootai.nodes.writer import writer_node
from rootai.state import InvestigationState, RouterDecision


def route_after_planner(state: InvestigationState) -> str:
    """
    Read Planner's needs_clarification flag. If ambiguous, jump straight
    to Writer (which produces a clarification-flavored brief). Otherwise
    proceed to SQL Explorer.

    Not an LLM call; a pure state inspection.
    """
    # LangGraph may pass state as dict OR Pydantic model depending on version
    needs = state.get("needs_clarification") if isinstance(state, dict) else state.needs_clarification
    if needs:
        return "writer"
    return "sql_explorer"


def route_after_router(state: InvestigationState) -> str:
    """
    Read the Router node's decision from state and return the next node key.
    """
    decision = state.get("router_decision") if isinstance(state, dict) else state.router_decision
    if decision in (RouterDecision.CONCLUDE, RouterDecision.ABORT):
        return "writer"
    return "sql_explorer"


def build_graph():
    """Assemble the LangGraph state machine."""
    graph = StateGraph(InvestigationState)

    graph.add_node("planner", planner_node)
    graph.add_node("sql_explorer", sql_explorer_node)
    graph.add_node("python_analyst", python_analyst_node)
    graph.add_node("hypothesis_former", hypothesis_former_node)
    graph.add_node("router", router_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")

    # Ambiguity guardrail: Planner -> Writer if needs_clarification, else -> SQL Explorer
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "writer": "writer",
            "sql_explorer": "sql_explorer",
        },
    )

    graph.add_edge("sql_explorer", "python_analyst")
    graph.add_edge("python_analyst", "hypothesis_former")
    graph.add_edge("hypothesis_former", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sql_explorer": "sql_explorer",
            "writer": "writer",
        },
    )

    graph.add_edge("writer", END)

    return graph.compile()


compiled_graph = build_graph()