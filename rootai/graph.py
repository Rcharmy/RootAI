"""
rootai/graph.py

LangGraph state machine wiring. Assembles the 6-node investigation flow
and exposes a compiled graph for the app.

Flow (Phase 2 stub):
    START
      -> Planner (once, step 0)
      -> SQL Explorer
      -> Python Analyst
      -> Hypothesis Former
      -> Router (decides CONTINUE / CONCLUDE / ABORT)
          -> if CONTINUE: back to SQL Explorer
          -> if CONCLUDE or ABORT: Writer
      -> END

Design notes:
- Router as a NODE, not a conditional edge. This is a deliberate choice
  documented in design_decisions.md. Trades one extra LLM call per hop
  for auditability of stopping decisions in the trace log.
- The conditional_edge FUNCTION after the Router reads state.router_decision
  and returns the destination node key. That function is not itself an LLM
  call, it's a pure state inspection.
- InvestigationState is passed as the graph's state schema. LangGraph reads
  the Annotated[..., reducer] hints and merges partial returns from each node.
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


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_router(state: InvestigationState) -> str:
    """
    Read the Router node's decision from state and return the next node key.

    This is a plain state-inspection function, NOT an LLM call. The LLM
    reasoning happened inside router_node; this function only reads what
    the Router already wrote to state.router_decision.
    """
    decision = state.router_decision
    if decision in (RouterDecision.CONCLUDE, RouterDecision.ABORT):
        return "writer"
    # CONTINUE or REFINE both loop back to the SQL Explorer.
    # In Phase 3 REFINE will route differently (back to Hypothesis Former
    # with a specific hypothesis to dig into) but for the skeleton both
    # go the same way.
    return "sql_explorer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assemble the LangGraph state machine.

    Returns a compiled graph object with .invoke() and .stream() methods.
    """
    graph = StateGraph(InvestigationState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("sql_explorer", sql_explorer_node)
    graph.add_node("python_analyst", python_analyst_node)
    graph.add_node("hypothesis_former", hypothesis_former_node)
    graph.add_node("router", router_node)
    graph.add_node("writer", writer_node)

    # Linear edges: START -> planner -> sql_explorer -> python_analyst -> hypothesis_former -> router
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "sql_explorer")
    graph.add_edge("sql_explorer", "python_analyst")
    graph.add_edge("python_analyst", "hypothesis_former")
    graph.add_edge("hypothesis_former", "router")

    # Conditional edge: router decides next hop
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sql_explorer": "sql_explorer",
            "writer": "writer",
        },
    )

    # Writer is the terminal node
    graph.add_edge("writer", END)

    return graph.compile()


# Singleton compiled graph, imported by app.py
compiled_graph = build_graph()