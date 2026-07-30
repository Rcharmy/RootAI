"""
rootai/state.py

State schema for the RootAI investigation agent.

Design notes:
- InvestigationState is the single source of truth passed between LangGraph nodes.
- Fields marked with Annotated reducers accumulate across node calls.
- Fields without reducers get replaced on each write (last-write-wins).
- Component models (Hypothesis, Evidence, etc.) are Pydantic so LLM structured
  outputs can be validated at node boundaries via with_structured_output.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from operator import add
from typing import Annotated, Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InvestigationStatus(str, Enum):
    PENDING = "pending"
    NEEDS_CLARIFICATION = "needs_clarification"
    RUNNING = "running"
    CONCLUDED = "concluded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class NodeName(str, Enum):
    PLANNER = "planner"
    SQL_EXPLORER = "sql_explorer"
    PYTHON_ANALYST = "python_analyst"
    HYPOTHESIS_FORMER = "hypothesis_former"
    ROUTER = "router"
    WRITER = "writer"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class RouterDecision(str, Enum):
    CONTINUE = "continue"
    REFINE = "refine"
    CONCLUDE = "conclude"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------

class DatasetContext(BaseModel):
    """Static description of the dataset the agent is investigating."""
    name: str = "olist_ecommerce"
    tables: dict[str, list[str]] = Field(default_factory=dict)
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    time_column: str = "order_purchase_timestamp"
    grain: Literal["order", "order_item", "customer"] = "order_item"
    notes: Optional[str] = None


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: f"h_{uuid4().hex[:8]}")
    statement: str
    rationale: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    refuting_evidence_ids: list[str] = Field(default_factory=list)
    dimensions_to_test: list[str] = Field(default_factory=list)
    created_at_step: int = 0
    updated_at_step: int = 0


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: f"e_{uuid4().hex[:8]}")
    step: int
    source_node: NodeName
    description: str
    finding: str
    supports_hypothesis_ids: list[str] = Field(default_factory=list)
    refutes_hypothesis_ids: list[str] = Field(default_factory=list)
    magnitude: Optional[float] = None
    data_ref: Optional[str] = None


class SQLQuery(BaseModel):
    id: str = Field(default_factory=lambda: f"q_{uuid4().hex[:8]}")
    step: int
    query: str
    rationale: str
    row_count: Optional[int] = None
    columns: list[str] = Field(default_factory=list)
    result_preview: Optional[str] = None
    result_ref: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    passed_guardrails: bool = True


class PythonAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: f"p_{uuid4().hex[:8]}")
    step: int
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    result_summary: Optional[str] = None
    result_ref: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class PriorInvestigation(BaseModel):
    id: str
    question: str
    similarity_score: float
    summary: str
    key_causes: list[str] = Field(default_factory=list)
    verdict_confidence: Optional[float] = None


class RankedCause(BaseModel):
    rank: int
    cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    contribution_estimate: Optional[str] = None
    recommended_action: Optional[str] = None


class ExecutiveBrief(BaseModel):
    tl_dr: str
    ranked_causes: list[RankedCause]
    chart_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)


class ActionLogEntry(BaseModel):
    step: int
    node: NodeName
    action: str
    input_summary: str
    output_summary: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BudgetTracker(BaseModel):
    max_steps: int = 15
    max_cost_usd: float = 0.10
    max_wall_seconds: int = 180
    steps_used: int = 0
    cost_usd: float = 0.0
    tokens_used: int = 0
    reason_stopped: Optional[str] = None

    def exceeded(self) -> bool:
        return (
            self.steps_used >= self.max_steps
            or self.cost_usd >= self.max_cost_usd
        )


class KPIQuestion(BaseModel):
    kpi_name: str
    direction: Literal["up", "down", "unknown"] = "unknown"
    magnitude_pct: Optional[float] = None
    time_window: dict[str, str] = Field(default_factory=dict)
    comparison_window: dict[str, str] = Field(default_factory=dict)
    grain: Optional[str] = None
    raw_question: str


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def upsert_by_id(existing: list[BaseModel], updates: list[BaseModel]) -> list[BaseModel]:
    """Merge reducer: replace items with matching id, append new ones.

    Used for Hypothesis and Evidence: these get revised as the investigation
    progresses (status flips, confidence updates). Plain list-append would
    produce duplicates.
    """
    if not updates:
        return existing
    if not existing:
        return list(updates)
    by_id = {item.id: item for item in existing}
    for item in updates:
        by_id[item.id] = item
    return list(by_id.values())


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------

class InvestigationState(BaseModel):
    """Root LangGraph state. Passed between nodes; each node returns a partial
    dict which LangGraph merges using the reducers declared below."""

    # Identity
    investigation_id: str = Field(default_factory=lambda: f"inv_{uuid4().hex[:10]}")
    original_question: str
    structured_question: Optional[KPIQuestion] = None

    # Status / clarification
    status: InvestigationStatus = InvestigationStatus.PENDING
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_response: Optional[str] = None

    # Dataset / config
    dataset: DatasetContext
    python_tool_mode: Literal["restricted", "free"] = "restricted"

    # Working memory (accumulating)
    hypotheses: Annotated[list[Hypothesis], upsert_by_id] = Field(default_factory=list)
    evidence: Annotated[list[Evidence], upsert_by_id] = Field(default_factory=list)
    sql_queries: Annotated[list[SQLQuery], add] = Field(default_factory=list)
    python_analyses: Annotated[list[PythonAnalysis], add] = Field(default_factory=list)
    dead_ends: Annotated[list[str], add] = Field(default_factory=list)

    # RAG memory
    similar_prior_investigations: list[PriorInvestigation] = Field(default_factory=list)

    # Control flow (last-write-wins)
    plan: Optional[str] = None
    current_step: int = 0
    current_node: Optional[NodeName] = None
    next_action: Optional[str] = None
    router_decision: Optional[RouterDecision] = None
    router_rationale: Optional[str] = None

    # Budget
    budget: BudgetTracker = Field(default_factory=BudgetTracker)

    # Trace (accumulating)
    action_log: Annotated[list[ActionLogEntry], add] = Field(default_factory=list)
    errors: Annotated[list[str], add] = Field(default_factory=list)

    # Output
    final_brief: Optional[ExecutiveBrief] = None

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")