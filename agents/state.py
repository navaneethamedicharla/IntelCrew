"""
State models for the Competitive Intelligence Briefing Crew LangGraph workflow.
Defines all Pydantic models and the central TypedDict used as graph state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    """Lifecycle status for each agent."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ClaimStatus(str, Enum):
    """Verification status for extracted claims."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class WorkflowPhase(str, Enum):
    """Current phase of the overall workflow."""

    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    WRITING = "writing"
    GOVERNANCE = "governance"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Source / Research models ──────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """A single researched source."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    url: str
    summary: str
    publication_date: Optional[str] = None
    domain: Optional[str] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    raw_content: Optional[str] = None
    is_trusted_domain: bool = False


class ResearchResult(BaseModel):
    """Aggregated research output."""

    sources: List[SourceDocument] = Field(default_factory=list)
    visited_urls: List[str] = Field(default_factory=list)
    failed_urls: List[str] = Field(default_factory=list)
    search_queries_used: List[str] = Field(default_factory=list)
    rag_chunks_used: int = 0
    total_sources: int = 0


# ── Analyst models ────────────────────────────────────────────────────────────


class Claim(BaseModel):
    """An extracted factual claim with verification metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    category: str  # pricing, product, partnership, acquisition, trend, risk
    source_ids: List[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence: Optional[str] = None
    rejection_reason: Optional[str] = None


class CompetitorProfile(BaseModel):
    """Profile of a single competitor."""

    name: str
    website: Optional[str] = None
    pricing_changes: List[str] = Field(default_factory=list)
    product_launches: List[str] = Field(default_factory=list)
    partnerships: List[str] = Field(default_factory=list)
    acquisitions: List[str] = Field(default_factory=list)
    competitive_advantages: List[str] = Field(default_factory=list)
    business_risks: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Output of the Analyst Agent."""

    competitor_profiles: List[CompetitorProfile] = Field(default_factory=list)
    extracted_claims: List[Claim] = Field(default_factory=list)
    market_signals: List[str] = Field(default_factory=list)
    technology_trends: List[str] = Field(default_factory=list)
    customer_trends: List[str] = Field(default_factory=list)
    market_movements: List[str] = Field(default_factory=list)
    # Second-pass LLM synthesis: keyed by section name, value is a ~150-word
    # structured narrative the Writer compresses into its ~100-word output.
    # Keys: executive_summary, competitor_pricing, product_updates,
    #       market_signals, business_risks, strategic_recommendations, opportunities
    section_syntheses: Dict[str, str] = Field(default_factory=dict)


# ── Verification models ───────────────────────────────────────────────────────


class VerificationResult(BaseModel):
    """Output of the Fact Verification Agent."""

    verified_claims: List[Claim] = Field(default_factory=list)
    rejected_claims: List[Claim] = Field(default_factory=list)
    unverified_claims: List[Claim] = Field(default_factory=list)
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_notes: List[str] = Field(default_factory=list)


# ── Report models ─────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A formatted citation."""

    id: str
    number: int
    title: str
    url: str
    domain: Optional[str] = None
    accessed_date: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))


class ReportSection(BaseModel):
    """A single section of the final report."""

    title: str
    content: str
    citations: List[str] = Field(default_factory=list)  # citation IDs


class FinalReport(BaseModel):
    """The complete generated report."""

    title: str
    topic: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    executive_summary: str = ""
    competitor_pricing: str = ""
    product_updates: str = ""
    market_signals: str = ""
    business_risks: str = ""
    strategic_recommendations: str = ""
    opportunities: str = ""
    references: List[Citation] = Field(default_factory=list)
    run_metadata: Dict[str, Any] = Field(default_factory=dict)
    audit_summary: str = ""
    markdown_content: str = ""      # Full markdown for in-app display (includes metadata)
    pdf_markdown: str = ""          # Lean markdown for PDF export (no metadata/audit pages)
    citation_coverage: float = 0.0
    overall_confidence: float = 0.0
    word_count: int = 0


# ── Governance models ─────────────────────────────────────────────────────────


class GovernanceCheckResult(BaseModel):
    """Result of governance validation."""

    passed: bool
    citation_coverage: float
    confidence_score: float
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    refusal_triggered: bool = False
    refusal_reason: Optional[str] = None
    checked_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Trace / Audit models ──────────────────────────────────────────────────────


class TraceEntry(BaseModel):
    """A single entry in the execution trace."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    agent: str
    phase: str
    message: str
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogEntry(BaseModel):
    """An audit log entry for governance / compliance."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    event_type: str
    agent: str
    description: str
    data: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"  # info | warning | error


class RunMetadata(BaseModel):
    """Metadata about a single workflow run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    topic: str = ""
    total_steps: int = 0
    total_sources: int = 0
    total_claims: int = 0
    verified_claims: int = 0
    rejected_claims: int = 0
    search_queries: int = 0
    tool_calls: int = 0
    errors: int = 0
    status: str = "running"


# ── Agent status tracker ──────────────────────────────────────────────────────


class AgentStatusTracker(BaseModel):
    """Tracks the status of each agent in the pipeline."""

    supervisor: AgentStatus = AgentStatus.PENDING
    research: AgentStatus = AgentStatus.PENDING
    analyst: AgentStatus = AgentStatus.PENDING
    fact_verification: AgentStatus = AgentStatus.PENDING
    writer: AgentStatus = AgentStatus.PENDING
    governance: AgentStatus = AgentStatus.PENDING


# ── LangGraph State TypedDict ─────────────────────────────────────────────────


class BriefingState(TypedDict, total=False):
    """
    Central state object passed between all LangGraph nodes.

    Every field is Optional so nodes only need to update what they change.
    The graph uses this as the single source of truth for the entire run.
    """

    # Run identity
    run_id: str
    topic: str
    max_steps: int
    max_sources: int
    current_step: int
    workflow_phase: str

    # Research
    research_result: Optional[ResearchResult]
    rag_enabled: bool

    # Analysis
    analysis_result: Optional[AnalysisResult]

    # Verification
    verification_result: Optional[VerificationResult]

    # Writing
    final_report: Optional[FinalReport]
    human_approved: Optional[bool]
    revision_requested: bool
    revision_count: int

    # Governance
    governance_result: Optional[GovernanceCheckResult]

    # Trace & audit
    execution_trace: List[TraceEntry]
    audit_log: List[AuditLogEntry]
    run_metadata: RunMetadata
    agent_status: AgentStatusTracker

    # Errors
    errors: List[str]
    warnings: List[str]

    # Control flow
    should_terminate: bool
    termination_reason: Optional[str]


def create_initial_state(
    topic: str,
    max_steps: int = 20,
    max_sources: int = 10,
    rag_enabled: bool = True,
) -> BriefingState:
    """Create a fresh initial state for a new workflow run."""
    run_id = str(uuid.uuid4())
    return BriefingState(
        run_id=run_id,
        topic=topic,
        max_steps=max_steps,
        max_sources=max_sources,
        current_step=0,
        workflow_phase=WorkflowPhase.INITIALIZING.value,
        research_result=None,
        rag_enabled=rag_enabled,
        analysis_result=None,
        verification_result=None,
        final_report=None,
        human_approved=None,
        revision_requested=False,
        revision_count=0,
        governance_result=None,
        execution_trace=[],
        audit_log=[],
        run_metadata=RunMetadata(run_id=run_id, topic=topic),
        agent_status=AgentStatusTracker(),
        errors=[],
        warnings=[],
        should_terminate=False,
        termination_reason=None,
    )
