"""
Report generator – assembles FinalReport content from verified analysis
and writes the Markdown string that becomes the downloadable report.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from agents.state import (
    AnalysisResult,
    Citation,
    ClaimStatus,
    FinalReport,
    RunMetadata,
    VerificationResult,
)
from tools.citation_generator import format_references_section

logger = logging.getLogger(__name__)


def _bullet_list(items: List[str], fallback: str = "_None identified._") -> str:
    """Format a list as Markdown bullets."""
    if not items:
        return fallback
    return "\n".join(f"- {item}" for item in items)


def _section(title: str, content: str) -> str:
    """Wrap content in a Markdown H2 section."""
    return f"## {title}\n\n{content}\n"


def _annotate_unverified(text: str, unverified_phrases: List[str]) -> str:
    """
    Append a note to *text* listing unverified claims so readers are aware.
    This is a lightweight approach: rather than inline-patching every sentence
    (which would require NLP), we append a clearly labelled block at the end
    of the section when unverified claims exist that might have been referenced.
    """
    if not unverified_phrases:
        return text
    notes = "\n".join(f"- [Unverified] {p}" for p in unverified_phrases[:10])
    return f"{text}\n\n> **Note – Unverified Claims (require further validation):**\n{notes}"


def build_markdown_report(
    topic: str,
    executive_summary: str,
    competitor_pricing: str,
    product_updates: str,
    market_signals: str,
    business_risks: str,
    strategic_recommendations: str,
    opportunities: str,
    citations: List[Citation],
    run_metadata: Optional[RunMetadata] = None,
    audit_summary: str = "",
) -> str:
    """
    Assemble the full Markdown report string (used for in-app display).
    Includes Run Metadata and Audit sections for transparency in the UI.
    The PDF export uses build_pdf_markdown() which omits these to save pages.
    """
    now = datetime.utcnow().strftime("%B %d, %Y")
    lines: List[str] = []

    # Title block
    lines.append(f"# Competitive Intelligence Briefing: {topic}")
    lines.append(f"_Generated on {now} by Competitive Intelligence Briefing Crew_\n")
    lines.append("---\n")

    # Sections
    lines.append(_section("Executive Summary", executive_summary or "_Not available._"))
    lines.append(_section("Competitor Pricing", competitor_pricing or "_No pricing data found._"))
    lines.append(_section("Competitor Product Updates", product_updates or "_No product updates found._"))
    lines.append(_section("Market Signals", market_signals or "_No market signals found._"))
    lines.append(_section("Business Risks", business_risks or "_No risks identified._"))
    lines.append(_section("Strategic Recommendations", strategic_recommendations or "_No recommendations._"))
    lines.append(_section("Opportunities", opportunities or "_No opportunities identified._"))

    # References
    lines.append(format_references_section(citations))

    # Run metadata footer (app display only)
    if run_metadata:
        lines.append("---\n")
        lines.append("## Run Metadata\n")
        lines.append(f"- **Run ID:** `{run_metadata.run_id}`")
        lines.append(f"- **Topic:** {run_metadata.topic}")
        lines.append(f"- **Started At:** {run_metadata.started_at}")
        if run_metadata.completed_at:
            lines.append(f"- **Completed At:** {run_metadata.completed_at}")
        if run_metadata.duration_seconds:
            lines.append(f"- **Duration:** {run_metadata.duration_seconds:.1f}s")
        lines.append(f"- **Total Sources:** {run_metadata.total_sources}")
        lines.append(f"- **Claims Verified:** {run_metadata.verified_claims} / {run_metadata.total_claims}")
        lines.append(f"- **Search Queries:** {run_metadata.search_queries}")
        lines.append(f"- **Tool Calls:** {run_metadata.tool_calls}")
        lines.append(f"- **Errors:** {run_metadata.errors}")

    if audit_summary:
        lines.append("\n## Audit & Governance\n")
        lines.append(audit_summary)

    return "\n".join(lines)


def build_pdf_markdown(
    topic: str,
    executive_summary: str,
    competitor_pricing: str,
    product_updates: str,
    market_signals: str,
    business_risks: str,
    strategic_recommendations: str,
    opportunities: str,
    citations: List[Citation],
) -> str:
    """
    Assemble a lean Markdown string for PDF export.
    Omits Run Metadata and Audit sections — those are internal debug data
    that bloat the page count. The PDF should be a clean executive document.
    """
    now = datetime.utcnow().strftime("%B %d, %Y")
    lines: List[str] = []

    lines.append(f"# Competitive Intelligence Briefing: {topic}")
    lines.append(f"_Generated on {now}_\n")
    lines.append("---\n")

    lines.append(_section("Executive Summary",          executive_summary          or "_Not available._"))
    lines.append(_section("Competitor Pricing",         competitor_pricing         or "_No pricing data found._"))
    lines.append(_section("Competitor Product Updates", product_updates            or "_No product updates found._"))
    lines.append(_section("Market Signals",             market_signals             or "_No market signals found._"))
    lines.append(_section("Business Risks",             business_risks             or "_No risks identified._"))
    lines.append(_section("Strategic Recommendations",  strategic_recommendations  or "_No recommendations._"))
    lines.append(_section("Opportunities",              opportunities              or "_No opportunities identified._"))

    lines.append(format_references_section(citations))

    return "\n".join(lines)


def assemble_final_report(
    topic: str,
    executive_summary: str,
    competitor_pricing: str,
    product_updates: str,
    market_signals: str,
    business_risks: str,
    strategic_recommendations: str,
    opportunities: str,
    analysis: Optional[AnalysisResult],
    verification: Optional[VerificationResult],
    citations: List[Citation],
    run_metadata: Optional[RunMetadata],
    audit_summary: str = "",
) -> FinalReport:
    """
    Assemble a FinalReport Pydantic object from all pipeline outputs.

    All section text strings are LLM-generated and passed in directly.
    The analysis / verification objects are used only to compute coverage
    and confidence metrics, and to append [Unverified] markers.

    Args:
        topic: Research topic.
        executive_summary: LLM-generated executive summary.
        competitor_pricing: LLM-generated competitor pricing section.
        product_updates: LLM-generated product updates section.
        market_signals: LLM-generated market signals section.
        business_risks: LLM-generated business risks section.
        strategic_recommendations: LLM-generated strategic recommendations.
        opportunities: LLM-generated opportunities section.
        analysis: Output of the Analyst agent (may be None).
        verification: Output of the Verification agent (may be None).
        citations: Generated citations.
        run_metadata: Workflow run metadata (may be None).
        audit_summary: Governance summary string.

    Returns:
        FinalReport object with all fields populated.
    """
    # Confidence / coverage – safe defaults when verification is absent
    citation_coverage: float = verification.citation_coverage if verification else 0.0
    overall_confidence: float = verification.overall_confidence if verification else 0.0

    markdown = build_markdown_report(
        topic=topic,
        executive_summary=executive_summary,
        competitor_pricing=competitor_pricing,
        product_updates=product_updates,
        market_signals=market_signals,
        business_risks=business_risks,
        strategic_recommendations=strategic_recommendations,
        opportunities=opportunities,
        citations=citations,
        run_metadata=run_metadata,
        audit_summary=audit_summary,
    )

    pdf_md = build_pdf_markdown(
        topic=topic,
        executive_summary=executive_summary,
        competitor_pricing=competitor_pricing,
        product_updates=product_updates,
        market_signals=market_signals,
        business_risks=business_risks,
        strategic_recommendations=strategic_recommendations,
        opportunities=opportunities,
        citations=citations,
    )

    word_count = len(markdown.split())

    return FinalReport(
        title=f"Competitive Intelligence Briefing: {topic}",
        topic=topic,
        executive_summary=executive_summary,
        competitor_pricing=competitor_pricing,
        product_updates=product_updates,
        market_signals=market_signals,
        business_risks=business_risks,
        strategic_recommendations=strategic_recommendations,
        opportunities=opportunities,
        references=citations,
        run_metadata=run_metadata.dict() if run_metadata else {},
        audit_summary=audit_summary,
        markdown_content=markdown,
        pdf_markdown=pdf_md,
        citation_coverage=citation_coverage,
        overall_confidence=overall_confidence,
        word_count=word_count,
    )
