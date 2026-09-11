"""
Writer Agent – generates the professional competitive intelligence report
sections using the LLM, then assembles the complete FinalReport.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import (
    add_audit,
    add_error,
    add_trace,
    call_with_retry,
    call_llm_with_retry,
    check_runaway,
    get_llm,
    increment_step,
    set_agent_status,
)
from agents.state import (
    AgentStatus,
    AnalysisResult,
    BriefingState,
    Claim,
    ClaimStatus,
    VerificationResult,
    WorkflowPhase,
)
from tools.citation_generator import generate_citations
from tools.report_generator import assemble_final_report

logger = logging.getLogger(__name__)
AGENT_NAME = "writer_agent"

_SYSTEM_PROMPT = """You are a Principal Analyst at a top-tier competitive intelligence firm writing for a C-suite audience.

Your job is to SUMMARIZE and PRIORITIZE — not to dump everything you know.

Core rules:
- Each section is a tight executive summary of that topic. Lead with the most important finding. Cut everything that doesn't add new information.
- Every sentence must earn its place: if it repeats something already said or adds no new fact, cut it.
- Be specific: name entities, numbers, dates from the sources. Never be vague.
- Structure: 1 opening insight sentence → 3-5 key points (bullets if listing, prose if analyzing) → 1 closing "so what."
- Verified findings: state confidently. Unverified: prefix with "Unconfirmed —".
- Include a compact comparison table only when 2+ entities have directly comparable data.
- Recommendations must name a specific action, not a category.
- The goal is a report a senior executive can read in under 5 minutes and act on immediately."""


# ── Context helpers ───────────────────────────────────────────────────────────


def _claims_text(claims: List[Claim], label: str = "verified") -> str:
    """Format a list of claims as a readable bullet list with status label."""
    if not claims:
        return f"No {label} claims available."
    lines: List[str] = []
    for c in claims[:35]:
        marker = "" if label == "verified" else " [Unverified]"
        evidence = ""
        if c.supporting_evidence and label == "verified":
            evidence = f' — Evidence: "{c.supporting_evidence[:100]}"'
        lines.append(
            f"- [{c.category}]{marker} {c.text} (confidence: {c.confidence:.0%}){evidence}"
        )
    return "\n".join(lines)


def _competitor_profiles_text(analysis: Optional[AnalysisResult]) -> str:
    """Summarise competitor profiles into a detailed, structured string for the LLM."""
    if not analysis or not analysis.competitor_profiles:
        return "No competitor profiles available."
    parts: List[str] = []
    for p in analysis.competitor_profiles[:8]:
        items: List[str] = []
        if hasattr(p, "market_position") and p.market_position:
            items.append(f"Position: {p.market_position}")
        if hasattr(p, "pricing_model") and p.pricing_model:
            items.append(f"Pricing model: {p.pricing_model}")
        if p.pricing_changes:
            items.append("Pricing: " + "; ".join(p.pricing_changes[:4]))
        if p.product_launches:
            items.append("Product launches: " + "; ".join(p.product_launches[:4]))
        if hasattr(p, "ai_capabilities") and p.ai_capabilities:
            items.append("AI capabilities: " + "; ".join(p.ai_capabilities[:4]))
        if p.partnerships:
            items.append("Partnerships: " + "; ".join(p.partnerships[:3]))
        if p.acquisitions:
            items.append("Acquisitions: " + "; ".join(p.acquisitions[:3]))
        if p.competitive_advantages:
            items.append("Advantages: " + "; ".join(p.competitive_advantages[:3]))
        if p.business_risks:
            items.append("Risks: " + "; ".join(p.business_risks[:3]))
        if hasattr(p, "recent_news") and p.recent_news:
            items.append("Recent news: " + "; ".join(p.recent_news[:2]))
        detail = "\n    ".join(items) if items else "Limited data available."
        website = f" ({p.website})" if p.website else ""
        parts.append(f"- {p.name}{website}:\n    {detail}")
    return "\n".join(parts)


def _source_summaries_text(research) -> str:
    """Return the top-8 source summaries with URLs for citation context."""
    if not research or not research.sources:
        return "No source summaries available."
    lines: List[str] = []
    for src in research.sources[:8]:
        summary = (src.summary or "").strip()
        if summary:
            lines.append(f"- [{src.title}] ({src.url}) {summary[:400]}")
    return "\n".join(lines) if lines else "No source summaries available."


def _build_context(
    research,
    analysis: Optional[AnalysisResult],
    verification: Optional[VerificationResult],
) -> str:
    """Assemble a rich context string for the LLM from all pipeline outputs.

    Structure (priority order):
    1. ANALYST SYNTHESES — pre-digested per-section narratives from the second
       LLM pass in the Analyst Agent.  These are the primary input.  The Writer
       compresses each one into its ~100-word section output.
    2. HIGH-PRIORITY CLAIMS — verified or high-confidence claims to supplement
       any section where synthesis is thin.
    3. Supporting detail — competitor profiles, signals, source summaries — so
       the Writer can pull in extra specifics if needed.
    """

    # ── 1. Analyst section syntheses (primary) ────────────────────────────────
    syntheses: dict = {}
    if analysis and analysis.section_syntheses:
        syntheses = analysis.section_syntheses
    if syntheses:
        synth_lines = "\n\n".join(
            f"[{key.upper().replace('_', ' ')}]\n{text}"
            for key, text in syntheses.items()
        )
        synth_block = f"=== ANALYST SECTION SYNTHESES (primary — use these as your core input) ===\n{synth_lines}"
    else:
        synth_block = (
            "=== ANALYST SECTION SYNTHESES ===\n"
            "Not available — use the raw claims and signals below."
        )

    # ── 2. High-priority claims ────────────────────────────────────────────────
    verified = sorted(
        verification.verified_claims if verification else [],
        key=lambda c: c.confidence,
        reverse=True,
    )
    unverified = verification.unverified_claims if verification else []
    all_claims = verified + unverified
    high_priority = [
        c for c in all_claims
        if getattr(c, "strategic_importance", "medium") == "high" or c.confidence >= 0.8
    ][:10]
    high_priority_text = _claims_text(high_priority, "high-priority") if high_priority else "None flagged."

    verified_text  = _claims_text(verified,   "verified")
    unverified_text = _claims_text(unverified, "unverified")

    # ── 3. Competitor profiles + signals ──────────────────────────────────────
    competitors_text = _competitor_profiles_text(analysis)

    signals: List[str] = []
    tech_trends: List[str] = []
    customer_trends: List[str] = []
    market_movements: List[str] = []
    if analysis:
        signals        = analysis.market_signals[:12]
        tech_trends    = analysis.technology_trends[:8]
        customer_trends = analysis.customer_trends[:8]
        market_movements = getattr(analysis, "market_movements", [])[:8]

    signals_text   = "\n".join(f"- {s}" for s in signals)   or "No market signals identified."
    tech_text      = "\n".join(f"- {t}" for t in tech_trends) or "No technology trends identified."
    customer_text  = "\n".join(f"- {t}" for t in customer_trends) or "No customer trends identified."
    movements_text = "\n".join(f"- {m}" for m in market_movements) or "No market movements identified."

    sources_text = _source_summaries_text(research)

    return (
        f"{synth_block}\n\n"
        f"=== HIGH-PRIORITY CLAIMS (supplement syntheses where needed) ===\n{high_priority_text}\n\n"
        f"=== ALL VERIFIED CLAIMS ===\n{verified_text}\n\n"
        f"=== UNVERIFIED CLAIMS (use 'Unconfirmed —' prefix) ===\n{unverified_text}\n\n"
        f"=== COMPETITOR PROFILES ===\n{competitors_text}\n\n"
        f"=== MARKET SIGNALS ===\n{signals_text}\n\n"
        f"=== TECHNOLOGY TRENDS ===\n{tech_text}\n\n"
        f"=== CUSTOMER TRENDS ===\n{customer_text}\n\n"
        f"=== MARKET MOVEMENTS (M&A / Funding / Regulatory) ===\n{movements_text}\n\n"
        f"=== SOURCE SUMMARIES (TOP 8) ===\n{sources_text}"
    )


# ── LLM section writer ────────────────────────────────────────────────────────



# ── Placeholder detection (module-level so writer_node can also use it) ───────
# These phrases indicate the LLM produced a cop-out / refusal instead of real
# analysis. They must be matched as STANDALONE phrases, not substrings, to
# avoid false positives on valid text like "Google constrained its API pricing".
_PLACEHOLDER_PHRASES = (
    "no analysis could be generated",
    "limited data was retrieved",
    "could not be generated",
    "unable to generate",
    "insufficient data in retrieved sources",
    "no relevant information found",
    "not enough data to generate",
    "data was not available for this",
    "research was constrained by",          # only when used as an apology opener
    "constrained by available source",      # explicit apology form
    "constrained by the content of",        # explicit apology form
    "i don't have access to",
    "i do not have access to",
    "as an ai, i",
    "as an ai assistant",
    "i cannot provide",
    "i'm unable to",
    "i am unable to",
    "require validated competitive data",   # the exact fallback phrase we wrote
    "require further validation",
    "please re-run",
)


def _is_placeholder(text: str) -> bool:
    """Return True only if the text is clearly a refusal/fallback, not real analysis."""
    t = text.lower().strip()
    # Too short to be real analysis
    if len(t) < 80:
        return True
    # Check for exact cop-out phrases
    return any(p in t for p in _PLACEHOLDER_PHRASES)


def _write_section(
    section_name: str,
    topic: str,
    context: str,
    instruction: str,
    fallback: str = "",
) -> str:
    """
    Call the LLM to write a specific report section.

    Uses call_llm_with_retry() internally so the model is always rebuilt from
    the current _active_model — if the first model hits a quota error mid-call,
    it automatically switches to the next fallback and retries.

    If both attempts return placeholder-like text, returns *fallback*.
    """
    def _call(instruction_override: str) -> str:
        user_msg = (
            f"Topic: {topic}\n\n"
            f"Context:\n{context[:12000]}\n\n"
            f"Task: Write the '{section_name}' section. {instruction_override}\n"
            f"Write a thorough, complete section. Use multiple paragraphs and bullet lists as needed. Cover every relevant data point from the context. Be specific, analytical, and comprehensive — do not truncate or summarise prematurely."
        )
        response = call_llm_with_retry(
            messages=[
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ],
            temperature=0.2,
            max_retries=3,
            base_delay=5.0,
            label=section_name,
        )
        return response.content.strip()

    try:
        result = _call(instruction)

        # If the LLM returns a useless placeholder, try once more with stricter prompt
        if _is_placeholder(result):
            logger.info(
                "[writer] Section '%s' returned placeholder on first attempt — retrying with stricter prompt",
                section_name,
            )
            stricter = (
                f"{instruction} "
                "IMPORTANT: You MUST write substantive content. "
                "The context above contains real research findings — use them. "
                "Do NOT write 'no analysis could be generated' or similar. "
                "If data is limited, summarise what IS known and state confidence level."
            )
            result = _call(stricter)

        if _is_placeholder(result):
            logger.warning(
                "[writer] Section '%s' still returned placeholder after retry — using fallback",
                section_name,
            )
            return fallback or f"_{section_name}: LLM call failed. Re-run to regenerate._"

        return result

    except Exception as exc:
        logger.warning(
            "[writer] LLM call failed for section '%s': %s — using fallback.", section_name, exc
        )
        return fallback or f"_{section_name} could not be generated due to an error._"


# ── Writer node ───────────────────────────────────────────────────────────────


def writer_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Writer Agent.

    Generates all report sections using the LLM, assembles citations,
    and populates state.final_report.

    Resilient design:
    - Proceeds even if analysis or verification results are None/empty (uses defaults).
    - Includes unverified claims with [Unverified] marker instead of hiding them.
    - Falls back gracefully if any individual LLM section call fails.
    - Writes 6 LLM sections: executive_summary, competitor_pricing, product_updates,
      market_signals, business_risks, strategic_recommendations, AND opportunities.

    Args:
        state: Current BriefingState.

    Returns:
        Updated BriefingState.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "writer", AgentStatus.RUNNING)
    topic = state.get("topic", "")
    add_trace(state, AGENT_NAME, "writing", f"Writer agent started for topic: '{topic}'")

    research = state.get("research_result")
    analysis: Optional[AnalysisResult] = state.get("analysis_result")
    verification: Optional[VerificationResult] = state.get("verification_result")

    # Warn but continue – do NOT fail if upstream results are missing
    if not analysis:
        add_trace(state, AGENT_NAME, "warning", "No analysis result available; writer will use defaults.")
    if not verification:
        add_trace(state, AGENT_NAME, "warning", "No verification result available; writer will use defaults.")

    try:
        meta = state.get("run_metadata")

        # ── Build rich context for all LLM sections ────────────────────────────
        full_context = _build_context(research, analysis, verification)

        # ── Pre-load analyst syntheses for direct use ─────────────────────────
        _syntheses: dict = {}
        if analysis and analysis.section_syntheses:
            _syntheses = analysis.section_syntheses

        # ── Helper to write + track tool call ─────────────────────────────────
        def write(section_name: str, instruction: str, fallback: str = "",
                  synthesis_key: str = "") -> str:
            """
            Write one report section. Always runs the Writer LLM for a final polish
            pass — even when analyst synthesis is available. The synthesis text is
            injected into the context so the Writer blends it with raw claims.

            This guarantees every section goes through the Writer's style/quality
            standards rather than dumping raw synthesis text verbatim.
            """
            # Build a section-specific context: analyst synthesis first (if available),
            # then the full pipeline context as supplementary detail.
            synth_text = ""
            if synthesis_key and synthesis_key in _syntheses:
                synth_text = _syntheses[synthesis_key].strip()

            if synth_text and len(synth_text) >= 100 and not _is_placeholder(synth_text):
                # Prepend synthesis to the section context so the Writer uses it as
                # its primary input and enriches it with claims/profiles.
                section_context = (
                    f"=== ANALYST PRE-SYNTHESIS FOR THIS SECTION ===\n"
                    f"{synth_text}\n\n"
                    f"=== FULL PIPELINE CONTEXT (supplement synthesis with these) ===\n"
                    f"{full_context[:10000]}"
                )
            else:
                section_context = full_context

            text = _write_section(section_name, topic, section_context, instruction, fallback)
            if meta:
                meta.tool_calls += 1
            return text

        # ── Section 1: Executive Summary ───────────────────────────────────────
        executive_summary = write(
            "Executive Summary",
            "Write the Executive Summary. "
            "Start with the single most important competitive development from the sources and its strategic implication. "
            "Then summarize the 2-3 most significant competitor moves — name each entity, the action, and why it matters. "
            "End with the dominant market trend and the top priority action. "
            "Only include what is new, specific, and actionable. Cut anything generic.",
            fallback=(
                f"{len(research.sources) if research else 0} sources retrieved for '{topic}'. "
                "Summary generation failed — re-run to regenerate."
            ),
            synthesis_key="executive_summary",
        )

        # ── Section 2: Competitor Pricing ──────────────────────────────────────
        competitor_pricing = write(
            "Competitor Pricing Analysis",
            "Write the Competitor Pricing Analysis. "
            "If 2+ entities have pricing data, open with a compact comparison table (Entity | Key Tier | Price | Notes). "
            "Then write 2-3 sentences: who leads on price, who targets premium, and the most important pricing gap or opportunity. "
            "Only include pricing facts from the sources. Mark anything unconfirmed as 'Unconfirmed —'.",
            fallback=(
                "Pricing data was not present in the retrieved sources. "
                "Check entity websites directly for current pricing."
            ),
            synthesis_key="competitor_pricing",
        )

        # ── Section 3: Recent Developments ────────────────────────────────────
        product_updates = write(
            "Recent Developments & Capability Updates",
            "Write the Recent Developments section. "
            "List only the most significant product launches, service updates, or capability changes from the sources. "
            "For each: name it, when it happened (if known), and one sentence on why it matters competitively. "
            "Cut anything minor. Close with one sentence on who is moving fastest and what that means. "
            "Mark unconfirmed items as 'Unconfirmed —'.",
            fallback=(
                "No specific product or service updates were found in the retrieved sources."
            ),
            synthesis_key="product_updates",
        )

        # ── Section 4: Market Signals ──────────────────────────────────────────
        market_signals = write(
            "Market Signals & Trends",
            "Write the Market Signals & Trends section. "
            "Identify the strongest signals from the sources — only include ones with specific evidence. "
            "For each signal: state the observation and its strategic implication (who benefits, who is at risk). "
            "Cut any signal that is generic or unsupported by the sources. "
            "End with the single biggest macro force shaping this space. "
            "Mark unconfirmed signals as 'Unconfirmed —'.",
            fallback=(
                f"Market signals for '{topic}' could not be extracted. Re-run to regenerate."
            ),
            synthesis_key="market_signals",
        )

        # ── Section 5: Business Risks ──────────────────────────────────────────
        business_risks = write(
            "Business Risks",
            "Write the Business Risks section. "
            "List only the material risks that are supported by specific evidence from the sources, ranked by severity. "
            "For each: bold title, one sentence on the threat and potential impact, severity (High/Medium/Low), and one mitigation action. "
            "Cut speculative or generic risks not grounded in the sources. "
            "Mark unconfirmed risks as 'Unconfirmed —'.",
            fallback=(
                "Risk data was limited in the retrieved sources. Re-run for a full risk assessment."
            ),
            synthesis_key="business_risks",
        )

        # ── Section 6: Strategic Recommendations ──────────────────────────────
        strategic_recommendations = write(
            "Strategic Recommendations",
            "Write the Strategic Recommendations section. "
            "Give only recommendations that are directly supported by evidence from the sources. "
            "For each: state WHAT to do in one sentence, WHY (cite the specific finding), and the timeline (Immediate / Near-term / Strategic). "
            "Rank by urgency. Cut anything generic or not grounded in the data.",
            fallback=(
                f"Recommendations for '{topic}' could not be generated. Re-run to regenerate."
            ),
            synthesis_key="strategic_recommendations",
        )

        # ── Section 7: Opportunities ──────────────────────────────────────────
        opportunities = write(
            "Key Opportunities",
            "Write the Key Opportunities section. "
            "Identify only opportunities that are clearly supported by the sources — a gap, weakness, or unmet need with evidence. "
            "For each: name the opportunity, cite the evidence, and state one concrete action to capture it. "
            "Mark speculative opportunities as 'Unconfirmed —'. Cut anything not grounded in the data.",
            fallback=(
                f"Opportunity analysis for '{topic}' could not be completed. Re-run to regenerate."
            ),
            synthesis_key="opportunities",
        )

        # ── Generate citations ────────────────────────────────────────────────
        sources = research.sources if research else []
        citations = generate_citations(sources)

        # ── Assemble final report ─────────────────────────────────────────────
        from tools.audit_logger import format_audit_summary
        audit_entries = state.get("audit_log", [])
        audit_summary = format_audit_summary(audit_entries)

        if meta:
            meta.completed_at = datetime.utcnow().isoformat()
            try:
                elapsed_total = (
                    datetime.fromisoformat(meta.completed_at)
                    - datetime.fromisoformat(meta.started_at)
                ).total_seconds()
                meta.duration_seconds = round(elapsed_total, 1)
            except Exception:
                meta.duration_seconds = round(time.perf_counter() - start_ts, 1)
            meta.status = "completed"

        # citation_coverage / overall_confidence – safe defaults if verification missing
        citation_coverage = verification.citation_coverage if verification else 0.0
        overall_confidence = verification.overall_confidence if verification else 0.0

        final_report = assemble_final_report(
            topic=topic,
            executive_summary=executive_summary,
            competitor_pricing=competitor_pricing,
            product_updates=product_updates,
            market_signals=market_signals,
            business_risks=business_risks,
            strategic_recommendations=strategic_recommendations,
            opportunities=opportunities,
            analysis=analysis,
            verification=verification,
            citations=citations,
            run_metadata=meta,
            audit_summary=audit_summary,
        )

        state["final_report"] = final_report
        state["workflow_phase"] = WorkflowPhase.GOVERNANCE.value

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
        add_trace(
            state,
            AGENT_NAME,
            "completed",
            f"Report written. {final_report.word_count} words, "
            f"{len(citations)} citations. Coverage: {final_report.citation_coverage:.0%}",
            duration_ms=elapsed_ms,
            metadata={"word_count": final_report.word_count, "citations": len(citations)},
        )
        add_audit(
            state,
            event_type="report_written",
            agent=AGENT_NAME,
            description=(
                f"Report assembled: {final_report.word_count} words, {len(citations)} citations"
            ),
            data={"word_count": final_report.word_count, "citations": len(citations)},
        )
        set_agent_status(state, "writer", AgentStatus.COMPLETED)
        logger.info(
            "Writer complete: %d words, %d citations", final_report.word_count, len(citations)
        )

    except Exception as exc:
        msg = f"Writer agent failed: {exc}"
        add_error(state, msg)
        add_trace(state, AGENT_NAME, "failed", msg)
        set_agent_status(state, "writer", AgentStatus.FAILED)
        logger.error(msg, exc_info=True)

    return state
