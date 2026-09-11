"""
Analyst Agent – reads research sources and extracts structured competitive
intelligence using the LLM.

Key design decisions:
- Each source is processed individually so one bad article never blocks others.
- JSON parsing handles markdown fences, truncated JSON, and common LLM quirks.
- Falls back to sentence-level extraction from raw text if LLM returns empty.
- Detailed logging at every step: articles read, claims extracted, failures.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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
    CompetitorProfile,
    WorkflowPhase,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "analyst_agent"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior competitive intelligence analyst at a top-tier strategy consulting firm.

Given a single research article, extract EVERY piece of business intelligence it contains.

Extract the following categories — leave a category as an empty list [] if nothing is found, but look hard:
- competitor_names: List every company name mentioned as a competitor or market player.
- pricing: Specific pricing data (dollar amounts, tier names, % changes, "free trial", "enterprise pricing").
- product_launches: Named product releases, features, or platform updates with dates if mentioned.
- ai_features: Specific AI/ML capabilities named in the article (copilots, automation, generative AI features).
- partnerships: Named partner + what they do together.
- acquisitions: Acquired company + rationale if stated.
- market_trends: Observable market-level facts with evidence (growth %, market size, adoption stats).
- funding: Funding rounds, amounts, investors, valuations.
- customer_adoption: Customer wins, churn stats, NPS data, user counts, enterprise adoption.
- business_risks: Specific risks named in the article (competitive threats, regulatory, pricing pressure).
- strategic_opportunities: Gaps or openings described in the article.

Rules:
- Extract ONLY facts explicitly stated in the article. No fabrication.
- Be SPECIFIC: include exact names, numbers, dates, and product names.
- Each finding is a short factual sentence (max 120 words).
- Confidence: 0.9 = direct quote with numbers; 0.7 = clearly stated; 0.5 = implied.
- source_url must be the exact URL provided.

Return ONLY valid JSON. No prose before or after.

JSON schema:
{
  "competitor_names": ["string"],
  "findings": [
    {
      "category": "pricing|product|ai_feature|partnership|acquisition|market|funding|customer|risk|opportunity",
      "claim": "specific factual sentence",
      "evidence": "exact quote or close paraphrase from article",
      "confidence": 0.0,
      "source_url": "string"
    }
  ]
}"""


def _build_source_prompt(topic: str, source_title: str, source_url: str, source_text: str) -> str:
    """Build a per-source extraction prompt."""
    # Trim text to stay within token budget.
    # Use up to 4000 chars to capture meaningful content from each article.
    # This keeps the individual call well within Groq's context limits.
    text = source_text[:4000].strip()
    return f"""Topic: {topic}

Article Title: {source_title}
Article URL: {source_url}

Article Text:
{text}

Extract ALL competitive intelligence findings from this article. Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# JSON parsing — handles all common LLM output quirks
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from an LLM response.
    Handles: markdown fences (```json...```), leading/trailing prose,
    truncated JSON (attempts to close unclosed objects/arrays).
    """
    if not raw:
        return {}

    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        # Extract content between first ``` and last ```
        parts = text.split("```")
        # parts[1] is inside the first fence pair
        for part in parts[1:]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                text = candidate
                break

    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start != -1:
        text = text[start:]

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to repair truncated JSON by closing unclosed structures
    repaired = _repair_json(text)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    logger.debug("[analyst] JSON parse failed on: %s...", raw[:100])
    return {}


def _repair_json(text: str) -> str:
    """
    Attempt to close unclosed JSON by counting brackets.
    Only fixes truncation — does not fix malformed keys/values.
    """
    # Find the last complete object boundary
    # Truncate at last complete entry (after last }, or last "])
    last_good = max(text.rfind("},"), text.rfind("}]"), text.rfind('"}'))
    if last_good > 0:
        text = text[: last_good + 1]

    # Count and close unclosed brackets
    opens = text.count("{") - text.count("}")
    arr_opens = text.count("[") - text.count("]")
    suffix = ""
    for _ in range(max(0, arr_opens)):
        suffix += "]"
    for _ in range(max(0, opens)):
        suffix += "}"
    return text + suffix


# ---------------------------------------------------------------------------
# Per-source extraction
# ---------------------------------------------------------------------------

def _extract_from_source(
    topic: str,
    source_title: str,
    source_url: str,
    source_text: str,
    source_id: str,
) -> Tuple[List[Claim], List[str]]:
    """
    Extract competitive intelligence from a single source using the LLM.

    Uses call_llm_with_retry() so the model is rebuilt after any quota switch.
    Returns (claims, competitor_names).
    Errors are caught and logged — never propagated.
    """
    claims: List[Claim] = []
    competitor_names: List[str] = []

    if not source_text or len(source_text.strip()) < 100:
        logger.info(
            "[analyst] Skipping %s — insufficient text (%d chars)",
            source_url[:60], len(source_text or ""),
        )
        return claims, competitor_names

    try:
        prompt = _build_source_prompt(topic, source_title, source_url, source_text)
        response = call_llm_with_retry(
            messages=[
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            max_retries=3,
            base_delay=5.0,
            label=f"extract:{source_title[:40]}",
        )
        raw = response.content or ""
        parsed = _parse_llm_json(raw)

        if not parsed:
            logger.info(
                "[analyst] LLM returned empty/unparseable JSON for %s — raw: %s...",
                source_url[:60], raw[:80],
            )
            return claims, competitor_names

        competitor_names = [
            str(n) for n in parsed.get("competitor_names", []) if n
        ]

        for f in parsed.get("findings", []):
            claim_text = (f.get("claim") or "").strip()
            if not claim_text or len(claim_text) < 15:
                continue
            category = (f.get("category") or "market").lower()
            # Normalise category
            category_map = {
                "ai_feature": "technology",
                "ai_features": "technology",
                "funding": "market",
                "customer": "customer",
                "opportunity": "market",
            }
            category = category_map.get(category, category)
            if category not in (
                "pricing", "product", "partnership", "acquisition",
                "market", "technology", "customer", "risk",
            ):
                category = "market"

            claims.append(
                Claim(
                    text=claim_text[:300],
                    category=category,
                    source_ids=[source_id],
                    status=ClaimStatus.UNVERIFIED,
                    confidence=min(1.0, max(0.0, float(f.get("confidence", 0.6)))),
                    supporting_evidence=(f.get("evidence") or "")[:300] or None,
                )
            )

        logger.info(
            "[analyst] Source '%s' → %d claims, %d competitors",
            source_title[:50], len(claims), len(competitor_names),
        )

    except Exception as exc:
        logger.warning(
            "[analyst] LLM extraction failed for %s: %s",
            source_url[:60], exc,
        )

    return claims, competitor_names


# ---------------------------------------------------------------------------
# Fallback: sentence-level extraction from raw text
# ---------------------------------------------------------------------------

def _fallback_extract(sources_data: List[Dict]) -> Tuple[List[Claim], List[str]]:
    """
    When every LLM call fails, extract sentences from raw source text
    as UNVERIFIED claims so the Writer still has content.
    """
    claims: List[Claim] = []
    competitors: List[str] = []

    for item in sources_data:
        text = item.get("text", "")
        if not text:
            continue
        src_id = item.get("id", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
        for sent in sentences[:3]:
            claims.append(
                Claim(
                    text=sent[:250],
                    category="market",
                    source_ids=[src_id] if src_id else [],
                    status=ClaimStatus.UNVERIFIED,
                    confidence=0.35,
                )
            )
        if len(claims) >= 20:
            break

    logger.info(
        "[analyst] Fallback sentence extraction: %d claims from raw text", len(claims)
    )
    return claims, competitors


# ---------------------------------------------------------------------------
# Second-pass synthesis: structured section narratives for the Writer
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """You are a senior competitive intelligence analyst at a top-tier strategy consulting firm. Synthesize the research findings below into a structured intelligence briefing.

ABSOLUTE RULES — violating any of these means the output is rejected:
1. You MUST populate ALL 7 JSON keys — never omit a key or return an empty string.
2. Every section MUST be 150-200 words of dense, specific analytical text.
3. FORBIDDEN phrases (output will be rejected if any appear):
   - "insufficient data", "no data", "not available", "no information", "no relevant"
   - "I cannot", "I don't have", "as an AI", "I was unable", "beyond my"
   - "constrained by" (as a limitation disclaimer)
   - "require validated", "require more data", "please re-run"
4. The SOURCE SNIPPETS below contain REAL web data fetched just now — use them as your primary raw material.
5. If a section lacks specific numbers/dates, synthesize patterns from the snippets and mark as "Inferred from sources."
6. Every sentence must deliver a finding, comparison, or insight. No filler.
7. Name specific competitors, products, dollar amounts, percentages, and dates wherever the sources provide them.

For each section:
- Lead with the single most important finding
- Compare competitors directly (e.g. "ChatGPT Plus at $20/mo vs Gemini Advanced at $19.99/mo")
- Explain strategic implications — WHY each finding matters competitively
- Close with the key takeaway or recommended watch-point

Return ONLY valid JSON with exactly these 7 keys, nothing else before or after:
{
  "executive_summary": "150-200 words of substantive analysis",
  "competitor_pricing": "150-200 words of substantive analysis",
  "product_updates": "150-200 words of substantive analysis",
  "market_signals": "150-200 words of substantive analysis",
  "business_risks": "150-200 words of substantive analysis",
  "strategic_recommendations": "150-200 words of substantive analysis",
  "opportunities": "150-200 words of substantive analysis"
}"""


def _build_synthesis_prompt(
    topic: str,
    claims: List[Claim],
    profiles: List[CompetitorProfile],
    market_signals: List[str],
    technology_trends: List[str],
    customer_trends: List[str],
    market_movements: List[str],
    sources_data: List[Dict],
) -> str:
    """Assemble the user message for the synthesis LLM call."""

    # Claims grouped by category
    def _fmt_claims(cat: str) -> str:
        items = [c for c in claims if c.category == cat]
        if not items:
            return "  None."
        return "\n".join(
            f"  - {c.text} (confidence: {c.confidence:.0%})"
            for c in items[:10]
        )

    # Competitor profiles summary
    comp_lines: List[str] = []
    for p in profiles[:8]:
        parts = []
        if p.pricing_changes:
            parts.append("Pricing: " + "; ".join(p.pricing_changes[:3]))
        if p.product_launches:
            parts.append("Products: " + "; ".join(p.product_launches[:3]))
        if p.competitive_advantages:
            parts.append("AI/Tech: " + "; ".join(p.competitive_advantages[:3]))
        if p.partnerships:
            parts.append("Partnerships: " + "; ".join(p.partnerships[:2]))
        if p.acquisitions:
            parts.append("Acquisitions: " + "; ".join(p.acquisitions[:2]))
        if p.business_risks:
            parts.append("Risks: " + "; ".join(p.business_risks[:2]))
        detail = " | ".join(parts) if parts else "Limited data."
        comp_lines.append(f"  - {p.name}: {detail}")
    comp_text = "\n".join(comp_lines) if comp_lines else "  None identified."

    # Source snippets — 800 chars per source gives the LLM real content to work from
    src_lines: List[str] = []
    for s in sources_data[:8]:
        snippet = (s.get("text") or "")[:800].strip()
        if snippet:
            src_lines.append(f"  [{s.get('title', '')[:70]}] ({s.get('url','')[:80]}):\n  {snippet}")
    src_text = "\n\n".join(src_lines) if src_lines else "  None."

    def _fmt_list(items: List[str], limit: int = 8) -> str:
        return "\n".join(f"  - {x}" for x in items[:limit]) or "  None."

    return f"""Topic: {topic}

=== SOURCE SNIPPETS FROM WEB SEARCH (PRIMARY DATA — use these as your main source) ===
{src_text}

=== PRICING CLAIMS ===
{_fmt_claims("pricing")}

=== PRODUCT / AI FEATURE CLAIMS ===
{_fmt_claims("product")}
{_fmt_claims("technology")}

=== MARKET / CUSTOMER CLAIMS ===
{_fmt_claims("market")}
{_fmt_claims("customer")}

=== RISK CLAIMS ===
{_fmt_claims("risk")}

=== PARTNERSHIP / ACQUISITION CLAIMS ===
{_fmt_claims("partnership")}
{_fmt_claims("acquisition")}

=== COMPETITOR PROFILES ===
{comp_text}

=== MARKET SIGNALS ===
{_fmt_list(market_signals)}

=== TECHNOLOGY TRENDS ===
{_fmt_list(technology_trends)}

=== CUSTOMER TRENDS ===
{_fmt_list(customer_trends)}

=== MARKET MOVEMENTS (M&A / Funding) ===
{_fmt_list(market_movements)}

Synthesize all of the above into the 7 JSON section summaries. IMPORTANT: Always draw from the SOURCE SNIPPETS above — they contain real, current web data. Be specific, analytical, and use real names/numbers/dates where available."""


def _synthesize_sections(
    topic: str,
    claims: List[Claim],
    profiles: List[CompetitorProfile],
    market_signals: List[str],
    technology_trends: List[str],
    customer_trends: List[str],
    market_movements: List[str],
    sources_data: List[Dict],
) -> Dict[str, str]:
    """
    Second-pass synthesis: one LLM call that produces a structured dict of
    per-section narratives (~150-200 words each).

    Uses call_llm_with_retry() so the model is rebuilt after any quota switch.
    Returns an empty dict if the call fails.
    """
    empty: Dict[str, str] = {}

    # Run synthesis as long as we have source snippets OR claims/signals
    if not claims and not market_signals and not profiles and not sources_data:
        logger.warning("[analyst] Skipping synthesis — no data at all (no claims, signals, profiles, or sources).")
        return empty

    try:
        prompt = _build_synthesis_prompt(
            topic=topic,
            claims=claims,
            profiles=profiles,
            market_signals=market_signals,
            technology_trends=technology_trends,
            customer_trends=customer_trends,
            market_movements=market_movements,
            sources_data=sources_data,
        )
        response = call_llm_with_retry(
            messages=[
                SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            max_retries=3,
            base_delay=5.0,
            label="synthesis",
        )
        raw = (response.content or "").strip()
        parsed = _parse_llm_json(raw)

        if not parsed:
            logger.warning("[analyst] Synthesis LLM returned empty/unparseable JSON — raw: %s...", raw[:200])
            return empty

        # Validate keys; only keep expected section names with real content
        expected_keys = {
            "executive_summary", "competitor_pricing", "product_updates",
            "market_signals", "business_risks", "strategic_recommendations",
            "opportunities",
        }
        # Only drop a section if it is truly empty or absurdly short (< 30 chars)
        # Do NOT drop based on specific phrases — the prompt already bans them
        result = {}
        for k, v in parsed.items():
            if k not in expected_keys:
                continue
            text = str(v).strip()
            if text and len(text) >= 30:
                result[k] = text
            else:
                logger.info("[analyst] Synthesis section '%s' too short (%d chars) — dropping", k, len(text))

        logger.info(
            "[analyst] Synthesis complete — %d/%d sections populated",
            len(result), len(expected_keys),
        )
        return result

    except Exception as exc:
        logger.warning("[analyst] Synthesis LLM call failed: %s — Writer will use raw claims.", exc)
        return empty


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def analyst_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Analyst Agent.

    Processes each research source individually with error isolation.
    Aggregates all findings into a single AnalysisResult.
    Never fails the workflow — returns whatever was extracted.

    Args:
        state: Current BriefingState.
    Returns:
        Updated BriefingState with analysis_result populated.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "analyst", AgentStatus.RUNNING)
    topic = state.get("topic", "")
    add_trace(state, AGENT_NAME, "analyzing", f"Analyst started for: '{topic}'")

    # --- Collect source data -------------------------------------------------
    research = state.get("research_result")
    sources_data: List[Dict] = []
    if research:
        for src in research.sources:
            text = src.raw_content or src.summary or ""
            if text and len(text.strip()) >= 80:
                sources_data.append({
                    "id": src.id,
                    "title": src.title or src.url,
                    "url": src.url,
                    "text": text,
                })

    logger.info(
        "[analyst] === Starting extraction | topic='%s' | %d sources with text ===",
        topic, len(sources_data),
    )

    if not sources_data:
        msg = "No source text available for analysis — all sources lack content."
        add_error(state, msg)
        add_trace(state, AGENT_NAME, "failed", msg)
        set_agent_status(state, "analyst", AgentStatus.FAILED)
        state["analysis_result"] = AnalysisResult()
        logger.warning("[analyst] %s", msg)
        return state

    # --- Per-source LLM extraction -------------------------------------------
    all_claims: List[Claim] = []
    all_competitor_names: List[str] = []
    llm_success_count = 0
    llm_fail_count = 0

    try:
        meta = state.get("run_metadata")

        for item in sources_data:
            claims, comp_names = _extract_from_source(
                topic=topic,
                source_title=item["title"],
                source_url=item["url"],
                source_text=item["text"],
                source_id=item["id"],
            )
            if meta:
                meta.tool_calls += 1

            if claims:
                llm_success_count += 1
                all_claims.extend(claims)
                all_competitor_names.extend(comp_names)
            else:
                llm_fail_count += 1
                logger.info(
                    "[analyst] No claims extracted from '%s' (url: %s)",
                    item["title"][:50], item["url"][:60],
                )

        logger.info(
            "[analyst] LLM extraction done: %d sources succeeded, %d returned 0 claims, "
            "%d total claims, %d unique competitors",
            llm_success_count, llm_fail_count,
            len(all_claims), len(set(all_competitor_names)),
        )

    except Exception as exc:
        msg = f"Analyst LLM setup failed: {exc}"
        add_error(state, msg)
        logger.error("[analyst] %s", msg, exc_info=True)

    # --- Fallback if LLM produced nothing ------------------------------------
    if not all_claims:
        logger.warning(
            "[analyst] LLM produced 0 claims — running sentence-level fallback"
        )
        all_claims, all_competitor_names = _fallback_extract(sources_data)

    # --- Deduplicate claims by text fingerprint ------------------------------
    seen_fps: set = set()
    unique_claims: List[Claim] = []
    for c in all_claims:
        fp = c.text[:60].lower().strip()
        if fp not in seen_fps:
            seen_fps.add(fp)
            unique_claims.append(c)
    if len(unique_claims) < len(all_claims):
        logger.info(
            "[analyst] Deduplication: %d → %d claims",
            len(all_claims), len(unique_claims),
        )
    all_claims = unique_claims

    # --- Build competitor profiles from aggregated data ----------------------
    profiles: List[CompetitorProfile] = []
    seen_competitors: set = set()
    for name in all_competitor_names:
        if not name or name.lower() in seen_competitors:
            continue
        seen_competitors.add(name.lower())
        # Collect claims for this competitor
        comp_pricing = [c.text for c in all_claims if c.category == "pricing" and name.lower() in c.text.lower()]
        comp_products = [c.text for c in all_claims if c.category == "product" and name.lower() in c.text.lower()]
        comp_risks = [c.text for c in all_claims if c.category == "risk" and name.lower() in c.text.lower()]
        comp_partnerships = [c.text for c in all_claims if c.category == "partnership" and name.lower() in c.text.lower()]
        comp_acquisitions = [c.text for c in all_claims if c.category == "acquisition" and name.lower() in c.text.lower()]
        comp_advantages = [c.text for c in all_claims if c.category == "technology" and name.lower() in c.text.lower()]
        profiles.append(
            CompetitorProfile(
                name=name,
                pricing_changes=comp_pricing[:4],
                product_launches=comp_products[:4],
                partnerships=comp_partnerships[:3],
                acquisitions=comp_acquisitions[:3],
                competitive_advantages=comp_advantages[:3],
                business_risks=comp_risks[:3],
            )
        )

    # --- Categorise signals / trends from claims -----------------------------
    market_signals = list(dict.fromkeys(
        c.text for c in all_claims if c.category == "market"
    ))[:12]
    technology_trends = list(dict.fromkeys(
        c.text for c in all_claims if c.category == "technology"
    ))[:8]
    customer_trends = list(dict.fromkeys(
        c.text for c in all_claims if c.category == "customer"
    ))[:8]
    market_movements = list(dict.fromkeys(
        c.text for c in all_claims
        if c.category in ("acquisition", "funding")
    ))[:8]

    # --- Second-pass LLM synthesis: produce structured section narratives ----
    section_syntheses: Dict[str, str] = {}
    logger.info("[analyst] Running second-pass synthesis for %d claims across %d sources...",
                len(all_claims), len(sources_data))
    section_syntheses = _synthesize_sections(
        topic=topic,
        claims=all_claims,
        profiles=profiles,
        market_signals=market_signals,
        technology_trends=technology_trends,
        customer_trends=customer_trends,
        market_movements=market_movements,
        sources_data=sources_data,
    )
    meta = state.get("run_metadata")
    if meta:
        meta.tool_calls += 1

    result = AnalysisResult(
        competitor_profiles=profiles,
        extracted_claims=all_claims,
        market_signals=market_signals,
        technology_trends=technology_trends,
        customer_trends=customer_trends,
        market_movements=market_movements,
        section_syntheses=section_syntheses,
    )
    state["analysis_result"] = result

    meta = state.get("run_metadata")
    if meta:
        meta.total_claims = len(all_claims)

    state["workflow_phase"] = WorkflowPhase.VERIFYING.value
    elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)

    status_str = AgentStatus.COMPLETED if all_claims else AgentStatus.FAILED
    set_agent_status(state, "analyst", status_str)

    add_trace(
        state, AGENT_NAME, "completed",
        f"Extracted {len(all_claims)} claims from {llm_success_count}/{len(sources_data)} sources. "
        f"{len(profiles)} competitor profiles.",
        duration_ms=elapsed_ms,
        metadata={
            "total_claims": len(all_claims),
            "sources_processed": len(sources_data),
            "llm_success": llm_success_count,
            "llm_fail": llm_fail_count,
            "competitors": len(profiles),
        },
    )
    add_audit(
        state,
        event_type="analysis_complete",
        agent=AGENT_NAME,
        description=f"Extracted {len(all_claims)} claims and {len(profiles)} competitor profiles",
        data={"claims": len(all_claims), "competitors": len(profiles)},
    )
    logger.info(
        "[analyst] === Done | %d claims | %d competitors | %d market signals | %.1f ms ===",
        len(all_claims), len(profiles), len(market_signals), elapsed_ms,
    )
    return state
