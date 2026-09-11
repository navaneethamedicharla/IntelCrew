import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from dotenv import load_dotenv; load_dotenv()
import logging
logging.basicConfig(level=logging.WARNING)

from agents.state import BriefingState
from agents.research_agent import research_node
from agents.analyst_agent import analyst_node
from agents.writer_agent import writer_node

print("=== END-TO-END SMOKE TEST ===")
print("Topic: electric vehicle battery market 2025")
print()

state = BriefingState(
    topic="electric vehicle battery market 2025",
    max_sources=4,
    max_steps=20,
    run_id="smoke_final",
)

state = research_node(state)
research = state.get("research_result")
print(f"[RESEARCH] {len(research.sources)} sources retrieved via Tavily/web search")

state = analyst_node(state)
analysis = state.get("analysis_result")
synth = analysis.section_syntheses or {}
print(f"[ANALYST]  {len(analysis.extracted_claims)} claims, {len(analysis.competitor_profiles)} competitors, {len(synth)}/7 synthesis sections")

state = writer_node(state)
report = state.get("final_report")

SECTIONS = [
    "executive_summary", "competitor_pricing", "product_updates",
    "market_signals", "business_risks", "strategic_recommendations", "opportunities"
]

print(f"[WRITER]   FinalReport generated")
print()
all_ok = True
# Use precise multi-word phrases to avoid false positives on legitimate content.
# e.g. "supply-constrained" is real content, not a placeholder.
PLACEHOLDER_PHRASES = [
    "insufficient data",
    "not in my knowledge",
    "research was constrained",
    "constrained by available source",
    "constrained by the content",
    "enable tavily",
    "no relevant information",
    "as an ai",
    "i cannot provide",
    "data was not available",
    "unable to generate",
    "could not be generated",
    "no analysis could be generated",
]
for sec in SECTIONS:
    val = getattr(report, sec, None) or ""
    is_placeholder = any(p in val.lower() for p in PLACEHOLDER_PHRASES)
    status = "PLACEHOLDER" if is_placeholder else "OK"
    if is_placeholder:
        all_ok = False
        # Show which phrase triggered it
        bad = [p for p in PLACEHOLDER_PHRASES if p in val.lower()]
        print(f"  {status:12} [{sec}] {len(val)} chars  <-- triggered by: {bad}")
    else:
        print(f"  {status:12} [{sec}] {len(val)} chars")

print()
if all_ok:
    print("RESULT: ALL SECTIONS PASS - pipeline working end-to-end with real content")
else:
    print("RESULT: SOME SECTIONS HAVE PLACEHOLDER TEXT")

print()
print("--- Executive Summary preview (first 500 chars) ---")
es = getattr(report, "executive_summary", "") or ""
print(es[:500])
