"""
Research Agent – searches the web (and optionally the RAG knowledge base)
to gather sources on the given topic. Fills state.research_result.

Key behaviours
--------------
* Runs multiple search queries in parallel (ThreadPoolExecutor).
* Fetches article text concurrently (max 5 workers) via concurrent_fetch.
* Deduplicates URLs before any network call.
* NEVER returns 0 sources – a Wikipedia fallback is injected when all live
  searches fail, so downstream agents always have something to work with.
* Logs every significant step (search started, provider used, URLs found,
  duplicates removed, articles fetched, failures).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from agents.base_agent import (
    add_audit,
    add_error,
    add_trace,
    check_runaway,
    increment_step,
    set_agent_status,
)
from agents.state import (
    AgentStatus,
    BriefingState,
    ResearchResult,
    SourceDocument,
    WorkflowPhase,
)
from config import TAVILY_API_KEY, search_config
from tools.article_fetch import concurrent_fetch, fetch_article
from tools.source_extractor import extract_sources
from tools.web_search import build_search_queries, web_search

logger = logging.getLogger(__name__)
AGENT_NAME = "research_agent"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_single_query(query: str, max_sources: int) -> tuple[list, str]:
    """
    Execute one search query and return (results, query).
    Safe to call inside a thread – exceptions are caught.
    """
    try:
        logger.info("[research] Search started – query='%s'", query[:80])
        results = web_search(
            query,
            max_results=max_sources,
            tavily_api_key=TAVILY_API_KEY,
        )
        provider = "tavily" if TAVILY_API_KEY else "duckduckgo"
        logger.info(
            "[research] Provider=%s | query='%s' → %d results",
            provider,
            query[:80],
            len(results),
        )
        return results, query
    except Exception as exc:
        logger.warning("[research] Query failed '%s': %s", query[:80], exc)
        return [], query


def _search_web(
    topic: str, max_sources: int
) -> tuple[List[SourceDocument], List[str], List[str]]:
    """
    Run multiple search queries in parallel and collect deduplicated sources.

    Returns
    -------
    sources        : validated SourceDocument list
    used_queries   : queries that returned at least 1 result
    failed_queries : queries recorded as failed/empty
    """
    queries = build_search_queries(topic)
    # Cap the number of queries sensibly
    query_limit = max(3, max_sources // 3)
    queries = queries[:query_limit]

    logger.info(
        "[research] Launching %d search queries in parallel for topic='%s'",
        len(queries),
        topic,
    )

    all_raw: list = []
    used_queries: List[str] = []
    failed_queries: List[str] = []

    # Run all queries concurrently
    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
        futures = {pool.submit(_run_single_query, q, max_sources): q for q in queries}
        for future in as_completed(futures):
            results, query = future.result()
            if results:
                all_raw.extend(results)
                used_queries.append(query)
            else:
                failed_queries.append(f"search:{query}")

    logger.info(
        "[research] Parallel search complete – %d raw results across %d successful queries",
        len(all_raw),
        len(used_queries),
    )

    # ── Fallback: if every search returned nothing, inject a synthetic source ─
    if not all_raw:
        logger.warning(
            "[research] All search queries returned 0 results for topic='%s'. "
            "Injecting Wikipedia fallback source.",
            topic,
        )
        all_raw.append(
            {
                "title": f"General knowledge: {topic}",
                "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                "snippet": (
                    f"No live search results available for {topic}. "
                    "Analysis based on LLM knowledge."
                ),
                "domain": "wikipedia.org",
                "source": "fallback",
            }
        )

    # Deduplicate raw results by URL before building SourceDocuments
    seen_raw_urls: set[str] = set()
    unique_raw: list = []
    for item in all_raw:
        u = item.get("url", "").strip()
        if u and u not in seen_raw_urls:
            seen_raw_urls.add(u)
            unique_raw.append(item)
        elif u in seen_raw_urls:
            logger.debug("[research] Duplicate URL removed before extraction: %s", u)

    duplicates_removed = len(all_raw) - len(unique_raw)
    if duplicates_removed:
        logger.info(
            "[research] Removed %d duplicate URLs before source extraction (%d → %d)",
            duplicates_removed,
            len(all_raw),
            len(unique_raw),
        )

    sources = extract_sources(unique_raw, topic, max_sources=max_sources)
    logger.info(
        "[research] extract_sources produced %d SourceDocument objects",
        len(sources),
    )
    return sources, used_queries, failed_queries


def _enrich_sources(sources: List[SourceDocument]) -> List[SourceDocument]:
    """
    Fetch full article text for the top sources concurrently.
    Stores publisher, publication_date, and clean text on each SourceDocument.
    Sources that fail the quality gate (<150 chars) keep their search snippet only.
    """
    fetch_limit = min(8, len(sources))   # fetch up to 8 sources
    if fetch_limit == 0:
        return sources

    urls_to_fetch = [s.url for s in sources[:fetch_limit]]
    logger.info(
        "[research] Fetching full text for %d/%d sources (max_workers=5)",
        len(urls_to_fetch), len(sources),
    )

    fetch_results = concurrent_fetch(urls_to_fetch, timeout=20, max_chars=6000, max_workers=5)
    fetch_map = {r["url"]: r for r in fetch_results if r.get("url")}

    enriched: List[SourceDocument] = []
    clean_count = 0
    skip_count  = 0

    for i, src in enumerate(sources):
        if i < fetch_limit:
            result = fetch_map.get(src.url)
            if result:
                if result.get("error"):
                    skip_count += 1
                    logger.info(
                        "[research] Skip %s — %s",
                        src.url[:60], result["error"],
                    )
                    # Keep snippet as raw_content fallback so analyst always has text
                    if not src.raw_content and src.summary:
                        src.raw_content = src.summary
                elif result.get("text"):
                    src.raw_content = result["text"]
                    if result.get("title") and not src.title:
                        src.title = result["title"]
                    # Store publisher and publication_date in summary prefix
                    publisher   = result.get("publisher") or src.domain or ""
                    pub_date    = result.get("publication_date") or src.publication_date or ""
                    meta_prefix = ""
                    if publisher:
                        meta_prefix += f"Publisher: {publisher}. "
                    if pub_date:
                        meta_prefix += f"Published: {pub_date}. "
                    # Summary = first 400 chars of clean text
                    src.summary = (meta_prefix + src.raw_content[:400]).strip()
                    # Store on SourceDocument fields if available
                    if pub_date and not src.publication_date:
                        src.publication_date = pub_date
                    clean_count += 1
                    logger.debug(
                        "[research] Enriched %s — %d chars clean text",
                        src.url[:60], len(src.raw_content),
                    )
                else:
                    skip_count += 1
                    # Keep snippet as raw_content fallback
                    if not src.raw_content and src.summary:
                        src.raw_content = src.summary
            else:
                # No fetch result at all — use snippet
                if not src.raw_content and src.summary:
                    src.raw_content = src.summary
        else:
            # Not fetched (beyond fetch_limit) — use snippet as content
            if not src.raw_content and src.summary:
                src.raw_content = src.summary
        enriched.append(src)

    logger.info(
        "[research] Article enrichment complete — %d clean / %d skipped / %d not fetched",
        clean_count, skip_count, len(sources) - fetch_limit,
    )
    return enriched


def _search_rag(topic: str, state: BriefingState) -> List[SourceDocument]:
    """Query the RAG knowledge base if available."""
    rag_sources: List[SourceDocument] = []
    if not state.get("rag_enabled", False):
        return rag_sources
    try:
        from rag.knowledge_base import query_knowledge_base

        chunks = query_knowledge_base(topic, top_k=5)
        for i, chunk in enumerate(chunks):
            rag_sources.append(
                SourceDocument(
                    title=f"Knowledge Base Chunk {i+1}",
                    url="rag://knowledge-base",
                    summary=chunk.get("text", "")[:500],
                    domain="knowledge-base",
                    relevance_score=chunk.get("score", 0.5),
                    raw_content=chunk.get("text"),
                    is_trusted_domain=True,
                )
            )
        logger.info("[research] RAG returned %d chunks", len(rag_sources))
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[research] RAG query failed: %s", exc)
    return rag_sources


# ---------------------------------------------------------------------------
# LangGraph node entry-point
# ---------------------------------------------------------------------------


def research_node(state: BriefingState) -> BriefingState:
    """
    LangGraph node: Research Agent.

    Searches web sources and optionally the RAG knowledge base.
    Updates state with research_result.

    Guarantees
    ----------
    * Never routes to END due to empty sources – at minimum the Wikipedia
      fallback source is present.
    * Logs every significant step for full observability.

    Args:
        state: Current BriefingState.

    Returns:
        Updated BriefingState.
    """
    start_ts = time.perf_counter()
    increment_step(state)

    if check_runaway(state):
        return state

    set_agent_status(state, "research", AgentStatus.RUNNING)
    topic = state.get("topic", "")
    max_sources = state.get("max_sources", 10)

    logger.info(
        "[research] === Research node started | topic='%s' max_sources=%d ===",
        topic,
        max_sources,
    )
    add_trace(state, AGENT_NAME, "researching", f"Beginning research for topic: '{topic}'")

    try:
        # ── Web search (parallel queries) ─────────────────────────────────────
        sources, used_queries, failed_urls = _search_web(topic, max_sources)

        logger.info(
            "[research] Web search phase done – %d sources, %d queries used, %d failed",
            len(sources),
            len(used_queries),
            len(failed_urls),
        )

        # ── Deduplicate SourceDocuments by URL (safety net) ───────────────────
        seen_doc_urls: set[str] = set()
        deduped_sources: List[SourceDocument] = []
        for src in sources:
            if src.url not in seen_doc_urls:
                seen_doc_urls.add(src.url)
                deduped_sources.append(src)
        if len(deduped_sources) < len(sources):
            logger.info(
                "[research] SourceDocument deduplication: %d → %d",
                len(sources),
                len(deduped_sources),
            )
        sources = deduped_sources

        # ── Enrich top sources with full article text ─────────────────────────
        sources = _enrich_sources(sources)

        # ── RAG knowledge base ────────────────────────────────────────────────
        rag_sources = _search_rag(topic, state)
        all_sources = sources + rag_sources

        # ── Final safety net: guarantee at least 1 source ─────────────────────
        if not all_sources:
            logger.warning(
                "[research] Still 0 sources after all steps – injecting final fallback for topic='%s'",
                topic,
            )
            all_sources.append(
                SourceDocument(
                    title=f"General knowledge: {topic}",
                    url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                    summary=(
                        f"No live search results available for {topic}. "
                        "Analysis based on LLM knowledge."
                    ),
                    domain="wikipedia.org",
                    relevance_score=0.3,
                )
            )

        # Update visited URLs
        visited = [s.url for s in all_sources if s.url != "rag://knowledge-base"]

        result = ResearchResult(
            sources=all_sources,
            visited_urls=visited,
            failed_urls=failed_urls,
            search_queries_used=used_queries,
            rag_chunks_used=len(rag_sources),
            total_sources=len(all_sources),
        )
        state["research_result"] = result

        # Update run metadata
        meta = state.get("run_metadata")
        if meta:
            meta.total_sources = result.total_sources
            meta.search_queries = len(used_queries)
            meta.tool_calls += len(used_queries)

        state["workflow_phase"] = WorkflowPhase.ANALYZING.value
        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)

        logger.info(
            "[research] === Research complete | %d sources (%d RAG) | %d failures | %.1f ms ===",
            len(all_sources),
            len(rag_sources),
            len(failed_urls),
            elapsed_ms,
        )

        add_trace(
            state,
            AGENT_NAME,
            "completed",
            f"Research complete. Found {len(all_sources)} sources "
            f"({len(rag_sources)} from RAG). {len(failed_urls)} failures.",
            duration_ms=elapsed_ms,
            metadata={
                "total_sources": len(all_sources),
                "failed_urls": len(failed_urls),
                "rag_chunks": len(rag_sources),
            },
        )
        add_audit(
            state,
            event_type="research_complete",
            agent=AGENT_NAME,
            description=f"Collected {len(all_sources)} sources for '{topic}'",
            data={"sources": len(all_sources), "queries": len(used_queries)},
        )
        set_agent_status(state, "research", AgentStatus.COMPLETED)

    except Exception as exc:
        msg = f"Research agent failed: {exc}"
        add_error(state, msg)
        add_trace(state, AGENT_NAME, "failed", msg)
        set_agent_status(state, "research", AgentStatus.FAILED)
        logger.error("[research] %s", msg, exc_info=True)

        # Provide a minimal fallback result so the pipeline can still proceed
        fallback_source = SourceDocument(
            title=f"General knowledge: {topic}",
            url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
            summary=(
                f"Research agent encountered an error for topic '{topic}'. "
                "Analysis based on LLM general knowledge."
            ),
            domain="wikipedia.org",
            relevance_score=0.3,
        )
        state["research_result"] = ResearchResult(
            sources=[fallback_source],
            visited_urls=[fallback_source.url],
            failed_urls=[f"agent_error:{exc}"],
            search_queries_used=[],
            rag_chunks_used=0,
            total_sources=1,
        )
        logger.info(
            "[research] Fallback ResearchResult injected with 1 Wikipedia source."
        )

    return state
