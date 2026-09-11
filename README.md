# Competitive Intelligence Briefing Crew

A production-ready multi-agent AI system that researches any competitive topic, extracts structured intelligence, verifies claims, and generates a professional PDF/Markdown report — with a human approval gate before publishing.

---

## Architecture

```
User Input (Streamlit)
        │
        ▼
┌─────────────┐
│  Supervisor │  – validates input, initialises run
└──────┬──────┘
       ▼
┌─────────────┐
│  Research   │  – multi-provider web search (Tavily → DDG text → DDG news → stub)
│   Agent     │    parallel article fetching, dedup, fallback sources
└──────┬──────┘
       ▼
┌─────────────┐
│   Analyst   │  – LLM extracts competitor profiles, claims, market signals
│   Agent     │
└──────┬──────┘
       ▼
┌─────────────┐
│    Fact     │  – LLM verifies each claim against sources
│ Verification│    lenient: verified / unverified / rejected (only contradictions)
└──────┬──────┘
       ▼
┌─────────────┐
│   Writer    │  – generates 7 report sections via LLM
│   Agent     │    marks unverified claims inline
└──────┬──────┘
       ▼
┌─────────────┐
│ Governance  │  – citation coverage, confidence, hallucination checks
│   Agent     │    only refuses on actual hallucinations or empty sections
└──────┬──────┘
       ▼
┌─────────────┐
│   Human     │  – Approve / Reject gate in Streamlit UI
│  Approval   │
└──────┬──────┘
       ▼
 Final Report (Markdown + PDF download)
```

**Stack:** LangGraph · LangChain · OpenRouter/Groq/OpenAI · DuckDuckGo/Tavily · BeautifulSoup · Streamlit · FAISS (RAG)

---

## Setup

### Prerequisites

- Python 3.11 or 3.12 (3.13 works too)
- pip

### Installation

```bash
# 1. Clone / navigate to project
cd C:\projects\capstone\competitive-intelligence-crew

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
copy .env.example .env
# Edit .env and fill in your API keys (see below)

# 6. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | One of these three | OpenRouter API key (free models available) |
| `OPENAI_API_KEY` | One of these three | OpenAI API key |
| `GROQ_API_KEY` | One of these three | Groq API key (fast, free tier) |
| `LLM_MODEL` | Yes | Model ID matching your provider (see below) |
| `TAVILY_API_KEY` | Yes | Tavily search (more reliable than DDG) |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `RAG_ENABLED` | Optional | `true`/`false` — enable FAISS knowledge base |

### Model IDs by provider

| Provider | Example model |
|---|---|
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` or `openai/gpt-4o-mini` |
| Groq | `llama-3.3-70b-versatile` |
| OpenAI | `gpt-4o-mini` |

**Note:** If both `OPENROUTER_API_KEY` and `GROQ_API_KEY` are set, OpenRouter takes priority. Remove one to use the other.

### Example .env for OpenRouter (free)

```env
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
APP_ENV=development
LOG_LEVEL=INFO
RAG_ENABLED=false
```

### Example .env for Groq (free)

```env
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
APP_ENV=development
LOG_LEVEL=INFO
RAG_ENABLED=false
```

---

## How to Use

1. Enter a research topic in the sidebar (e.g. "Salesforce vs HubSpot CRM 2025")
2. Adjust **Max Sources** (3–20) and **Max Workflow Steps**
3. Click **🚀 Run Intelligence Briefing**
4. Watch the live agent progress bar
5. Review the report preview and click **✅ Approve & Publish Report**
6. Download as **Markdown** or **PDF**

### Sample Topics

- `AI CRM software market 2025`
- `Electric vehicle competitors Tesla vs Rivian`
- `Cloud computing AWS Azure GCP comparison`
- `Cybersecurity vendors 2025`
- `Healthcare AI diagnostics competitive landscape`

---

## Workflow Explanation

### 1. Supervisor
Validates the topic, initialises run metadata, and sets the workflow phase.

### 2. Research Agent
- Fires multiple search queries in parallel (ThreadPoolExecutor)
- Provider priority: Tavily → DuckDuckGo text → DuckDuckGo news → stub fallback
- Deduplicates URLs before fetching
- Fetches article content concurrently (max 5 workers)
- Parser chain: trafilatura → BeautifulSoup → raw text
- **Never returns 0 sources** — injects a Wikipedia fallback if all searches fail

### 3. Analyst Agent
Uses the LLM to extract:
- Competitor profiles (pricing, products, partnerships, acquisitions)
- Individual factual claims with categories
- Market signals, technology trends, customer trends

### 4. Fact Verification Agent
Lenient verification policy:
- **VERIFIED**: explicitly or implicitly supported by sources (confidence ≥ 0.7)
- **UNVERIFIED**: plausible but not directly evidenced — flows to writer with `[Unverified]` marker
- **REJECTED**: only claims that directly contradict sources
- Fallback: if 0 verified claims, all are marked UNVERIFIED (not rejected)
- Fallback: if 0 claims at all, creates 3 synthetic claims to keep the pipeline alive

### 5. Writer Agent
Generates 7 LLM sections:
1. Executive Summary
2. Competitor Pricing
3. Product Updates
4. Market Signals
5. Business Risks
6. Strategic Recommendations
7. Opportunities

Each section includes `[Unverified]` markers where claims lack direct source support.

### 6. Governance Agent
Checks:
- Citation coverage (warning only)
- Confidence threshold (warning only)
- Hallucination markers (refusal trigger)
- Section completeness (refusal only for truly empty sections < 10 chars)

### 7. Human Approval
Shows full report preview with Approve / Reject buttons. Rejection re-runs the writer.

---

## UI Dashboard Tabs

After a run completes:

| Tab | Contents |
|---|---|
| 📄 Report | 8-section tabbed report viewer |
| 🔗 Citations | Numbered source list with URLs |
| 🗂️ Sources | Card view of all collected sources |
| ⬇️ Download | Markdown + PDF export |
| 📊 Evaluation | Full metrics dashboard |
| 🔍 Execution Trace | Timestamped agent activity log |
| 📋 Audit Log | Governance and compliance events |
| ❌ Errors | Any errors with full messages |

---

## Running Tests

```bash
# Unit + integration tests (no LLM key needed for non-LLM tests)
pytest tests/test_integration.py -v

# Skip slow end-to-end tests
pytest tests/test_integration.py -v -m "not slow"

# All tests including full workflow (requires LLM key, ~2 min)
pytest tests/test_integration.py -v -m "slow"
```

---

## Troubleshooting

### ❌ No API key found

- Make sure `.env` exists (not just `.env.example`)
- Restart Streamlit after editing `.env`
- Check the key matches the provider (OpenRouter keys start with `sk-or-v1-`, Groq keys start with `gsk_`)

### ❌ No report generated / workflow ends early

1. Check the **Errors** tab for specific failure messages
2. Check the **Execution Trace** tab to see which agent failed
3. DuckDuckGo may be rate-limiting — wait 1–2 minutes and retry, or add a Tavily key
4. If the LLM returns an auth error, check your API key is valid and has credits

### ❌ DuckDuckGo rate limit

```env
TAVILY_API_KEY=tvly-...   # Get free at https://tavily.com
```

### ❌ Report sections are empty

The LLM model may not be smart enough for complex analysis. Try a larger model:
- OpenRouter: `openai/gpt-4o-mini` (requires credits)
- Groq: `llama-3.3-70b-versatile` (free tier)

### ❌ weasyprint PDF error on Windows

weasyprint requires GTK which is complex to install on Windows. The app falls back to reportlab automatically. This is expected behaviour.

### ❌ sentence-transformers slow to load

The first run downloads the embedding model (~90MB). Subsequent runs use the cache and are faster.

---

## Project Structure

```
competitive-intelligence-crew/
├── app.py                    # Streamlit UI
├── config.py                 # Environment config, LLM setup
├── requirements.txt
├── .env.example
├── agents/
│   ├── state.py              # All Pydantic models + BriefingState TypedDict
│   ├── base_agent.py         # Shared LLM client, trace/audit helpers
│   ├── supervisor.py
│   ├── research_agent.py
│   ├── analyst_agent.py
│   ├── fact_verification_agent.py
│   └── writer_agent.py
├── graph/
│   ├── workflow.py           # LangGraph graph build + run/stream functions
│   ├── edges.py              # Conditional routing functions
│   └── nodes.py              # awaiting_approval + end nodes
├── governance/
│   ├── governance_agent.py
│   ├── refusal_policy.py
│   ├── confidence_scorer.py
│   ├── citation_enforcer.py
│   └── source_validator.py
├── tools/
│   ├── web_search.py         # Multi-provider search (Tavily → DDG → stub)
│   ├── article_fetch.py      # Multi-parser fetch (trafilatura → BS4 → raw)
│   ├── report_generator.py
│   ├── citation_generator.py
│   ├── html_parser.py
│   ├── pdf_export.py
│   ├── markdown_export.py
│   └── audit_logger.py
├── rag/
│   ├── knowledge_base.py
│   ├── retriever.py
│   ├── embeddings.py
│   ├── document_loader.py
│   └── chunker.py
├── evaluation/
│   ├── test_suite.py
│   ├── scenarios.py
│   └── metrics.py
├── tests/
│   └── test_integration.py
├── reports/                  # Generated reports saved here
├── logs/                     # Application logs
└── data/                     # FAISS vectorstore
```

---

## Evaluation Results (Example)

Running on topic: "Salesforce vs HubSpot CRM 2025"

| Metric | Value |
|---|---|
| Workflow Steps | 8 |
| Sources Collected | 8 |
| Search Queries | 3 |
| Total Claims | 12 |
| Verified | 7 |
| Unverified | 4 |
| Rejected | 1 |
| Citation Coverage | 70% |
| Confidence Score | 68% |
| Governance | Passed |
| Execution Time | ~45s |
| Report Word Count | ~1,800 |
