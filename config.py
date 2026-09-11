"""
Configuration module for Competitive Intelligence Briefing Crew.
Loads and validates all environment variables and application settings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default="openrouter")
    model: str = Field(default="openai/gpt-4o-mini")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256)
    request_timeout: int = Field(default=60, ge=10)


class SearchConfig(BaseModel):
    """Web search configuration."""

    max_results: int = Field(default=10, ge=1, le=50)
    max_retries: int = Field(default=3, ge=1)
    timeout: int = Field(default=30, ge=5)
    trusted_domains: List[str] = Field(
        default_factory=lambda: [
            "reuters.com",
            "bloomberg.com",
            "techcrunch.com",
            "forbes.com",
            "wsj.com",
            "ft.com",
            "businessinsider.com",
            "venturebeat.com",
            "gartner.com",
            "mckinsey.com",
            "hbr.org",
            "arxiv.org",
            "sec.gov",
            "crunchbase.com",
            "pitchbook.com",
        ]
    )


class RAGConfig(BaseModel):
    """RAG pipeline configuration."""

    enabled: bool = Field(default=True)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    vector_store: str = Field(default="faiss")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = Field(default=5, ge=1)
    vectorstore_path: Path = Field(
        default_factory=lambda: BASE_DIR / "data" / "vectorstore"
    )

    class Config:
        arbitrary_types_allowed = True


class GovernanceConfig(BaseModel):
    """Governance and safety configuration."""

    min_citation_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    min_confidence_score: float = Field(default=0.6, ge=0.0, le=1.0)
    max_unverified_claims: int = Field(default=0, ge=0)
    enable_refusal_policy: bool = Field(default=True)
    require_human_approval: bool = Field(default=True)


class WorkflowConfig(BaseModel):
    """LangGraph workflow configuration."""

    max_steps: int = Field(default=20, ge=5, le=100)
    max_search_queries: int = Field(default=15, ge=1)
    max_retries: int = Field(default=3, ge=1)
    enable_rag: bool = Field(default=True)


class PathConfig(BaseModel):
    """File system paths."""

    base_dir: Path = Field(default_factory=lambda: BASE_DIR)
    reports_dir: Path = Field(default_factory=lambda: BASE_DIR / "reports")
    logs_dir: Path = Field(default_factory=lambda: BASE_DIR / "logs")
    data_dir: Path = Field(default_factory=lambda: BASE_DIR / "data")
    assets_dir: Path = Field(default_factory=lambda: BASE_DIR / "assets")

    class Config:
        arbitrary_types_allowed = True

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for attr in ["reports_dir", "logs_dir", "data_dir", "assets_dir"]:
            path: Path = getattr(self, attr)
            path.mkdir(parents=True, exist_ok=True)
        # Also create vectorstore subdirectory
        (self.data_dir / "vectorstore").mkdir(parents=True, exist_ok=True)


# ── Module-level singletons ──────────────────────────────────────────────────

llm_config = LLMConfig(
    provider=os.getenv("LLM_PROVIDER", "openrouter"),
    model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
    request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
)

search_config = SearchConfig(
    max_results=int(os.getenv("MAX_SEARCH_RESULTS", "15")),
    max_retries=int(os.getenv("MAX_RETRIES", "3")),
    timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
)

rag_config = RAGConfig(
    enabled=os.getenv("RAG_ENABLED", "true").lower() == "true",
    chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
    chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
    vector_store=os.getenv("VECTOR_STORE", "faiss"),
)

governance_config = GovernanceConfig()

workflow_config = WorkflowConfig(
    max_steps=int(os.getenv("MAX_WORKFLOW_STEPS", "20")),
    max_search_queries=int(os.getenv("MAX_SEARCH_RESULTS", "15")),
    max_retries=int(os.getenv("MAX_RETRIES", "3")),
)

path_config = PathConfig()
path_config.ensure_dirs()

# API keys (read at runtime)
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")


def _is_real_key(value: Optional[str]) -> bool:
    """
    Return True only if *value* looks like a real API key.
    Filters out common placeholder strings that get left in .env files.
    """
    if not value:
        return False
    placeholders = {
        "your_openai_api_key_here",
        "your_openrouter_api_key_here",
        "your_groq_api_key_here",
        "your_tavily_api_key_here",
        "your_langsmith_api_key_here",
        "your_api_key_here",
        "placeholder",
        "changeme",
        "xxxx",
    }
    stripped = value.strip().lower()
    if stripped in placeholders:
        return False
    if stripped.startswith("your_") or stripped.endswith("_here"):
        return False
    # Must be at least 20 chars to be a real key
    return len(stripped) >= 20


def get_active_api_key() -> str:
    """
    Return the first *real* LLM API key found, in priority order:
    OpenRouter → Groq → OpenAI.

    Placeholder values (e.g. 'your_openai_api_key_here') are skipped.
    """
    if _is_real_key(OPENROUTER_API_KEY):
        return OPENROUTER_API_KEY  # type: ignore[return-value]
    if _is_real_key(GROQ_API_KEY):
        return GROQ_API_KEY  # type: ignore[return-value]
    if _is_real_key(OPENAI_API_KEY):
        return OPENAI_API_KEY  # type: ignore[return-value]
    raise EnvironmentError(
        "No valid LLM API key found. Set OPENROUTER_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY "
        "in your .env file (current OPENAI_API_KEY appears to be a placeholder)."
    )


def get_llm_base_url() -> str:
    """Return the base URL for the active LLM provider."""
    if _is_real_key(OPENROUTER_API_KEY):
        return "https://openrouter.ai/api/v1"
    if _is_real_key(GROQ_API_KEY):
        # Use the LLM_MODEL env var as-is — Groq's API accepts its own model IDs
        # (e.g. "groq/compound-mini") directly.
        llm_config.model = os.getenv("LLM_MODEL", "groq/compound-mini")
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


def configure_logging() -> logging.Logger:
    """Configure application-wide logging."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = path_config.logs_dir / "app.log"
    try:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except Exception:
        pass  # non-fatal if log dir not writable

    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Quiet noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "requests", "openai"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("competitive_intelligence")


logger = configure_logging()
