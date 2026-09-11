"""
Base agent utilities – shared LLM client factory and trace helpers
used by all agents in the pipeline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from agents.state import AgentStatus, AuditLogEntry, BriefingState, TraceEntry
from config import get_active_api_key, get_llm_base_url, llm_config, GROQ_API_KEY, _is_real_key

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Groq model fallback chain ─────────────────────────────────────────────────
# When a model's daily quota (TPD / RPD) is exhausted, we transparently switch
# to the next model in this list.  Ordered: best quality first.
_GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound-mini",
]

# Models confirmed exhausted for this process lifetime (reset on restart).
_exhausted_models: set = set()

# Currently active model name.  Set by _pick_groq_model(); may differ from
# llm_config.model after a runtime fallback.
_active_model: str = ""


# ── Error classifiers ─────────────────────────────────────────────────────────

def _is_quota_error(msg: str) -> bool:
    """True when the error is a *daily* quota exhaustion (TPD / RPD)."""
    m = msg.lower()
    return (
        ("tpd" in m and ("limit" in m or "exceed" in m)) or
        ("rpd" in m and ("limit" in m or "exceed" in m)) or
        "tokens per day" in m or
        "requests per day" in m or
        ("daily" in m and ("limit" in m or "exceed" in m))
    )


def _is_rate_limit(msg: str) -> bool:
    """True for any 429-class error (transient spike OR quota)."""
    return "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()


def _is_too_large(msg: str) -> bool:
    return "413" in msg or "request_too_large" in msg.lower() or "request entity too large" in msg.lower()


# ── LLM builder ───────────────────────────────────────────────────────────────

def _make_llm(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    """Build a ChatOpenAI instance pointed at Groq with the given model."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        timeout=llm_config.request_timeout,
    )


def _pick_groq_model() -> str:
    """
    Return the best available Groq model without sending any probe call.

    Strategy:
    1. If _active_model is set and NOT exhausted → reuse it.
    2. Walk the fallback chain; return first non-exhausted model.
    3. If all models are exhausted → log error, return the configured model.
    """
    global _active_model

    if _active_model and _active_model not in _exhausted_models:
        return _active_model

    # get_llm_base_url() sets llm_config.model to the correct Groq model name
    get_llm_base_url()
    configured = llm_config.model
    candidates = [configured] + [m for m in _GROQ_FALLBACK_MODELS if m != configured]

    for model in candidates:
        if model not in _exhausted_models:
            if model != _active_model:
                if _active_model:
                    logger.info("[get_llm] Model switch: '%s' -> '%s' (previous model exhausted)",
                                _active_model, model)
                else:
                    logger.info("[get_llm] Selected model: '%s'", model)
            _active_model = model
            llm_config.model = model
            return model

    logger.error(
        "[get_llm] All Groq models quota-exhausted. Daily limits reset at midnight UTC. "
        "Returning '%s' anyway.", configured
    )
    _active_model = configured
    return configured


# ── Core retry helper ─────────────────────────────────────────────────────────

def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 5.0,
    label: str = "",
) -> T:
    """
    Call *fn* with automatic retry on 429 / 413 errors.

    QUOTA EXHAUSTION HANDLING
    -------------------------
    When a Groq quota-exhaustion 429 is detected the function:
      1. Marks the current _active_model as exhausted.
      2. Calls _pick_groq_model() to select the next available model.
      3. Does NOT sleep (the new model has fresh quota).
      4. Retries the call — note that *fn* must close over a mutable llm
         reference (or use get_llm() internally) to pick up the new model.
         Callers should therefore NOT capture a hardcoded llm object in fn;
         instead they should call get_fresh_llm() inside the lambda, or use
         call_llm_with_retry() (below) which handles this automatically.

    Args:
        fn:          Zero-arg callable that performs the LLM call.
        max_retries: Maximum retry attempts (default 3).
        base_delay:  Initial back-off for transient rate spikes (doubles each retry).
        label:       Human-readable label for log messages.
    """
    global _active_model

    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)

            # ── Daily quota exhausted: switch model, retry immediately ────────
            if _is_quota_error(msg):
                exhausted = _active_model
                if exhausted:
                    logger.warning(
                        "[retry] Quota exhausted for '%s'%s — switching model (attempt %d/%d)",
                        exhausted, f" ({label})" if label else "", attempt + 1, max_retries,
                    )
                    _exhausted_models.add(exhausted)
                    _active_model = ""   # force re-evaluation on next _pick_groq_model() call
                    _pick_groq_model()   # updates _active_model + llm_config.model
                last_exc = exc
                continue  # retry immediately on the new model

            # ── Transient rate spike or payload too large: back off ───────────
            if _is_rate_limit(msg) or _is_too_large(msg):
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "[retry] %s%s — waiting %.0fs before retry %d/%d",
                        "Rate-limit" if _is_rate_limit(msg) else "Payload too large",
                        f" ({label})" if label else "", delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning("[retry] All %d retries exhausted%s",
                                   max_retries, f" ({label})" if label else "")
                continue

            # ── Any other error: raise immediately ────────────────────────────
            raise

    raise last_exc  # type: ignore[misc]


def call_llm_with_retry(
    messages: List[BaseMessage],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    base_delay: float = 5.0,
    label: str = "",
) -> Any:
    """
    High-level helper: build an LLM, invoke it with *messages*, and retry
    with automatic model switching on quota exhaustion.

    Unlike call_with_retry() + a lambda, this function rebuilds the LLM
    object after every model switch so the new model is actually used.

    Use this in agents instead of:
        call_with_retry(lambda: llm.invoke(messages), ...)

    Args:
        messages:    LangChain message list (SystemMessage + HumanMessage, etc.)
        temperature: Override LLM temperature.
        max_tokens:  Override max tokens.
        max_retries: Maximum retry/fallback attempts.
        base_delay:  Initial back-off for transient rate spikes.
        label:       Human-readable label for log messages.

    Returns:
        LLM response object.
    """
    global _active_model   # declared at top so all reads/writes in this function are to the global

    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        # Always build a fresh LLM so we use the currently-active model
        llm = get_llm(temperature=temperature, max_tokens=max_tokens)
        try:
            return llm.invoke(messages)
        except Exception as exc:
            msg = str(exc)

            # Daily quota exhausted → switch model
            if _is_quota_error(msg):
                exhausted = _active_model
                if exhausted:
                    logger.warning(
                        "[call_llm] Quota exhausted for '%s'%s — switching model (attempt %d/%d)",
                        exhausted, f" ({label})" if label else "", attempt + 1, max_retries,
                    )
                    _exhausted_models.add(exhausted)
                    _active_model = ""
                    _pick_groq_model()
                last_exc = exc
                continue   # rebuild llm on next iteration

            # Transient rate spike or too-large
            if _is_rate_limit(msg) or _is_too_large(msg):
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "[call_llm] %s%s — waiting %.0fs before retry %d/%d",
                        "Rate-limit" if _is_rate_limit(msg) else "Payload too large",
                        f" ({label})" if label else "", delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning("[call_llm] All %d retries exhausted%s",
                                   max_retries, f" ({label})" if label else "")
                continue

            raise   # non-retryable error

    raise last_exc  # type: ignore[misc]


def get_llm(temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> ChatOpenAI:
    """
    Return a ChatOpenAI client for the active Groq model (no probe call).

    Model selection is done by _pick_groq_model() which reads the
    _exhausted_models set and picks the best remaining option without
    making any API call.  If a subsequent real call gets a quota error,
    call_with_retry() / call_llm_with_retry() handle the switch.

    Args:
        temperature: Override temperature (uses config default if None).
        max_tokens:  Override max_tokens (uses config default if None).
    """
    base_url = get_llm_base_url()   # mutates llm_config.model for Groq
    api_key  = get_active_api_key()
    temp     = temperature if temperature is not None else llm_config.temperature
    tokens   = max_tokens or llm_config.max_tokens

    if "groq.com" in base_url and _is_real_key(GROQ_API_KEY):
        model = _pick_groq_model()
        return _make_llm(model, temp, tokens)

    # Non-Groq provider
    return ChatOpenAI(
        model=llm_config.model,
        temperature=temp,
        max_tokens=tokens,
        api_key=api_key,
        base_url=base_url,
        timeout=llm_config.request_timeout,
    )


# ── State helpers ─────────────────────────────────────────────────────────────

def add_trace(
    state: BriefingState,
    agent: str,
    phase: str,
    message: str,
    duration_ms: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    entry = TraceEntry(agent=agent, phase=phase, message=message,
                       duration_ms=duration_ms, metadata=metadata or {})
    trace: List[TraceEntry] = state.get("execution_trace", [])
    trace.append(entry)
    state["execution_trace"] = trace


def add_audit(
    state: BriefingState,
    event_type: str,
    agent: str,
    description: str,
    data: Optional[Dict[str, Any]] = None,
    severity: str = "info",
) -> None:
    entry = AuditLogEntry(event_type=event_type, agent=agent,
                          description=description, data=data or {}, severity=severity)
    audit_log: List[AuditLogEntry] = state.get("audit_log", [])
    audit_log.append(entry)
    state["audit_log"] = audit_log


def add_error(state: BriefingState, message: str) -> None:
    errors: List[str] = state.get("errors", [])
    errors.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {message}")
    state["errors"] = errors
    meta = state.get("run_metadata")
    if meta:
        meta.errors += 1


def set_agent_status(state: BriefingState, agent_name: str, status: AgentStatus) -> None:
    tracker = state.get("agent_status")
    if tracker and hasattr(tracker, agent_name):
        setattr(tracker, agent_name, status)


def increment_step(state: BriefingState) -> int:
    step = state.get("current_step", 0) + 1
    state["current_step"] = step
    meta = state.get("run_metadata")
    if meta:
        meta.total_steps = step
    return step


def check_runaway(state: BriefingState) -> bool:
    step     = state.get("current_step", 0)
    max_steps = state.get("max_steps", 20)
    if step >= max_steps:
        state["should_terminate"] = True
        state["termination_reason"] = f"Max steps ({max_steps}) exceeded at step {step}"
        logger.warning("Runaway guard triggered at step %d/%d", step, max_steps)
        return True
    return False
