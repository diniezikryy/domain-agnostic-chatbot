"""
Simple wrapper to add retry/backoff to OpenAI client methods using tenacity.

This module monkey-patches the provided OpenAI client instance's
chat.completions.create and embeddings.create methods with tenacity
retries that detect rate-limit and transient network failures.

It is intentionally conservative: if tenacity isn't installed, it
becomes a no-op and prints a helpful message.
"""
from typing import Callable
import inspect
import functools

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
except Exception:  # pragma: no cover - best-effort runtime
    retry = None
    stop_after_attempt = None
    wait_exponential = None
    retry_if_exception = None

def _is_retryable_exception(exc: BaseException) -> bool:
    """Return True for exceptions that are likely transient/rate-limit related."""
    if exc is None:
        return False
    text = str(exc).lower()
    # common OpenAI rate limit messages
    if any(x in text for x in ("rate limit", "rate_limit", "rate-limit", "429", "tpm", "please try again")):
        return True

    # httpx HTTPStatusError with 429
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            try:
                if exc.response.status_code == 429:
                    return True
            except Exception:
                pass
    except Exception:
        # httpx may not be available; ignore
        pass

    # OpenAI SDK RateLimitError if available
    try:
        import openai
        if hasattr(openai, 'error') and hasattr(openai.error, 'RateLimitError'):
            if isinstance(exc, openai.error.RateLimitError):
                return True
    except Exception:
        pass

    return False


def _wrap_sync(func: Callable) -> Callable:
    """Wrap a synchronous function with tenacity retry if available."""
    if retry is None:
        return func

    @functools.wraps(func)
    @retry(retry=retry_if_exception(_is_retryable_exception), wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6), reraise=True)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


def wrap_openai_client(client) -> None:
    """Monkey-patch an OpenAI client instance to add retry/backoff.

    The function will attempt to replace client.chat.completions.create and
    client.embeddings.create with retrying wrappers. If the attributes are not
    present, it will silently skip them.
    """
    try:
        # chat completions
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions') and hasattr(client.chat.completions, 'create'):
            original = client.chat.completions.create
            client.chat.completions._original_create = original
            client.chat.completions.create = _wrap_sync(original)

        # embeddings
        if hasattr(client, 'embeddings') and hasattr(client.embeddings, 'create'):
            original_e = client.embeddings.create
            client.embeddings._original_create = original_e
            client.embeddings.create = _wrap_sync(original_e)

    except Exception as e:  # pragma: no cover - defensive
        print(f"[openai_retry] Could not wrap OpenAI client for retries: {e}")
