"""Utility for deriving safe throttling delays from TPM limits."""

import math
import os
from typing import Optional

DEFAULT_TPM_ENV = "OPENAI_SAFE_TPM_LIMIT"  # tokens per minute limit reported by OpenAI
DEFAULT_TOKENS_ENV = "OPENAI_TOKENS_PER_REQUEST"  # estimated tokens per request
DEFAULT_SAFETY_ENV = "OPENAI_SAFETY_FACTOR"  # multiplier (0 < factor <= 1)


def _get_int_env(var_name: str, default: int) -> int:
    try:
        value = os.getenv(var_name)
        if value is None:
            return default
        return int(value)
    except ValueError:
        return default


def _get_float_env(var_name: str, default: float) -> float:
    try:
        value = os.getenv(var_name)
        if value is None:
            return default
        return float(value)
    except ValueError:
        return default


def compute_safe_delay(
    *,
    limit_tpm: Optional[int] = None,
    tokens_per_request: Optional[int] = None,
    safety_factor: Optional[float] = None,
) -> float:
    """Compute a safe delay between requests to stay within the TPM budget."""
    limit = limit_tpm if limit_tpm is not None else _get_int_env(DEFAULT_TPM_ENV, 30000)
    tokens = tokens_per_request if tokens_per_request is not None else _get_int_env(DEFAULT_TOKENS_ENV, 2500)
    safety = safety_factor if safety_factor is not None else _get_float_env(DEFAULT_SAFETY_ENV, 0.8)

    if limit <= 0 or tokens <= 0 or safety <= 0:
        return 0.0

    safe_tpm = limit * safety
    requests_per_minute = math.floor(safe_tpm / tokens)
    requests_per_minute = max(1, requests_per_minute)
    delay_seconds = 60.0 / requests_per_minute
    return delay_seconds
