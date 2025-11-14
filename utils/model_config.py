"""Helper for retrieving environment-configured model names."""

import os
from typing import Dict, Tuple

MODEL_STAGE_CONFIG: Dict[str, Tuple[str, str]] = {
    "intent": ("OPENAI_INTENT_MODEL", "gpt-4o-mini"),
    "expansion": ("OPENAI_EXPANSION_MODEL", "gpt-4o-mini"),
    "generation": ("OPENAI_GENERATION_MODEL", "gpt-4o"),
    "stream": ("OPENAI_STREAM_MODEL", "gpt-4-1106-preview"),
    "hyde": ("OPENAI_HYDE_MODEL", "gpt-4o-mini"),
    "evaluation": ("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
    "research": ("OPENAI_RESEARCH_MODEL", "gpt-4-1106-preview"),
}


def get_model_name(stage: str) -> str:
    """Return the configured model name for a given stage."""
    config = MODEL_STAGE_CONFIG.get(stage)
    if config is None:
        raise KeyError(f"Unknown model stage: {stage}")
    env_var, default = config
    return os.getenv(env_var, default)
