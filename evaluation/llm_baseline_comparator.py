"""Simple baseline LLM comparator for generating non-RAG responses.

This module contains a small compatibility layer that extracts token usage
robustly from various OpenAI SDK response shapes (attribute-style, mapping-style,
or SDK-specific types). It also supports optional debug output via the
EXPERIMENT_DEBUG_USAGE environment variable.
"""

import os
import time
from openai import OpenAI
from config.settings import Settings
from typing import Any


def _extract_total_tokens(response: Any) -> int:
    """Robustly extract total token usage from a response object.

    Attempts multiple access patterns to handle SDK versions and model families.
    Returns 0 when no usable count is found.
    """
    try:
        # 1) Attribute-style: response.usage.total_tokens
        usage = getattr(response, 'usage', None)
        if usage is not None:
            # attribute on usage
            total = getattr(usage, 'total_tokens', None)
            if total is not None:
                return int(total)
            # mapping-like usage
            try:
                u_map = dict(usage)
                if 'total_tokens' in u_map:
                    return int(u_map['total_tokens'])
            except Exception:
                pass

        # 2) Mapping-style top-level: response['usage']['total_tokens']
        try:
            r_map = dict(response)
            u = r_map.get('usage')
            if isinstance(u, dict) and 'total_tokens' in u:
                return int(u['total_tokens'])
        except Exception:
            pass

        # 3) If response is a plain dict
        if isinstance(response, dict):
            u = response.get('usage')
            if isinstance(u, dict) and 'total_tokens' in u:
                return int(u['total_tokens'])

    except Exception:
        # swallow any parsing errors and fall through to default
        pass

    return 0


class BaselineLLMComparator:
    """Generates baseline LLM responses without RAG."""

    def __init__(self, model: str = "gpt-4o-mini", user_profile: dict | None = None):
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.user_profile = user_profile or {}

    def get_baseline_response(self, query: str) -> dict:
        """Get a baseline response from the LLM without RAG.

        Returns a dict with keys: 'response', 'time_seconds', 'tokens_used'.
        """
        start_time = time.time()

        try:
            # Build personalized system message if profile provided
            system_message = "You are a helpful assistant. Answer the user's question directly and concisely."
            if self.user_profile:
                profile_summary = self._format_profile_context()
                system_message = f"You are a helpful insurance advisor assisting {profile_summary}. Answer the user's question directly and concisely, considering their profile context where relevant."

            # Default chat completion call (some models may accept slightly
            # different parameter names; SDK will raise if unsupported and we'll
            # still capture usage where possible).
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                # Use defaults where model-specific parameters differ; SDK will
                # raise if unsupported and we'll still capture usage where possible.
            )

            # Extract answer text (defensive)
            try:
                answer = response.choices[0].message.content
            except Exception:
                # Fallback to str of the first choice
                try:
                    answer = str(response.choices[0])
                except Exception:
                    answer = ''

            tokens_used = _extract_total_tokens(response)

            # Optional debug: dump raw usage when requested
            if os.getenv('EXPERIMENT_DEBUG_USAGE') == '1':
                try:
                    print('DEBUG: raw baseline response repr:')
                    print(repr(response))
                except Exception:
                    pass
                print('DEBUG: extracted tokens_used =', tokens_used)

            elapsed_time = time.time() - start_time
            return {
                'response': answer,
                'time_seconds': elapsed_time,
                'tokens_used': tokens_used
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            # Try to salvage tokens from the exception response if present
            try:
                # Some exceptions may include a response-like object; attempt extraction
                tokens_used = _extract_total_tokens(getattr(e, 'response', {}) or {})
            except Exception:
                tokens_used = 0

            return {
                'response': f"Error generating baseline response: {str(e)}",
                'time_seconds': elapsed_time,
                'tokens_used': tokens_used
            }
    
    def _format_profile_context(self) -> str:
        """Format user profile into a concise context string."""
        if not self.user_profile:
            return ""
        
        profile = self.user_profile
        name = profile.get("name", "User")
        age = profile.get("age", "Unknown")
        location = profile.get("location", "")
        policies = profile.get("policies", [])
        
        policy_names = [p.get("name", "") for p in policies if p.get("name")]
        policy_str = ", ".join(policy_names) if policy_names else "no policies"
        
        context = f"{name}, age {age}"
        if location:
            context += f", based in {location}"
        context += f", with {policy_str}"
        
        return context
