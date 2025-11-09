"""Core deep research functionality"""
import os
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict, Callable
from dataclasses import dataclass

# Load environment variables if not already loaded
if not os.getenv("OPENAI_KEY") and not os.getenv("FIRECRAWL_KEY"):
    try:
        from dotenv import load_dotenv
        # Load environment variables from .env.local in the project root
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env.local"
        load_dotenv(env_path)
    except ImportError:
        print("Warning: python-dotenv not installed. Environment variables may not be loaded from .env.local")

from tavily import TavilyClient

# Initialize Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Concurrency limit (keep for asyncio)
CONCURRENCY_LIMIT = int(os.getenv("TAVILY_CONCURRENCY", "2"))

from .ai.providers import generate_object, trim_prompt, parse_response
from .prompt import system_prompt

# Your existing deep research code here
# [Rest of your original deep_research.py content]