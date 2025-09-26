"""
Configuration Settings
Manages application configuration and environment variables.
"""

from dotenv import load_dotenv

# Load .env file
load_dotenv()
import os
from pathlib import Path
from typing import Dict, Any

class Settings:
    """Application settings."""

    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Paths
        self.batches_dir = "batches"
        self.documents_dir = "documents"

        # Document processing
        self.chunk_size = 800
        self.chunk_overlap = 100

        # Search settings
        self.faiss_top_k = 10
        self.bm25_top_k = 10
        self.faiss_weight = 0.6
        self.bm25_weight = 0.4

        # Embedding model
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536

        # Query processing
        self.max_context_length = 4000
        self.response_model = "gpt-4o-mini"

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not set")
            return False

        # Create directories if they don't exist
        Path(self.batches_dir).mkdir(exist_ok=True)
        Path(self.documents_dir).mkdir(exist_ok=True)

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "faiss_top_k": self.faiss_top_k,
            "bm25_top_k": self.bm25_top_k,
            "faiss_weight": self.faiss_weight,
            "bm25_weight": self.bm25_weight,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "max_context_length": self.max_context_length,
            "response_model": self.response_model
        }

# Global settings instance
settings = Settings()