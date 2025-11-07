#!/usr/bin/env python3
"""
Create a batch using text-embedding-3-large embeddings.
This script overrides the DocumentProcessor's embedding generator to use the large model.
"""
import sys
import os
from pathlib import Path

# Ensure repo root is in path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from document_processor import DocumentProcessor
from utils.embeddings import EmbeddingGenerator

BATCH_ID = "my_policies_large"
SOURCE_DIR = repo_root / "documents" / "my_policies"


def main():
    if not SOURCE_DIR.exists():
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    # Collect supported document files
    doc_files = []
    for ext in ("*.pdf", "*.docx", "*.txt", "*.md"):
        doc_files.extend(SOURCE_DIR.glob(ext))

    if not doc_files:
        print(f"No documents found in {SOURCE_DIR}")
        return

    print(f"Found {len(doc_files)} documents. Will create batch '{BATCH_ID}' using text-embedding-3-large.")

    dp = DocumentProcessor()
    # Override the embedding generator to use the large model
    dp.embedding_generator = EmbeddingGenerator(model_name="text-embedding-3-large")

    success = dp.create_batch(
        batch_id=BATCH_ID,
        document_paths=[str(p) for p in doc_files],
        batch_name="My Policies (Large Embeddings)",
        description="Batch created with text-embedding-3-large for testing."
    )

    if success:
        print(f"Batch '{BATCH_ID}' created successfully.")
    else:
        print(f"Batch '{BATCH_ID}' creation failed.")


if __name__ == '__main__':
    main()
