"""
Document Processor
Handles processing documents into batches with FAISS and BM25 indexes.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import utilities (to be created)
from utils.file_handlers import FileHandler
from utils.embeddings import EmbeddingGenerator
from utils.search import SearchIndexBuilder
from batch_manager import BatchManager

class DocumentProcessor:
    def __init__(self):
        self.file_handler = FileHandler()
        self.embedding_generator = EmbeddingGenerator()
        self.index_builder = SearchIndexBuilder()
        self.batch_manager = BatchManager()

    def create_batch(self, batch_id: str, document_paths: List[str],
                    batch_name: str = None, description: str = "") -> bool:
        """Create a new document batch with FAISS and BM25 indexes."""
        try:
            print(f"Processing {len(document_paths)} documents...")

            # Process all documents into chunks
            all_chunks = []
            all_metadata = []
            processed_docs = []

            for i, doc_path in enumerate(document_paths):
                print(f"Processing {Path(doc_path).name} ({i+1}/{len(document_paths)})...")

                # Extract text and create chunks
                chunks, metadata = self.file_handler.process_document(doc_path)

                if chunks:
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
                    processed_docs.append({
                        "filename": Path(doc_path).name,
                        "file_path": doc_path,
                        "processed_at": datetime.now().isoformat(),
                        "chunk_count": len(chunks),
                        "file_size_mb": round(Path(doc_path).stat().st_size / (1024 * 1024), 2)
                    })
                    print(f"{len(chunks)} chunks extracted")
                else:
                    print(f"Failed to process {Path(doc_path).name}")

            if not all_chunks:
                print("No chunks extracted from documents.")
                return False

            print(f"Total chunks: {len(all_chunks)}")

            # Create batch directory
            batch_dir = Path("batches") / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Build FAISS index
            print("Building FAISS index...")
            faiss_success = self.index_builder.build_faiss_index(
                chunks=all_chunks,
                output_dir=str(batch_dir / "faiss_index")
            )

            if not faiss_success:
                print("Failed to build FAISS index")
                return False

            # Build BM25 index
            print("Building BM25 index...")
            bm25_success = self.index_builder.build_bm25_index(
                chunks=all_chunks,
                metadata=all_metadata,
                output_file=str(batch_dir / "bm25_index.pkl")
            )

            if not bm25_success:
                print("Failed to build BM25 index")
                return False

            # Create metadata file
            batch_metadata = {
                "batch_id": batch_id,
                "name": batch_name or batch_id.replace('_', ' ').title(),
                "description": description,
                "created_at": datetime.now().isoformat(),
                "documents": processed_docs,
                "statistics": {
                    "total_documents": len(processed_docs),
                    "total_chunks": len(all_chunks),
                    "avg_chunk_size": round(sum(len(chunk) for chunk in all_chunks) / len(all_chunks))
                }
            }

            with open(batch_dir / "metadata.json", 'w') as f:
                json.dump(batch_metadata, f, indent=2)

            # Register batch
            self.batch_manager.register_batch(batch_id, {
                "name": batch_metadata["name"],
                "description": description,
                "doc_count": len(processed_docs),
                "created_at": batch_metadata["created_at"]
            })

            print("Batch created successfully!")
            return True

        except Exception as e:
            print(f"Error creating batch: {e}")
            return False