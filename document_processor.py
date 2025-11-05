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
            all_chunks_raw = []
            all_metadata_raw = []
            processed_docs = []

            for i, doc_path in enumerate(document_paths):
                print(f"Processing {Path(doc_path).name} ({i+1}/{len(document_paths)})...")

                # Extract text and create chunks
                chunks, metadata = self.file_handler.process_document(doc_path)

                if chunks:
                    all_chunks_raw.extend(chunks)
                    all_metadata_raw.extend(metadata)
                    processed_docs.append({
                        "filename": Path(doc_path).name,
                        "file_path": doc_path,
                        "processed_at": datetime.now().isoformat(),
                        "chunk_count": len(chunks), # This is the raw chunk count
                        "file_size_mb": round(Path(doc_path).stat().st_size / (1024 * 1024), 2)
                    })
                    print(f"  [OK] {len(chunks)} raw chunks extracted")
                else:
                    print(f"  [FAIL] Failed to process {Path(doc_path).name}")

            if not all_chunks_raw:
                print("No chunks extracted from documents.")
                return False

            print(f"Total raw chunks: {len(all_chunks_raw)}")

            all_chunks_clean = []
            all_metadata_clean = []

            for chunk, meta in zip(all_chunks_raw, all_metadata_raw):
                if chunk and chunk.strip(): # Check for None or empty strings
                    all_chunks_clean.append(chunk)
                    all_metadata_clean.append(meta)

            print(f"Total *clean* chunks for indexing: {len(all_chunks_clean)}")

            # Create batch directory
            batch_dir = Path("batches") / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Build FAISS index
            print("Building FAISS index...")
            faiss_success = self.index_builder.build_faiss_index(
                chunks=all_chunks_clean, # Use the clean list
                metadata=all_metadata_clean, # Use the clean list
                output_dir=str(batch_dir / "faiss_index")
            )

            if not faiss_success:
                print("Failed to build FAISS index")
                return False

            # Build BM25 index
            print("Building BM25 index...")
            bm25_success = self.index_builder.build_bm25_index(
                chunks=all_chunks_clean, # Use the clean list
                metadata=all_metadata_clean, # Use the clean list
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
                    "total_chunks": len(all_chunks_clean)
                }
            }

            with open(batch_dir / "metadata.json", 'w') as f:
                json.dump(batch_metadata, f, indent=2)

            # Register batch
            # We also fix the cosmetic bug where the chunk count wasn't passed
            self.batch_manager.register_batch(batch_id, {
                "name": batch_metadata["name"],
                "description": description,
                "doc_count": len(processed_docs),
                "created_at": batch_metadata["created_at"],
                "chunk_count": len(all_chunks_clean) # Pass the correct count
            })

            print("[SUCCESS] Batch created successfully!")
            return True

        except Exception as e:
            print(f"Error creating batch: {e}")
            return False