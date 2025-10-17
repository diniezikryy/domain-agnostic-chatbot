"""
Search Components
Handles FAISS and BM25 search indexing and querying.
Combines functionality from metric_query_faiss.py and bm25_chunk_search.py.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

class SearchIndexBuilder:
    """Builds FAISS and BM25 search indexes."""

    def __init__(self):
        self.embedding_generator = None

    def build_faiss_index(self, chunks: List[str], metadata: List[Dict], output_dir: str) -> bool:
        """Build FAISS index from text chunks."""
        try:
            import faiss
            from utils.embeddings import EmbeddingGenerator

            if not self.embedding_generator:
                self.embedding_generator = EmbeddingGenerator()

            print("Generating embeddings for FAISS index...")
            embeddings = self.embedding_generator.generate_embeddings(chunks)

            if not embeddings:
                print("Failed to generate embeddings")
                return False

            # Convert to numpy array
            embedding_matrix = np.array(embeddings).astype('float32')

            # Create FAISS index
            dimension = embedding_matrix.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embedding_matrix)
            index.add(embedding_matrix)

            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(index, str(Path(output_dir) / "index.faiss"))

            # Save chunks and metadata
            with open(Path(output_dir) / "index.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'metadata': metadata,
                }, f)

            print(f"FAISS index saved to {output_dir}")
            return True

        except ImportError:
            print("FAISS not installed. Install with: pip install faiss-cpu")
            return False
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            return False

    def build_bm25_index(self, chunks: List[str], metadata: List[Dict], output_file: str) -> bool:
        """Build BM25 index from text chunks."""
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize chunks for BM25
            tokenized_chunks = [chunk.lower().split() for chunk in chunks]

            # Create BM25 index
            bm25 = BM25Okapi(tokenized_chunks)

            # Save BM25 index with chunks and metadata
            with open(output_file, 'wb') as f:
                pickle.dump((bm25, chunks, metadata), f)

            print(f"BM25 index saved to {output_file}")
            return True

        except ImportError:
            print("rank-bm25 not installed. Install with: pip install rank-bm25")
            return False
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            return False


class HybridSearchEngine:
    """Performs hybrid search using both FAISS and BM25."""

    def __init__(self):
        self.faiss_index = None
        self.faiss_chunks = []
        self.faiss_metadata = []
        self.bm25_index = None
        self.bm25_chunks = []
        self.bm25_metadata = []
        self.embedding_generator = None

    def load_indexes(self, faiss_path: str, bm25_path: str) -> bool:
        """Load FAISS and BM25 indexes."""
        try:
            # Load FAISS index
            if not self._load_faiss_index(faiss_path):
                return False

            # Load BM25 index
            if not self._load_bm25_index(bm25_path):
                return False

            print("Both indexes loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading indexes: {e}")
            return False

    def _load_faiss_index(self, faiss_path: str) -> bool:
        """Load FAISS index."""
        try:
            import faiss
            from utils.embeddings import EmbeddingGenerator

            faiss_dir = Path(faiss_path)
            index_file = faiss_dir / "index.faiss"
            chunks_file = faiss_dir / "index.pkl"

            if not index_file.exists() or not chunks_file.exists():
                print(f"FAISS index files not found in {faiss_path}")
                return False

            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_file))

            # Load chunks
            with open(chunks_file, 'rb') as f:
                data = pickle.load(f)
                self.faiss_chunks = data['chunks']
                self.faiss_metadata = data.get('metadata', [])

            # Initialize embedding generator for query embeddings
            if not self.embedding_generator:
                self.embedding_generator = EmbeddingGenerator()

            print(f"FAISS index loaded: {len(self.faiss_chunks)} chunks")
            return True

        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False

    def _load_bm25_index(self, bm25_path: str) -> bool:
        """Load BM25 index."""
        try:
            if not Path(bm25_path).exists():
                print(f"BM25 index not found: {bm25_path}")
                return False

            with open(bm25_path, 'rb') as f:
                self.bm25_index, self.bm25_chunks, self.bm25_metadata = pickle.load(f)

            print(f"BM25 index loaded: {len(self.bm25_chunks)} chunks")
            return True

        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False

    def hybrid_search(self, query: str, top_k: int = 10,
                     required_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining FAISS and BM25 results.

        Args:
            query: Search query string
            top_k: Number of results to return
            required_sources: Optional list of required source identifiers (e.g., ['singlife', 'fwd'])
                            to ensure balanced retrieval for comparisons
        """
        if not self.faiss_index or not self.bm25_index:
            print("Indexes not loaded")
            return []

        try:
            # Get FAISS results (semantic search)
            faiss_results = self._faiss_search(query, top_k * 2)  # Get more for filtering

            # Get BM25 results (keyword search)
            bm25_results = self._bm25_search(query, top_k * 2)  # Get more for filtering

            # Combine and rank results
            combined_results = self._combine_results(faiss_results, bm25_results, top_k * 2)

            # If required_sources specified, ensure balanced representation
            if required_sources:
                combined_results = self._balance_by_sources(combined_results, required_sources, top_k)
            else:
                combined_results = combined_results[:top_k]

            return combined_results

        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def _balance_by_sources(self, results: List[Dict[str, Any]],
                           required_sources: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Ensure balanced representation from required sources."""

        # Categorize results by source
        source_results = {source: [] for source in required_sources}
        other_results = []

        for result in results:
            filename = result.get('metadata', {}).get('filename', '').lower()
            content = result.get('content', '').lower()

            categorized = False
            for source in required_sources:
                if source.lower() in filename or source.lower() in content:
                    source_results[source].append(result)
                    categorized = True
                    break

            if not categorized:
                other_results.append(result)

        # Interleave results from each required source
        balanced = []
        max_per_source = max(1, top_k // len(required_sources))

        # Round-robin selection from each source
        for source in required_sources:
            balanced.extend(source_results[source][:max_per_source])

        # Fill remaining slots with highest-scoring results
        remaining_slots = top_k - len(balanced)
        if remaining_slots > 0:
            # Collect remaining results from all sources
            remaining = []
            for source in required_sources:
                remaining.extend(source_results[source][max_per_source:])
            remaining.extend(other_results)

            # Sort by combined score and take top remaining
            remaining.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            balanced.extend(remaining[:remaining_slots])

        return balanced[:top_k]

    def _faiss_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform FAISS semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)

            if query_embedding.size == 0:
                return []

            # Normalize query embedding
            import faiss
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.faiss_index.search(query_embedding, top_k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.faiss_chunks):
                    results.append({
                        'content': self.faiss_chunks[idx],
                        'score': float(score),
                        'source': 'faiss',
                        'rank': i,
                        'metadata': self.faiss_metadata[idx] if idx < len(self.faiss_metadata) else {}
                    })

            return results

        except Exception as e:
            print(f"Error in FAISS search: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top results
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0:  # Only include positive scores
                    results.append({
                        'content': self.bm25_chunks[idx],
                        'score': float(scores[idx]),
                        'source': 'bm25',
                        'rank': i,
                        'metadata': self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
                    })

            return results

        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return []

    def _combine_results(self, faiss_results: List[Dict], bm25_results: List[Dict],
                        top_k: int) -> List[Dict[str, Any]]:
        """Combine FAISS and BM25 results with weighted scoring."""
        # Weight factors (can be tuned)
        faiss_weight = 0.6
        bm25_weight = 0.4

        # Normalize scores
        if faiss_results:
            max_faiss_score = max(r['score'] for r in faiss_results)
            for result in faiss_results:
                result['normalized_score'] = result['score'] / max_faiss_score if max_faiss_score > 0 else 0

        if bm25_results:
            max_bm25_score = max(r['score'] for r in bm25_results)
            for result in bm25_results:
                result['normalized_score'] = result['score'] / max_bm25_score if max_bm25_score > 0 else 0

        # Combine unique results
        seen_content = set()
        combined = []

        # Add FAISS results
        for result in faiss_results:
            content = result['content']
            if content not in seen_content:
                result['combined_score'] = result['normalized_score'] * faiss_weight
                combined.append(result)
                seen_content.add(content)

        # Add BM25 results (boost if already exists)
        for result in bm25_results:
            content = result['content']
            if content in seen_content:
                # Boost existing result
                for existing in combined:
                    if existing['content'] == content:
                        existing['combined_score'] += result['normalized_score'] * bm25_weight
                        break
            else:
                result['combined_score'] = result['normalized_score'] * bm25_weight
                combined.append(result)
                seen_content.add(content)

        # Sort by combined score and return top_k
        combined.sort(key=lambda x: x['combined_score'], reverse=True)

        return combined[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            'faiss_chunks': len(self.faiss_chunks),
            'bm25_chunks': len(self.bm25_chunks),
            'indexes_loaded': bool(self.faiss_index and self.bm25_index)
        }