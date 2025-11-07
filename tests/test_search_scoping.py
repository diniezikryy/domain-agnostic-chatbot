import os
from utils.search import HybridSearchEngine


def test_hybrid_search_scopes_to_allowed_docs():
    # Skip test if OPENAI_API_KEY not set to avoid accidental network calls during CI/local runs
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY not set in environment; skipping test_hybrid_search_scopes_to_allowed_docs.')
        return

    engine = HybridSearchEngine()
    faiss_dir = os.path.join('batches', 'my_policies', 'faiss_index')
    bm25_file = os.path.join('batches', 'my_policies', 'bm25_index.pkl')

    assert engine.load_indexes(faiss_dir, bm25_file), 'Failed to load indexes for testing'

    allowed = {'GREAT_SupremeHealth_Benefits.pdf'}
    results = engine.hybrid_search(query='riders premium coverage', top_k=20, allowed_doc_ids=allowed)

    # All returned filenames should be in the allowed set
    filenames = [r.get('metadata', {}).get('filename') for r in results]
    assert filenames, 'No results returned from scoped hybrid_search'
    assert all(f in allowed for f in filenames), f'Found filenames outside allowed set: {set(filenames) - allowed}'


def test_bm25_search_zeroes_disallowed_scores():
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY not set in environment; skipping test_bm25_search_zeroes_disallowed_scores.')
        return
    engine = HybridSearchEngine()
    faiss_dir = os.path.join('batches', 'my_policies', 'faiss_index')
    bm25_file = os.path.join('batches', 'my_policies', 'bm25_index.pkl')

    assert engine.load_indexes(faiss_dir, bm25_file), 'Failed to load indexes for testing'

    # Query tokens likely present in many chunks
    allowed = {'GREAT_SupremeHealth_Benefits.pdf'}
    bm25_results = engine._bm25_search(query='riders premium coverage', top_k=50, allowed_doc_ids=allowed)

    # Ensure bm25 results only contain allowed filenames
    filenames = [r.get('metadata', {}).get('filename') for r in bm25_results]
    assert filenames, 'No results returned from scoped BM25 search'
    assert all(f in allowed for f in filenames), f'BM25 returned disallowed filenames: {set(filenames) - allowed}'
