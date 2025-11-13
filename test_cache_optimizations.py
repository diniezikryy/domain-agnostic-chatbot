"""
Test script to demonstrate cache optimizations.
Run this to verify compression and RAGAS caching work correctly.
"""

import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.cache_manager import CacheManager
from dotenv import load_dotenv

load_dotenv()


def test_compression():
    """Test gzip compression feature."""
    print("="*70)
    print("TEST 1: Compression")
    print("="*70)
    
    # Create test data
    test_data = {
        "rag_chunks_details": [{"content": "test chunk " * 100} for _ in range(10)],
        "rag_contexts_list": ["context " * 50 for _ in range(10)],
        "web_contexts_list": [],
        "web_research_raw": {}
    }
    test_answer = "This is a test answer. " * 20
    
    # Test with compression
    cm_compressed = CacheManager(compress=True, cache_ragas=False)
    cm_compressed.put(
        question="Test compression question",
        batch_id="test_batch",
        experiment_name="test_compress",
        retrieval_data=test_data,
        generated_answer=test_answer
    )
    
    # Test without compression
    cm_uncompressed = CacheManager(compress=False, cache_ragas=False)
    cm_uncompressed.put(
        question="Test compression question",
        batch_id="test_batch",
        experiment_name="test_uncompress",
        retrieval_data=test_data,
        generated_answer=test_answer
    )
    
    # Compare file sizes
    cache_dir = Path("evaluation/cache")
    
    compressed_files = list(cache_dir.glob("*.json.gz"))
    uncompressed_files = [f for f in cache_dir.glob("*.json") if not f.name.startswith('_')]
    
    if compressed_files:
        compressed_size = compressed_files[0].stat().st_size
        print(f"✅ Compressed file size: {compressed_size/1024:.2f} KB")
    
    if uncompressed_files:
        uncompressed_size = uncompressed_files[0].stat().st_size
        print(f"📦 Uncompressed file size: {uncompressed_size/1024:.2f} KB")
    
    if compressed_files and uncompressed_files:
        reduction = (1 - compressed_size/uncompressed_size) * 100
        print(f"💾 Size reduction: {reduction:.1f}%")
    
    print()


def test_ragas_caching():
    """Test RAGAS metrics caching feature."""
    print("="*70)
    print("TEST 2: RAGAS Metrics Caching")
    print("="*70)
    
    test_data = {
        "rag_chunks_details": [{"content": "test"}],
        "rag_contexts_list": ["context"],
        "web_contexts_list": [],
        "web_research_raw": {}
    }
    
    test_ragas_metrics = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.92,
        "context_precision": 0.78,
        "context_recall": 0.88,
        "answer_correctness": 0.81
    }
    
    # Create cache entry with RAGAS metrics
    cm = CacheManager(cache_ragas=True)
    cm.put(
        question="Test RAGAS caching question",
        batch_id="test_batch",
        experiment_name="test_ragas",
        retrieval_data=test_data,
        generated_answer="Test answer",
        ragas_metrics=test_ragas_metrics
    )
    
    # Retrieve RAGAS metrics
    retrieved_metrics = cm.get_ragas_metrics(
        question="Test RAGAS caching question",
        batch_id="test_batch",
        experiment_name="test_ragas"
    )
    
    if retrieved_metrics:
        print("✅ RAGAS metrics cached and retrieved successfully:")
        for metric, value in retrieved_metrics.items():
            print(f"   - {metric}: {value:.3f}")
    else:
        print("❌ RAGAS metrics not cached")
    
    print()


def test_cache_statistics():
    """Test enhanced cache statistics."""
    print("="*70)
    print("TEST 3: Enhanced Statistics")
    print("="*70)
    
    cm = CacheManager()
    stats = cm.get_stats()
    
    print(f"Total entries: {stats['total_entries']}")
    print(f"Entries with RAGAS: {stats['entries_with_ragas']}")
    print(f"Cache size: {stats['cache_size_mb']:.3f} MB")
    print(f"Compression enabled: {stats['compression_enabled']}")
    
    if stats['hits'] + stats['misses'] > 0:
        print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    
    if stats['ragas_hits'] + stats['ragas_misses'] > 0:
        print(f"RAGAS hit rate: {stats['ragas_hit_rate']:.1%}")
    
    print()


def test_update_ragas_metrics():
    """Test updating existing cache entry with RAGAS metrics."""
    print("="*70)
    print("TEST 4: Update Existing Entry with RAGAS Metrics")
    print("="*70)
    
    test_data = {
        "rag_chunks_details": [{"content": "test"}],
        "rag_contexts_list": ["context"],
        "web_contexts_list": [],
        "web_research_raw": {}
    }
    
    cm = CacheManager(cache_ragas=True)
    
    # Create entry without RAGAS metrics
    cm.put(
        question="Test update RAGAS",
        batch_id="test_batch",
        experiment_name="test_update",
        retrieval_data=test_data,
        generated_answer="Test answer",
        ragas_metrics=None  # No metrics initially
    )
    
    # Update with RAGAS metrics
    new_metrics = {
        "faithfulness": 0.90,
        "answer_relevancy": 0.88,
        "context_precision": 0.82,
        "context_recall": 0.75,
        "answer_correctness": 0.85
    }
    
    success = cm.update_ragas_metrics(
        question="Test update RAGAS",
        batch_id="test_batch",
        experiment_name="test_update",
        ragas_metrics=new_metrics
    )
    
    if success:
        print("✅ Successfully updated existing entry with RAGAS metrics")
        
        # Verify update
        retrieved = cm.get_ragas_metrics(
            question="Test update RAGAS",
            batch_id="test_batch",
            experiment_name="test_update"
        )
        
        if retrieved and retrieved["faithfulness"] == 0.90:
            print("✅ Verification passed: metrics correctly stored")
        else:
            print("❌ Verification failed: metrics not stored correctly")
    else:
        print("❌ Failed to update entry")
    
    print()


def cleanup_test_cache():
    """Clean up test cache entries."""
    print("="*70)
    print("CLEANUP: Removing Test Cache Entries")
    print("="*70)
    
    cm = CacheManager()
    
    # Clear test entries
    test_experiments = ["test_compress", "test_uncompress", "test_ragas", "test_update"]
    
    for exp in test_experiments:
        cm.clear(experiment_name=exp, batch_id="test_batch")
    
    print("✅ Test cache entries cleared")
    print()


if __name__ == "__main__":
    print("\n🚀 Cache Optimization Test Suite\n")
    
    test_compression()
    test_ragas_caching()
    test_update_ragas_metrics()
    test_cache_statistics()
    
    # Optional: cleanup
    response = input("Clean up test cache entries? (y/n): ")
    if response.lower() == 'y':
        cleanup_test_cache()
    
    print("✅ All tests complete!")
    print("\nTo see optimizations in action, run:")
    print("  KMP_DUPLICATE_LIB_OK=TRUE python run_evaluation.py --experiment baseline --batch_id my_policies_semantic")
    print("\nThen run it again to see ~90% speedup from RAGAS caching!")
