import os
import json
import shutil
import pytest
from pathlib import Path
from run_evaluation import run_pipeline
from utils.cache_manager import CacheManager

def test_cache_invalidation_with_different_env_configs():
    """
    Integration test to verify that changing experiment-defining environment variables
    results in different cache keys and prevents reuse of stale cache results.
    """
    # Clear any existing cache to ensure clean test
    shutil.rmtree("evaluation/cache", ignore_errors=True)

    # Use minimal test data
    test_data_path = Path("test_data/evaluation_dataset_minimal.json")
    profiles_path = Path("test_data")
    batch_id = "my_policies"  # Assuming this batch exists

    # Load data manually
    with open(test_data_path, 'r') as f:
        questions_data = json.load(f)
    profiles_data = {}
    for item in questions_data:
        profile_id = item.get("user_profile_id")
        if profile_id and profile_id not in profiles_data:
            profile_path = profiles_path / f"{profile_id}.json"
            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    profiles_data[profile_id] = json.load(f)

    # Create cache manager
    cache_manager = CacheManager()

    # First run with config 1
    os.environ["RETRIEVAL_CANDIDATE_POOL"] = "10"
    os.environ["RERANK_KEEP_TOP_N"] = "3"
    os.environ["RRF_FUSION_K"] = "60"
    os.environ["RERANK_HYDE_WEIGHT"] = "0.5"

    results1 = run_pipeline(questions_data, profiles_data, "baseline", batch_id, cache_manager=cache_manager, stub_generation=True)

    # Get cache keys after first run
    # The internal index stores entries in `cache_index['entries']`
    cache_keys_after_first = set(cache_manager.cache_index.get('entries', {}).keys())

    # Second run with different config
    os.environ["RETRIEVAL_CANDIDATE_POOL"] = "20"
    os.environ["RERANK_KEEP_TOP_N"] = "5"
    os.environ["RRF_FUSION_K"] = "70"
    os.environ["RERANK_HYDE_WEIGHT"] = "0.7"

    results2 = run_pipeline(questions_data, profiles_data, "baseline", batch_id, cache_manager=cache_manager, stub_generation=True)

    # Get cache keys after second run
    cache_keys_after_second = set(cache_manager.cache_index.get('entries', {}).keys())

    # Assert that new keys were added (since configs changed, cache should not reuse)
    new_keys = cache_keys_after_second - cache_keys_after_first
    assert len(new_keys) > 0, (
        f"No new cache keys generated, indicating possible cache reuse with different configs. "
        f"Keys after first: {len(cache_keys_after_first)}, after second: {len(cache_keys_after_second)}"
    )

    # Ensure AT LEAST ONE cached entry contains experiment_params that reflect the environment
    latest_env = os.getenv('RETRIEVAL_CANDIDATE_POOL')
    found_latest = False
    for k in cache_keys_after_second:
        params = cache_manager.cache_index['entries'][k].get('experiment_params', {})
        if params.get('retrieval_candidate_pool') == latest_env:
            found_latest = True
            break
    assert found_latest, "No cache entries found with the latest experiment params after the second run"

    # Clean up env vars
    for key in ["RETRIEVAL_CANDIDATE_POOL", "RERANK_KEEP_TOP_N", "RRF_FUSION_K", "RERANK_HYDE_WEIGHT"]:
        if key in os.environ:
            del os.environ[key]