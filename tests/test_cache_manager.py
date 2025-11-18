import os
from pathlib import Path

import pytest

from utils.cache_manager import CacheManager


def test_cache_key_changes_when_env_changes(tmp_path, monkeypatch):
    # Set baseline env params
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "30")
    monkeypatch.setenv("RERANK_KEEP_TOP_N", "3")
    monkeypatch.setenv("RRF_FUSION_K", "20")
    monkeypatch.setenv("RERANK_HYDE_WEIGHT", "0.6")

    cache_dir = tmp_path / "cache"
    cm = CacheManager(str(cache_dir), compress=False)

    question = "Will this test use a different key?"
    batch_id = "my_policies"
    exp = "semantic_chunking"
    user_profile_id = "user_1"

    cm.put(question, batch_id, exp, {}, "answer1", user_profile_id)

    # Ensure we can fetch with same env
    fetched = cm.get(question, batch_id, exp, user_profile_id)
    assert fetched is not None

    # Change one env param and confirm cache key differs
    monkeypatch.setenv("RERANK_KEEP_TOP_N", "7")

    fetched_after = cm.get(question, batch_id, exp, user_profile_id)
    assert fetched_after is None

    # Set back and confirm restore
    monkeypatch.setenv("RERANK_KEEP_TOP_N", "3")
    fetched_again = cm.get(question, batch_id, exp, user_profile_id)
    assert fetched_again is not None


def test_cache_entry_includes_experiment_params(tmp_path, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "50")
    monkeypatch.setenv("RERANK_KEEP_TOP_N", "5")
    monkeypatch.setenv("RRF_FUSION_K", "40")
    monkeypatch.setenv("RERANK_HYDE_WEIGHT", "0.75")

    cache_dir = tmp_path / "cache2"
    cm = CacheManager(str(cache_dir), compress=False)

    question = "What is the test param include"
    batch_id = "my_policies"
    exp = "reranking"
    user_profile_id = None

    cm.put(question, batch_id, exp, {}, "answer2", user_profile_id)
    fetched = cm.get(question, batch_id, exp, user_profile_id)
    assert fetched is not None
    params = fetched.get("experiment_params")
    assert params is not None
    assert params.get("retrieval_candidate_pool") == "50"
    assert params.get("rerank_keep_top_n") == "5"
    assert params.get("rrf_fusion_k") == "40"
    assert params.get("rerank_hyde_weight") == "0.75"


def test_cache_handles_numpy_types(tmp_path):
    """
    Ensure CacheManager can safely store numpy types (float32, arrays) in retrieval_data.
    """
    try:
        import numpy as np
    except Exception:
        pytest.skip("numpy not available")

    cache_dir = tmp_path / "cache_numpy"
    cm = CacheManager(str(cache_dir), compress=False)

    question = "Does the cache accept numpy types?"
    batch_id = "my_policies"
    exp = "baseline"

    retrieval_data = {
        "scores": [np.float32(0.12345), np.float64(0.999)],
        "embedding": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }

    cm.put(question, batch_id, exp, retrieval_data, "ok")
    fetched = cm.get(question, batch_id, exp)
    assert fetched is not None
    # The retrieved data should not contain numpy types; it should be JSON-serializable
    assert isinstance(fetched.get("retrieval_data").get("scores")[0], float)
    assert isinstance(fetched.get("retrieval_data").get("embedding")[0], float)
