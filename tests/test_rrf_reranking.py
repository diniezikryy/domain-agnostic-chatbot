import os
import json
import shutil
from pathlib import Path
from run_evaluation import run_pipeline


def test_rrf_reranking_pipeline_runs_with_stubbed_generation():
    # remove any previous cache to avoid test pollution
    shutil.rmtree("evaluation/cache", ignore_errors=True)

    dataset_path = Path("test_data/evaluation_dataset_minimal.json")
    with open(dataset_path, 'r') as f:
        questions = json.load(f)

    profiles = {}
    # load user profiles from dataset
    for q in questions:
        pid = q.get("user_profile_id")
        if pid and pid not in profiles:
            profile_path = Path("test_data") / f"{pid}.json"
            if profile_path.exists():
                with open(profile_path, 'r') as pf:
                    profiles[pid] = json.load(pf)

    # Run the pipeline: RRF + reranking, but stub generation to avoid LLM usage
    results = run_pipeline(questions, profiles, "rrf_reranking", "my_policies", cache_manager=None, stub_generation=True)

    # Expect a result per question and presence of rag_chunks
    assert len(results) == len(questions)
    for entry in results:
        # Ensure rag_chunks exist (they might be empty if nothing found)
        assert 'rag_chunks' in entry
        # Each may have rerank_info, but it's optional. If present, it should be a dict
        if entry.get('rerank_info') is not None:
            assert isinstance(entry['rerank_info'], dict)
