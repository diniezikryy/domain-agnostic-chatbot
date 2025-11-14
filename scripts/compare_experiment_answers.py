"""Compare pipeline answers between two RAGAS experiment artifacts."""

import argparse
import json
import textwrap
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict


def load_experiment(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pipeline = {}
    for entry in data.get("pipeline_results", []):
        question_id = entry.get("question_id")
        if not question_id:
            continue
        pipeline[question_id] = entry
    local_metrics = {m.get("question_id"): m for m in data.get("local_metrics", []) if m.get("question_id")}
    metadata = data.get("metadata", {})
    return {"pipeline": pipeline, "local_metrics": local_metrics, "metadata": metadata}


def safe_snippet(text: str, width: int = 240) -> str:
    if not text:
        return "(empty)"
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= width:
        return cleaned
    return f"{cleaned[:width]}..."


def compare_answers(baseline: Dict[str, Dict], reranking: Dict[str, Dict]) -> None:
    shared_questions = sorted(set(baseline["pipeline"]) & set(reranking["pipeline"]))
    if not shared_questions:
        print("No overlapping questions found between the two artifacts.")
        return

    header = "Comparison: baseline vs reranking"
    print(f"\n{header}\n{'=' * len(header)}")

    for question_id in shared_questions:
        base_entry = baseline["pipeline"][question_id]
        rerank_entry = reranking["pipeline"][question_id]
        base_answer = base_entry.get("answer", "")
        rerank_answer = rerank_entry.get("answer", "")
        contexts_base = base_entry.get("contexts") or base_entry.get("rag_contexts", [])
        contexts_rerank = rerank_entry.get("contexts") or rerank_entry.get("rag_contexts", [])
        rag_chunks_base = base_entry.get("rag_chunks", [])
        rag_chunks_rerank = rerank_entry.get("rag_chunks", [])

        print(f"\n--- Question: {question_id} ---")
        question_text = base_entry.get('question') or rerank_entry.get('question') or "(unknown question)"
        print(f"Question text: {safe_snippet(question_text)}")
        print(f"Baseline contexts: {len(contexts_base)} (rag chunks: {len(rag_chunks_base)})")
        print(f"Reranking contexts: {len(contexts_rerank)} (rag chunks: {len(rag_chunks_rerank)})")

        base_recall = baseline["local_metrics"].get(question_id, {}).get("context_token_recall")
        rerank_recall = reranking["local_metrics"].get(question_id, {}).get("context_token_recall")
        if base_recall is not None or rerank_recall is not None:
            print(f"Context token recall — baseline: {base_recall}, reranking: {rerank_recall}")

        identical = base_answer.strip() == rerank_answer.strip()
        print(f"Answers identical? {'Yes' if identical else 'No'}")
        print("Baseline answer:")
        print(textwrap.indent(textwrap.fill(safe_snippet(base_answer), width=80), "    "))
        print("Reranking answer:")
        print(textwrap.indent(textwrap.fill(safe_snippet(rerank_answer), width=80), "    "))

        if not identical:
            print("Diff preview:")
            matcher = SequenceMatcher(None, base_answer, rerank_answer)
            changes = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                a_snip = base_answer[i1:i2].replace("\n", " ")
                b_snip = rerank_answer[j1:j2].replace("\n", " ")
                changes.append(
                    f"  {tag}: baseline='{safe_snippet(a_snip, 60)}' | rerank='{safe_snippet(b_snip, 60)}'"
                )
            if changes:
                print("\n".join(changes[:5]))
            else:
                print("  Answers differ only by whitespace or punctuation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs reranking RAGAS artifacts.")
    parser.add_argument("baseline", type=Path, help="Path to baseline JSON artifact")
    parser.add_argument("reranking", type=Path, help="Path to reranking JSON artifact")
    args = parser.parse_args()

    baseline = load_experiment(args.baseline)
    reranking = load_experiment(args.reranking)
    compare_answers(baseline, reranking)


if __name__ == "__main__":
    main()