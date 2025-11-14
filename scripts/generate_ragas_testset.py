import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import OpenAI

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import HeadlineSplitter, Parallel, default_transforms


def load_chunk_documents(batch_id: str) -> List[Document]:
    """Load chunk texts + metadata from the FAISS index pickle."""
    faiss_pickle = Path("batches") / batch_id / "faiss_index" / "index.pkl"
    if not faiss_pickle.exists():
        raise FileNotFoundError(f"FAISS pickle not found for batch '{batch_id}': {faiss_pickle}")

    with open(faiss_pickle, "rb") as f:
        data = pickle.load(f)

    chunks: List[str] = data.get("chunks", [])
    metadata: List[Dict[str, Any]] = data.get("metadata", [])

    documents: List[Document] = []
    for idx, chunk in enumerate(chunks):
        chunk_meta = metadata[idx] if idx < len(metadata) else {}
        documents.append(Document(page_content=chunk, metadata=chunk_meta))

    if not documents:
        raise ValueError(f"No chunks were loaded from {faiss_pickle}")

    return documents


def build_testset_generator() -> Tuple[TestsetGenerator, Any]:
    """Instantiate a Ragas TestsetGenerator using OpenAI models."""
    generator_model = os.getenv("TESTSET_GENERATOR_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("TESTSET_EMBED_MODEL", "text-embedding-3-small")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chat_llm = ChatOpenAI(model=generator_model, temperature=0.2)
    generator_llm = LangchainLLMWrapper(chat_llm)
    embeddings = OpenAIEmbeddings(client=openai_client, model=embed_model)

    generator = TestsetGenerator(
        llm=generator_llm,  # type: ignore[arg-type]
        embedding_model=embeddings,  # type: ignore[arg-type]
    )
    return generator, generator_llm


def coerce_sample_attr(sample: Any, attr: str, default: Any = None) -> Any:
    """Return attribute regardless of EvalSample/dict structure."""
    if hasattr(sample, attr):
        return getattr(sample, attr)
    if isinstance(sample, dict):
        return sample.get(attr, default)
    return default


def convert_eval_dataset(dataset, start_idx: int = 1) -> List[Dict[str, Any]]:
    """Convert the ragas evaluation dataset to our JSON format."""
    records: List[Dict[str, Any]] = []
    for offset, sample in enumerate(dataset, start=start_idx):
        question = coerce_sample_attr(sample, "user_input", "")
        reference = coerce_sample_attr(sample, "reference", "")
        contexts = coerce_sample_attr(sample, "reference_contexts", []) or []

        records.append(
            {
                "question_id": f"AUTO_{offset:03d}",
                "question": question,
                "ground_truth": reference,
                "reference_contexts": contexts,
                "user_profile_id": "auto_generated",
                "test_scenario": "ragas_auto_generated",
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG test questions using ragas.testset.generator")
    parser.add_argument("--batch_id", default="my_policies", help="Batch ID whose chunks will seed the generator")
    parser.add_argument("--size", type=int, default=10, help="Number of questions to generate")
    parser.add_argument(
        "--output",
        default="test_data/evaluation_dataset_auto_ragas.json",
        help="Path to write the generated dataset",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set before generating a testset")

    documents = load_chunk_documents(args.batch_id)
    generator, generator_llm = build_testset_generator()

    query_distribution = default_query_distribution(generator_llm)

    transforms = default_transforms(
        documents=list(documents),
        llm=generator_llm,
        embedding_model=generator.embedding_model,
    )

    def _has_headlines(node):
        try:
            return node.get_property("headlines") is not None
        except AttributeError:
            return False

    def _iter_transforms(node_transforms):
        if isinstance(node_transforms, list):
            for item in node_transforms:
                yield from _iter_transforms(item)
        elif isinstance(node_transforms, Parallel):
            for item in node_transforms.transformations:
                yield from _iter_transforms(item)
        else:
            yield node_transforms

    for transform in _iter_transforms(transforms):
        if isinstance(transform, HeadlineSplitter):
            transform.filter_nodes = _has_headlines

    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=args.size,
        transforms=transforms,
        query_distribution=query_distribution,
    )

    eval_dataset = dataset.to_evaluation_dataset()  # type: ignore[attr-defined]
    records = convert_eval_dataset(eval_dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated {len(records)} auto test questions -> {output_path}")


if __name__ == "__main__":
    main()
