#!/usr/bin/env python3
"""
Agent CFO Baseline - Uses existing hybrid search infrastructure
Place financial documents in: documents/cfo_financials/
"""

import sys
import os
from pathlib import Path
import time

# Import your existing components
from document_processor import DocumentProcessor
from batch_manager import BatchManager
from query_processor import QueryProcessor


def main():
    company_name = "Apple"
    batch_id = "cfo_financials"

    print(f"=== Agent CFO Baseline for {company_name} ===\n")

    # Step 1: Check for documents
    docs_dir = Path(f"documents/{batch_id}")
    if not docs_dir.exists():
        print(f"Creating directory: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please add financial PDFs to {docs_dir}/ then run again.")
        return

    # Find all PDFs
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}/")
        print("Add your financial documents (10-Ks, annual reports, etc.) and try again.")
        return

    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Step 2: Process documents (or skip if already done)
    batch_manager = BatchManager()
    existing_batch = batch_manager.get_batch_info(batch_id)

    if existing_batch:
        print(f"\n✓ Batch '{batch_id}' already exists")
        rebuild = input("Rebuild? (y/n): ").lower() == 'y'
        if not rebuild:
            print("Using existing batch...")
        else:
            print("\nRebuilding batch...")
            doc_processor = DocumentProcessor()
            start = time.time()
            doc_processor.create_batch(
                batch_id=batch_id,
                document_paths=[str(p) for p in pdf_files],
                batch_name=f"{company_name} Financial Documents"
            )
            print(f"Ingestion time: {time.time() - start:.2f}s")
    else:
        print("\nCreating new batch...")
        doc_processor = DocumentProcessor()
        start = time.time()
        success = doc_processor.create_batch(
            batch_id=batch_id,
            document_paths=[str(p) for p in pdf_files],
            batch_name=f"{company_name} Financial Documents"
        )
        if not success:
            print("Failed to create batch")
            return
        print(f"Ingestion time: {time.time() - start:.2f}s")

    # Step 3: Run benchmark queries
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK QUERIES")
    print("=" * 60)

    queries = [
        "Report the Gross Margin over the last 5 quarters, with values.",
        "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.",
        "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years."
    ]

    query_processor = QueryProcessor(batch_manager)

    # Access the search engine to get raw results for debugging
    from utils.search import HybridSearchEngine
    search_engine = HybridSearchEngine()
    paths = batch_manager.get_batch_paths(batch_id)
    search_engine.load_indexes(
        faiss_path=paths["faiss_index"],
        bm25_path=paths["bm25_index"]
    )

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {query}")
        print('─' * 60)

        start = time.time()
        response = query_processor.process_query(query, batch_id)
        query_time = time.time() - start

        # DEBUG: Show metadata from search results
        search_results = search_engine.hybrid_search(query, top_k=5)
        print(f"\n[DEBUG] Top 3 results with metadata:")
        for j, result in enumerate(search_results[:3], 1):
            metadata = result.get('metadata', {})
            print(f"  Result {j}:")
            print(f"    File: {metadata.get('filename', 'N/A')}")
            print(f"    Page: {metadata.get('page_number', 'N/A')}")
            print(f"    Year: {metadata.get('year', 'N/A')}")
            print(f"    Content: {result.get('content', '')[:80]}...")

        print(f"\nAnswer:\n{response}")
        print(f"\nTime: {query_time:.2f}s")

        results.append({
            'query': query,
            'answer': response,
            'time': query_time
        })

    # Step 4: Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average query time: {avg_time:.2f}s")
    print(f"Total documents: {len(pdf_files)}")

    # Try interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Ask questions about the financial documents (or 'quit' to exit)")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        start = time.time()
        answer = query_processor.process_query(question, batch_id)
        query_time = time.time() - start

        print(f"\nAnswer: {answer}")
        print(f"Time: {query_time:.2f}s")


if __name__ == "__main__":
    main()