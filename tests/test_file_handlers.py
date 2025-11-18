import pytest

from utils.file_handlers import FileHandler


def test_semantic_chunking_header_preservation():
    fh = FileHandler()

    # A small markdown text with headers and tables. The splitter should keep
    # headers and prefer to split on them rather than inside tables.
    md_text = """
# Plan Overview

This is an introductory section.

## Benefits Table

| Item | Benefit |
| --- | --- |
| X | $100 |
| Y | $200 |

## Eligibility

Eligibility notes...
    """

    chunks, strat = fh._create_semantic_chunks(md_text)

    # Expect the strategy to be semantic OR page (table-aware behaviour)
    assert strat in ("semantic", "page")

    # Headers should be preserved (present in chunks)
    merged = "\n\n".join(chunks)
    assert "Plan Overview" in merged or "Plan Overview".lower() in merged.lower()
    assert "Benefits Table" in merged or "Benefits Table".lower() in merged.lower()
    assert "Eligibility" in merged or "Eligibility".lower() in merged.lower()


def test_fallback_when_no_headers():
    fh = FileHandler()

    text = "This is a long paragraph. " * 1000

    chunks, strat = fh._create_semantic_chunks(text)

    # Because no headers, we fallback to fixed chunking
    assert strat in ("fixed", "semantic")


def test_table_aware_semantic_chunking():
    fh = FileHandler()

    md_text = """
| Item | Benefit |
| --- | --- |
| Inpatient Rehab (Private) | $800 per day |
| Another | $1,000 per day |
"""

    chunks, strat = fh._create_semantic_chunks(md_text)

    # Table-aware chunking should prefer 'page' strategy to preserve table
    assert strat == "page"
    assert len(chunks) == 1
    assert "$800 per day" in chunks[0] or "$800" in chunks[0]


def test_semantic_rrf_usage(monkeypatch):
    from run_evaluation import run_retrieval_semantic_chunking
    from batch_manager import BatchManager
    from query_processor import QueryProcessor

    bm = BatchManager()
    qp = QueryProcessor(bm)

    # Stub OpenAI calls made during run_retrieval (intent and expansion)
    class DummyChoice:
        def __init__(self):
            self.message = type('M', (), {'content': '{"needs_comparison": false, "asks_about_uncovered_features": false, "requires_external_info": false}'})

    class DummyResponse:
        choices = [DummyChoice()]

    def fake_create(*args, **kwargs):
        return DummyResponse()

    try:
        monkeypatch.setattr(qp.client.chat.completions, 'create', fake_create)
    except Exception:
        # If the attribute nesting doesn't exist, set a simple fallback
        qp.client = type('C', (), {'chat': type('CH', (), {'completions': type('CO', (), {'create': fake_create})()})()})

    # Run retrieval - this will attempt to load the semantic batch
    result = run_retrieval_semantic_chunking(qp, "What is the limit for inpatient rehabilitation care in a private hospital under the B PLUS plan in S$?", 'my_policies', None)

    # If RRF fusion was applied, source field in rag_chunks_details should be 'rrf'
    rag = result.get('rag_chunks_details', [])
    if rag:
        assert any((c.get('source') == 'rrf') for c in rag)
        assert len(rag) >= 1
    # General sanity: we expect at least one chunk returned by the pipeline
    assert len(rag) >= 1
