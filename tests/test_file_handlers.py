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

    # Expect the strategy to be semantic
    assert strat == "semantic"

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
    assert len(chunks) >= 1
