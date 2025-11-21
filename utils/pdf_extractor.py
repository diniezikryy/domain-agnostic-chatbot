# utils/pdf_extractor.py

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    _AZURE_AVAILABLE = True
except Exception:
    _AZURE_AVAILABLE = False

load_dotenv()


def split_pdf(pdf_path: str, chunk_size: int = 2) -> List[Dict[str, Any]]:
    """
    Split a PDF into smaller byte chunks, each containing up to `chunk_size` pages.

    Returns a list of dicts:
        [{ "bytes": <pdf_bytes>, "first_page": int }, ...]
    where first_page is 1-based index of the first page in this chunk in the original PDF.
    """
    path = Path(pdf_path)
    doc = fitz.open(path)
    chunks: List[Dict[str, Any]] = []

    total_pages = doc.page_count
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        sub_doc = fitz.open()
        for page_index in range(start, end):
            sub_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        pdf_bytes = sub_doc.tobytes()
        chunks.append({"bytes": pdf_bytes, "first_page": start + 1})

    return chunks


def table_to_markdown(table) -> str:
    """
    Convert an Azure DocumentTable into a GitHub-flavoured markdown table string.
    """
    if not getattr(table, "cells", None):
        return ""

    max_row = max(cell.row_index for cell in table.cells) + 1
    max_col = max(cell.column_index for cell in table.cells) + 1

    grid: List[List[str]] = [["" for _ in range(max_col)] for _ in range(max_row)]

    for cell in table.cells:
        text = cell.content or ""
        grid[cell.row_index][cell.column_index] = text.replace("\n", " ").strip()

    lines: List[str] = []

    # Header
    header = "|" + "|".join(col or " " for col in grid[0]) + "|"
    lines.append(header)
    # Separator
    separator = "|" + "|".join("---" for _ in grid[0]) + "|"
    lines.append(separator)

    # Rows
    for row in grid[1:]:
        line = "|" + "|".join(col or " " for col in row) + "|"
        lines.append(line)

    return "\n".join(lines)


def extract_pages_with_azure(
    pdf_path: str,
    endpoint: str,
    key: str,
) -> List[Dict[str, Any]]:
    """
    Use Azure Document Intelligence (prebuilt-layout) to extract structured text from a PDF.

    Returns:
        [
            { "page_num": int, "text": str },
            ...
        ]
    """
    if not endpoint or not key:
        raise ValueError("Azure endpoint and key must be provided")

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    pdf_path = str(pdf_path)
    chunks = split_pdf(pdf_path, chunk_size=2)

    page_outputs: List[Dict[str, Any]] = []
    page_offset = 0

    for chunk_num, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Processing chunk {chunk_num}/{len(chunks)}")

        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=chunk["bytes"],
        )
        result = poller.result()

        # Build a map from local page index -> text segments
        local_pages: Dict[int, Dict[str, Any]] = {}

        # Process tables first
        for table in getattr(result, "tables", []) or []:
            # Determine the page for this table using its bounding region
            page_index = 0
            if getattr(table, "bounding_regions", None):
                # Azure pages are 1-based
                page_index = table.bounding_regions[0].page_number - 1
            markdown = table_to_markdown(table)
            if not markdown:
                continue

            page_entry = local_pages.setdefault(
                page_index, {"tables": [], "paragraphs": []}
            )
            page_entry["tables"].append(markdown)

        # Process paragraphs
        for para in getattr(result, "paragraphs", []) or []:
            content = (para.content or "").strip()
            if not content:
                continue
            page_index = 0
            if getattr(para, "bounding_regions", None):
                page_index = para.bounding_regions[0].page_number - 1

            page_entry = local_pages.setdefault(
                page_index, {"tables": [], "paragraphs": []}
            )
            page_entry["paragraphs"].append(content)

        # Assemble text per page for this chunk
        for local_page_index, payload in sorted(
            local_pages.items(), key=lambda x: x[0]
        ):
            global_page_num = page_offset + local_page_index + 1

            tables_md = "\n\n".join(payload["tables"])
            paragraphs = payload["paragraphs"]

            # ---------- NEW: de-duplicate paragraphs that just repeat table content ----------
            deduped_paragraphs: List[str] = []
            if paragraphs:
                # Normalise table text into a single lowercased string without markdown noise
                import re as _re

                tables_plain = tables_md.replace("|", " ")
                tables_plain = _re.sub(r"\s+", " ", tables_plain).strip().lower()

                for p in paragraphs:
                    p_norm = _re.sub(r"\s+", " ", p).strip().lower()
                    # Heuristic:
                    # - if the paragraph is short AND fully contained in table text -> likely a repeat
                    # - otherwise keep it
                    if len(p_norm) <= 80 and p_norm and p_norm in tables_plain:
                        # skip duplicated table line
                        continue
                    deduped_paragraphs.append(p)

            # Build "other text" from deduped paragraphs only
            other_text = "\n".join(deduped_paragraphs) if deduped_paragraphs else ""

            parts: List[str] = []
            if tables_md:
                parts.append(tables_md)
            if other_text:
                if tables_md:
                    parts.append(
                        "\n\n**Additional text (footnotes, headers, etc.):**\n"
                    )
                parts.append(other_text)

            full_text = "\n".join(parts).strip()

            page_outputs.append(
                {
                    "page_num": global_page_num,
                    "text": full_text,
                }
            )

        page_offset += len(getattr(result, "pages", []) or [])

    # Sort outputs by page_num just in case
    page_outputs.sort(key=lambda x: x["page_num"])
    return page_outputs


def extract_pages_local(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Local, free PDF extraction fallback using PyMuPDF (fitz) and pdfplumber for table extraction.

    Returns a list of {'page_num': int, 'text': str} objects.
    """
    pages_out: List[Dict[str, Any]] = []
    path = Path(pdf_path)

    # Prefer pdfplumber for table extraction if available, fallback to fitz text otherwise
    use_pdfplumber = False
    try:
        import pdfplumber

        use_pdfplumber = True
    except Exception:
        use_pdfplumber = False

    if use_pdfplumber:
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    parts: List[str] = []

                    # Table extraction (if present) - convert to simple markdown table
                    tables = page.extract_tables()
                    table_texts = []
                    for t in tables:
                        # Convert rows to pipe-delimited markdown table
                        if not t:
                            continue
                        header = t[0]
                        rows = t[1:] if len(t) > 1 else []
                        md_lines = []
                        md_lines.append("|" + "|".join([str(x or "").strip() for x in header]) + "|")
                        md_lines.append("|" + "|".join(["---" for _ in header]) + "|")
                        for r in rows:
                            md_lines.append("|" + "|".join([str(x or "").strip() for x in r]) + "|")
                        table_md = "\n".join(md_lines)
                        table_texts.append(table_md)

                    if table_texts:
                        parts.append("\n\n".join(table_texts))

                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""

                    if text:
                        parts.append(text)

                    page_text = "\n\n".join(parts).strip()
                    pages_out.append({"page_num": i, "text": page_text})

            return pages_out
        except Exception:
            # Fall through to fitz-based extraction if pdfplumber fails
            pass

    # Fallback: use PyMuPDF directly
    try:
        import fitz

        doc = fitz.open(str(path))
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages_out.append({"page_num": i + 1, "text": text.strip()})
        doc.close()
    except Exception:
        # If everything fails, return empty list
        return []

    return pages_out


if __name__ == "__main__":
    import sys

    AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py /path/to/file.pdf")
        raise SystemExit(1)

    pdf_file = sys.argv[1]
    pages = extract_pages_with_azure(pdf_file, AZURE_ENDPOINT, AZURE_KEY)

    print(f"Extracted {len(pages)} pages")
    # Print page 2 as an example if it exists
    for page in pages:
        if page["page_num"] == 2:
            print("\n--- Example Output for Page 2 ---")
            print(page["text"])
