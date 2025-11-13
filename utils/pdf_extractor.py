import fitz  # PyMuPDF
import sys
from pathlib import Path
import re
import os
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Any, Tuple

load_dotenv()


def _clean_text(text: str) -> str:
    """Simple text cleaner."""
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def table_to_markdown(table):
    """Convert Azure table object to markdown with better cell handling."""
    if table.row_count == 0 or table.column_count == 0:
        return ""
    grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
    for cell in table.cells:
        content = cell.content.strip() if cell.content else ""
        if cell.column_span and cell.column_span > 1:
            for col_offset in range(cell.column_span):
                if cell.column_index + col_offset < table.column_count:
                    grid[cell.row_index][cell.column_index + col_offset] = (
                        content if col_offset == 0 else ""
                    )
        else:
            grid[cell.row_index][cell.column_index] = content
    lines = []
    lines.append("| " + " | ".join(grid[0]) + " |")
    lines.append("| " + " | ".join(["---"] * table.column_count) + " |")
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def get_text_spans_from_table(table):
    """Get all text spans that are part of a table."""
    spans = []
    for cell in table.cells:
        if cell.spans:
            for span in cell.spans:
                spans.append((span.offset, span.offset + span.length))
    return spans


def is_text_in_table(text_span, table_spans):
    """Check if a text span overlaps with any table span."""
    text_start, text_end = text_span
    for table_start, table_end in table_spans:
        if not (text_end <= table_start or text_start >= table_end):
            return True
    return False


def split_pdf(pdf_path, chunk_size=2):
    """Split PDF into chunks to avoid page limits."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    chunks = []
    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
        pdf_bytes = new_doc.tobytes()
        chunks.append(
            {
                "bytes": pdf_bytes,
                "start_page": start_page + 1,
                "end_page": end_page,
            }
        )
        new_doc.close()
    doc.close()
    return chunks


def extract_pages_with_azure(
    file_path: str, endpoint: str, key: str
) -> List[Dict[str, Any]]:
    """
    Extracts PDF content using Azure Document Intelligence.
    This is the new "worker" function that returns structured page data.
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        print(f"Error: File not found at '{pdf_path}'")
        return []

    client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    doc = fitz.open(pdf_path)
    original_page_count = len(doc)
    doc.close()
    print(f"[INFO] PDF has {original_page_count} pages")

    # This list will hold our final output
    page_outputs: List[Dict[str, Any]] = []

    try:
        # Use chunking for Free Tier (2-page limit)
        print(f"[INFO] Splitting PDF into chunks (2 pages each) for Free tier...")
        chunks = split_pdf(pdf_path, chunk_size=2)
        print(f"[INFO] Created {len(chunks)} chunks")

        page_offset = 0

        for chunk_num, chunk in enumerate(chunks, 1):
            print(f"--- Processing Chunk {chunk_num}/{len(chunks)} ---")
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout", document=chunk["bytes"]
            )
            result = poller.result()

            all_table_spans = []
            for table in result.tables:
                all_table_spans.extend(get_text_spans_from_table(table))

            # Process each page *within* this chunk
            for page_num_in_chunk in range(len(result.pages)):
                actual_page_num = page_offset + page_num_in_chunk + 1
                page = result.pages[page_num_in_chunk]

                # 1. Get tables on this page
                page_tables_md = []
                tables_on_page = [
                    table
                    for table in result.tables
                    if any(
                        region.page_number == page_num_in_chunk + 1
                        for cell in table.cells
                        for region in cell.bounding_regions
                    )
                ]
                for table in tables_on_page:
                    page_tables_md.append(table_to_markdown(table))

                # 2. Get non-table text on this page
                page_text_content = []
                paragraphs_on_page = [
                    para
                    for para in result.paragraphs
                    if para.bounding_regions[0].page_number == page_num_in_chunk + 1
                ]
                non_table_paragraphs = []
                for para in paragraphs_on_page:
                    if para.spans:
                        para_span = (
                            para.spans[0].offset,
                            para.spans[0].offset + para.spans[0].length,
                        )
                        if not is_text_in_table(para_span, all_table_spans):
                            non_table_paragraphs.append(para)
                page_text_content = [para.content for para in non_table_paragraphs]

                # 3. Combine tables and text into ONE string for this page
                full_page_content = (
                    "\n\n".join(page_tables_md)
                    + "\n\n**Additional text (footnotes, headers, etc.):**\n"
                    + _clean_text("\n".join(page_text_content))
                )

                # 4. Append to our output list
                page_outputs.append(
                    {"page_num": actual_page_num, "text": full_page_content}
                )

            page_offset += len(result.pages)

        print(f"[INFO] Successfully extracted {len(page_outputs)} pages.")
        return page_outputs

    except Exception as e:
        print(f"An error occurred during Azure extraction: {e}")
        import traceback

        traceback.print_exc()
        return []


if __name__ == "__main__":
    # This test block now calls the new function
    AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not AZURE_ENDPOINT or not AZURE_KEY:
        print("Error: Azure credentials not found in .env file")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Error: You must provide a file path as an argument.")
        print("Example: python pdf_extractor.py /path/to/your/file.pdf")
    else:
        # Call the new function
        pages_data = extract_pages_with_azure(sys.argv[1], AZURE_ENDPOINT, AZURE_KEY)

        if pages_data:
            print(f"\n--- SUCCESS: Extracted {len(pages_data)} pages ---")

            # Print page 2 as an example
            for page in pages_data:
                if page["page_num"] == 2:
                    print("\n--- Example Output for Page 2 ---")
                    print(page["text"])
