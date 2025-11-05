"""
File Handlers
Handles processing of different document formats (PDF, DOCX, TXT, MD).
Uses pdfplumber for advanced table extraction from PDFs.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
import pdfplumber
import pandas as pd
from docx import Document

class FileHandler:
    """Handles processing of various document formats."""

    # Maybe can try to increase the chunk size to 2000 to prevent tables from splitting.
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a document and return chunks with metadata."""
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return [], []

        # Extract text based on file type
        try:
            if file_path.suffix.lower() == '.pdf':
                # This function is now much more powerful
                page_texts = self._extract_pdf_text_and_tables(file_path)
            elif file_path.suffix.lower() == '.docx':
                page_texts = self._extract_docx_text(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                page_texts = self._extract_text_file(file_path)
            else:
                print(f"Unsupported file format: {file_path.suffix}")
                return [], []

            if not page_texts:
                print(f"No text extracted from {file_path.name}")
                return [], []

            # Create chunks and metadata
            all_chunks = []
            all_metadata = []

            for page_info in page_texts:
                page_num = page_info['page_num']
                page_content = page_info['text']

                # Create chunks for this page's content
                chunks = self._create_chunks(page_content)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": str(file_path),
                        "filename": file_path.name,
                        "page_number": page_num, # Now accurate per-chunk
                        "year": self._extract_year_from_filename(file_path.name),
                        "chunk_id": f"p{page_num}-{i}",
                        "chunk_size": len(chunk),
                        "file_type": file_path.suffix.lower()
                    })

            return all_chunks, all_metadata

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return [], []

    def _tables_to_markdown(self, tables: List[List[List[str]]]) -> str:
        """Converts tables extracted by pdfplumber into Markdown strings."""
        markdown_tables = []
        for table in tables:
            if not table:
                continue

            # Convert list of lists to pandas DataFrame
            # Use first row as header
            header = table[0]
            data = table[1:]

            # Clean header (replace None with empty string)
            header = [str(h) if h is not None else '' for h in header]

            try:
                df = pd.DataFrame(data, columns=header)
                # Convert DataFrame to Markdown, index=False drops the row numbers
                markdown_tables.append(df.to_markdown(index=False))
            except Exception as e:
                print(f"Warning: Could not convert table to markdown: {e}")

        return "\n\n".join(markdown_tables)

    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from filename."""
        import re
        match = re.search(r'(\d{2,4})', filename)
        if match:
            year = int(match.group(1))
            if year < 100:  # Two digit year
                year += 2000
            return year
        return None

    def _extract_pdf_text_and_tables(self, file_path: Path) -> List[Dict]:
        """Extract text and tables from PDF file using pdfplumber."""
        page_texts = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract plain text
                    text = page.extract_text() or ""

                    # Extract tables and convert to Markdown
                    tables = page.extract_tables()
                    markdown_tables = self._tables_to_markdown(tables)

                    # Combine text and tables for this page
                    full_page_content = f"{text}\n\n{markdown_tables}"

                    page_texts.append({
                        'page_num': page.page_number,
                        'text': self._clean_text(full_page_content)
                    })
            return page_texts
        except Exception as e:
            print(f"Error reading PDF {file_path.name}: {e}")
            return []

    def _extract_docx_text(self, file_path: Path) -> List[Dict]:
        """Extract text from DOCX file."""
        page_texts = []
        try:
            doc = Document(file_path)
            full_text = ""
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"

            # DOCX has no concept of pages, so we treat it as one page
            page_texts.append({
                'page_num': 1,
                'text': self._clean_text(full_text)
            })
            return page_texts
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            return []

    def _extract_text_file(self, file_path: Path) -> List[Dict]:
        """Extract text from TXT or MD file."""
        page_texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Treat as one page
            page_texts.append({
                'page_num': 1,
                'text': self._clean_text(text)
            })
            return page_texts
        except Exception as e:
            print(f"Error reading text file {file_path.name}: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive newlines but keep single newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        # Use simple sliding window. More advanced logic could be added here.
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            move = self.chunk_size - self.chunk_overlap
            start += move

            # Ensure last chunk captures the end
            if start + self.chunk_size > len(text) and start < len(text):
                chunks.append(text[start:])
                break

        return [c.strip() for c in chunks if c.strip()]