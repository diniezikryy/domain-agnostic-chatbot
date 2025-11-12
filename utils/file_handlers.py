"""
File Handlers
Handles processing of different document formats (PDF, DOCX, TXT, MD).
Uses pymupdf4llm for robust PDF text extraction with better table handling.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
from docx import Document


class FileHandler:
    """Handles processing of various document formats."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(
        self, file_path: str, chunking_strategy: str = "fixed"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a document and return chunks with metadata.
        
        Args:
            file_path: Path to document file
            chunking_strategy: "fixed" (default) or "semantic" (header-aware)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return [], []

        # Extract text based on file type
        try:
            if file_path.suffix.lower() == ".pdf":
                page_texts = self._extract_pdf_text_pymupdf4llm(file_path)
            elif file_path.suffix.lower() == ".docx":
                page_texts = self._extract_docx_text(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
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
                page_num = page_info["page_num"]
                page_content = page_info["text"]

                # Create chunks for this page's content with specified strategy
                chunks, actual_strategy = self._create_chunks(page_content, strategy=chunking_strategy)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadata.append(
                        {
                            "source": str(file_path),
                            "filename": file_path.name,
                            "page_number": page_num,
                            "year": self._extract_year_from_filename(file_path.name),
                            "chunk_id": f"p{page_num}-{i}",
                            "chunk_size": len(chunk),
                            "file_type": file_path.suffix.lower(),
                            "chunking_strategy": chunking_strategy,
                            "chunking_strategy_used": actual_strategy,
                        }
                    )

            return all_chunks, all_metadata

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return [], []

    def _extract_pdf_text_pymupdf4llm(self, file_path: Path) -> List[Dict]:
        """Extract text from PDF using pymupdf4llm (optimized for LLMs)."""
        page_texts = []
        try:
            import pymupdf4llm

            # Extract markdown-formatted text (handles tables well)
            md_text = pymupdf4llm.to_markdown(str(file_path))

            # Split by pages (pymupdf4llm includes page markers)
            pages = md_text.split("\n-----\n")  # Default page separator

            for page_num, page_content in enumerate(pages, 1):
                if page_content.strip():  # Skip empty pages
                    cleaned_text = self._clean_text(page_content)
                    page_texts.append({"page_num": page_num, "text": cleaned_text})

            # If no page separators found, treat as single page
            if len(pages) == 1:
                cleaned_text = self._clean_text(md_text)
                page_texts = [{"page_num": 1, "text": cleaned_text}]

            return page_texts

        except Exception as e:
            print(f"pymupdf4llm failed for {file_path.name}: {e}")
            # Fallback to basic PyMuPDF
            return self._extract_pdf_text_fallback(file_path)

    def _extract_pdf_text_fallback(self, file_path: Path) -> List[Dict]:
        """Fallback PDF extraction using basic PyMuPDF."""
        page_texts = []
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Try to extract tables
                try:
                    tables = page.find_tables()
                    table_text = ""
                    for table in tables:
                        df = table.to_pandas()
                        table_text += "\n\n" + df.to_string(index=False)
                    text += table_text
                except:
                    pass

                page_texts.append(
                    {"page_num": page_num + 1, "text": self._clean_text(text)}
                )

            doc.close()
            return page_texts

        except Exception as e:
            print(f"Fallback PDF extraction failed for {file_path.name}: {e}")
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
            page_texts.append({"page_num": 1, "text": self._clean_text(full_text)})
            return page_texts
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            return []

    def _extract_text_file(self, file_path: Path) -> List[Dict]:
        """Extract text from TXT or MD file."""
        page_texts = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Treat as one page
            page_texts.append({"page_num": 1, "text": self._clean_text(text)})
            return page_texts
        except Exception as e:
            print(f"Error reading text file {file_path.name}: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive newlines but keep structure
        text = re.sub(
            r"\n\s*\n\s*\n+", "\n\n", text
        )  # Multiple newlines -> double newline
        text = re.sub(r" +", " ", text)  # Multiple spaces -> single space
        text = re.sub(r"\t+", " ", text)  # Tabs -> spaces

        # Clean up common markdown artifacts from pymupdf4llm
        text = re.sub(r"\*\*\s*\*\*", "", text)  # Empty bold markers
        text = re.sub(r"_{2,}", "", text)  # Multiple underscores

        return text.strip()

    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from filename."""
        match = re.search(r"(\d{4})", filename)  # Look for 4-digit year
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:  # Reasonable year range
                return year

        # Fallback: look for 2-digit year
        match = re.search(r"(\d{2})", filename)
        if match:
            year = int(match.group(1))
            if year < 100:
                year += 2000  # Assume 20xx
                return year
        return None

    def _create_chunks(self, text: str, strategy: str = "fixed") -> Tuple[List[str], str]:
        """Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            strategy: "fixed" (default) or "semantic" (header-aware / table-preserving)
        """
        if strategy == "semantic":
            chunks, actual_strategy = self._create_semantic_chunks(text)
            return chunks, actual_strategy
        
        # Fixed-size chunking (original behavior)
        if len(text) <= self.chunk_size:
            return [text], "fixed"

        chunks = []
        start = 0

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

        return [c.strip() for c in chunks if c.strip()], "fixed"

    def _create_semantic_chunks(self, text: str) -> Tuple[List[str], str]:
        """Create semantic chunks by preserving headers and table boundaries.
        
        This is a simple header-based chunker that:
        - Splits on markdown headers (##, ###, etc.) when present
        - Preserves table blocks (lines starting with | or containing table delimiters)
        - Falls back to fixed-size chunking for non-markdown content
        - Hard limit: Max 6000 chars per chunk (~ 1500 tokens, safe for 8192 token models)
        """
        MAX_CHUNK_CHARS = 6000  # ~1500 tokens (safe for OpenAI's 8192 token limit)
        MIN_CHUNK_CHARS = 50
        
        # Check if the text appears to be markdown (contains headers or table markers)
        has_headers = re.search(r'^#{1,6}\s+', text, re.MULTILINE)
        has_tables = '|' in text and re.search(r'\|.*\|.*\|', text)

        if not has_headers and not has_tables:
            # No semantic structure found — fall back to fixed chunking
            print(f"  [WARN] No semantic structure detected in text (len={len(text)}). Falling back to FIXED chunking.")
            return self._create_chunks(text, strategy="fixed")

        chunks = []
        
        # Split by headers (## or ###) while preserving header lines
        header_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
        sections = re.split(header_pattern, text)
        
        # Sections will be: [text_before_first_header, header1, content1, header2, content2, ...]
        # Combine each header with its content
        current_chunk = ""
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # If section is a header line
            if header_pattern.match(section):
                # Start a new chunk with this header
                if current_chunk.strip() and len(current_chunk) > MIN_CHUNK_CHARS:
                    # Force-split if over max
                    if len(current_chunk) > MAX_CHUNK_CHARS:
                        chunks.extend(self._force_split_large_chunk(current_chunk, MAX_CHUNK_CHARS))
                    else:
                        chunks.append(current_chunk.strip())
                current_chunk = section + "\n"
            else:
                # Content section — append to current chunk
                current_chunk += section
                
                # Hard limit enforcement
                if len(current_chunk) > MAX_CHUNK_CHARS:
                    # Try to split on paragraph boundaries
                    paragraphs = current_chunk.split('\n\n')
                    if len(paragraphs) > 1:
                        # Take paragraphs until we hit the limit
                        temp_chunk = ""
                        for para in paragraphs:
                            if len(temp_chunk) + len(para) + 2 < MAX_CHUNK_CHARS:
                                temp_chunk += para + "\n\n"
                            else:
                                if temp_chunk.strip():
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = para + "\n\n"
                        current_chunk = temp_chunk
                    else:
                        # No paragraph boundaries — force split
                        chunks.extend(self._force_split_large_chunk(current_chunk, MAX_CHUNK_CHARS))
                        current_chunk = ""
        
        # Add final chunk (with force-split if needed)
        if current_chunk.strip():
            if len(current_chunk) > MAX_CHUNK_CHARS:
                chunks.extend(self._force_split_large_chunk(current_chunk, MAX_CHUNK_CHARS))
            else:
                chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        return [c for c in chunks if len(c.strip()) > MIN_CHUNK_CHARS], "semantic"

    def _force_split_large_chunk(self, text: str, max_size: int) -> List[str]:
        """Force split a chunk that exceeds max_size by breaking on newlines."""
        chunks = []
        lines = text.split('\n')
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 < max_size:
                current += line + "\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = line + "\n"
        if current.strip():
            chunks.append(current.strip())
        return chunks
