"""
File Handlers
Handles processing of different document formats (PDF, DOCX, TXT, MD).
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re

class FileHandler:
    """Handles processing of various document formats."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
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
                text = self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() == '.docx':
                text = self._extract_docx_text(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                text = self._extract_text_file(file_path)
            else:
                print(f"Unsupported file format: {file_path.suffix}")
                return [], []

            if not text.strip():
                print(f"No text extracted from {file_path.name}")
                return [], []

            # Create chunks
            chunks = self._create_chunks(text)

            # Create metadata for each chunk
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "file_type": file_path.suffix.lower()
                })

            return chunks, metadata

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return [], []

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2

            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            return self._clean_text(text)

        except ImportError:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"Error reading PDF {file_path.name}: {e}")
            return ""

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document

            doc = Document(file_path)
            text = ""

            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            return self._clean_text(text)

        except ImportError:
            print("python-docx not installed. Install with: pip install python-docx")
            return ""
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            return ""

    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from TXT or MD file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            return self._clean_text(text)

        except Exception as e:
            print(f"Error reading text file {file_path.name}: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)

        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                sentence_end = -1

                for punct in ['. ', '! ', '? ']:
                    pos = text.rfind(punct, search_start, end)
                    if pos > sentence_end:
                        sentence_end = pos + 1

                if sentence_end > start:
                    end = sentence_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)

            if start >= len(text):
                break

        return chunks