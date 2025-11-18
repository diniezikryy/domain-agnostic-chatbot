import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from docx import Document
from dotenv import load_dotenv

# Try importing Azure extractor
try:
    from .pdf_extractor import extract_pages_with_azure
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

load_dotenv()

class FileHandler:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load Azure keys
        self.azure_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.azure_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    def process_document(
        self, file_path: Union[str, Path], chunking_strategy: str = "page"
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Hybrid process_document.
        Args:
            file_path: Path to the file.
            chunking_strategy: 
                - "page" (Default): Keeps 1 page = 1 chunk. Best for tables/forms.
                - "semantic": Splits page by headers. Best for dense text.
                - "fixed": Traditional sliding window.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return [], []

        page_texts = []
        extraction_method = "standard"

        # 1. Attempt Azure Extraction (From Ingestion Repo)
        if (file_path.suffix.lower() == ".pdf" and AZURE_AVAILABLE and 
            self.azure_endpoint and self.azure_key):
            
            print(f"  [INFO] Using Azure Document Intelligence for {file_path.name}")
            try:
                page_texts = extract_pages_with_azure(
                    str(file_path), self.azure_endpoint, self.azure_key
                )
                if page_texts:
                    extraction_method = "azure"
            except Exception as e:
                print(f"  [WARN] Azure extraction failed: {e}. Falling back to standard.")
                page_texts = []

        # 2. Fallback Extraction (From Evaluation Repo)
        if not page_texts:
            if file_path.suffix.lower() == ".pdf":
                page_texts = self._extract_pdf_text_pymupdf4llm(file_path)
            elif file_path.suffix.lower() == ".docx":
                page_texts = self._extract_docx_text(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                page_texts = self._extract_text_file(file_path)

        if not page_texts:
            return [], []

        # 3. Smart Metadata Regex (From Ingestion Repo)
        INSURER_PLAN_KEYWORDS = {
            "GREAT_SupremeHealth_Benefits.pdf": re.compile(r"\b(P PLUS|P PRIME|A PLUS|B PLUS|STANDARD|GREAT TotalCare)\b", re.IGNORECASE),
            "SINGLIFE_TRAVEL_POLICY.pdf": re.compile(r"\b(Prestige|Plus|Lite)\b", re.IGNORECASE),
            "Manulife_Policy_Illustration_REDACTED.pdf": re.compile(r"(Critical Care Enhancer|Total and Permanent Disability Plus Rider)", re.IGNORECASE),
        }
        plan_regex = INSURER_PLAN_KEYWORDS.get(file_path.name)

        all_chunks = []
        all_metadata = []

        for page_info in page_texts:
            page_num = page_info["page_num"]
            page_content = page_info["text"]

            # --- Metadata Extraction (From Ingestion Repo) ---
            # Extract Heading
            heading_match = re.search(r"^\s*(#{1,3})\s*(.+)$", page_content, re.MULTILINE)
            table_header_match = re.search(r"^\s*\|(.+)\|", page_content, re.MULTILINE)
            
            if heading_match:
                page_heading = heading_match.group(2).strip()
            elif table_header_match:
                page_heading = table_header_match.group(1).split("|")[0].strip()
            else:
                first_line = next((line for line in page_content.split("\n") if line.strip()), f"Page {page_num}")
                page_heading = first_line.strip()[:100]

            # Extract Plans
            found_plans = []
            if plan_regex:
                found_plans = list(set(m.strip() for m in plan_regex.findall(page_content)))

            # --- Chunking Strategy Switcher ---
            chunks = []
            actual_strategy = chunking_strategy

            if chunking_strategy == "page":
                # Repo 2 Style: Keep the whole page to preserve table layout
                chunks = [page_content]
            
            elif chunking_strategy == "semantic":
                # Repo 1 Style: Split by headers
                chunks, actual_strategy = self._create_semantic_chunks(page_content)
            
            else:
                # Fixed chunking fallback
                chunks, actual_strategy = self._create_chunks(page_content)

            # --- Assembly ---
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "page_number": page_num,
                    "year": self._extract_year_from_filename(file_path.name),
                    "chunk_id": f"p{page_num}-{i}",
                    "chunk_size": len(chunk),
                    "file_type": file_path.suffix.lower(),
                    "chunking_strategy": actual_strategy,
                    "extraction_method": extraction_method,
                    "page_heading": page_heading,      # Smart Metadata
                    "plan_context": found_plans        # Smart Metadata
                })

        return all_chunks, all_metadata

    # --- Helper Methods (From Evaluation Repo) ---
    def _create_semantic_chunks(self, text: str) -> Tuple[List[str], str]:
        # Prefer LangChain's MarkdownHeaderTextSplitter if available. It is robust to
        # preserving header structure and avoids breaking tables while respecting
        # chunk size and overlap configuration.
        try:
            from langchain.text_splitter import MarkdownHeaderTextSplitter

            # Configurable chunk sizes: try to keep the same safe defaults used
            # previously but allow LangChain to do the heavy lifting with better
            # heuristics for markdown structures.
            max_chars = 2000
            overlap = 200

            splitter = MarkdownHeaderTextSplitter(
                chunk_size=max_chars,
                chunk_overlap=overlap,
                # Keep headers in the split to preserve context
                keep_headers=True,
            )

            chunks = splitter.split_text(text)
            # LangChain returns a list of strings or TextChunk objects; normalize
            final = [c if isinstance(c, str) else getattr(c, "text", str(c)) for c in chunks if c and str(c).strip()]
            # Fallback to smaller chunks if none produced
            if not final:
                return self._create_chunks(text)
            # If a page contains a table-like structure, prefer to keep
            # the page intact to avoid breaking the logical table into
            # smaller chunks which can harm retrieval of numeric limits.
            if self._is_table_like(text):
                # For small tables, keep the entire page to preserve table context.
                if len(text) <= max_chars * 3:
                    return [text], "page"

                # For very large tables (risking embedding token limits) split
                # into smaller sub-chunks that still preserve the header.
                splitted = self._split_table_into_chunks(text, rows_per_chunk=8)
                if splitted:
                    return splitted, "page"

                # Fallback: create fixed-size chunks to ensure we don't send
                # an over-sized prompt to the embedding model.
                return self._create_chunks(text)
            return final, "semantic"

        except Exception:
            # If LangChain isn't available or fails for any reason, fall back
            # to our heuristic fallback that uses headers via regex. This keeps
            # backward-compatibility with existing batches if LangChain is not
            # installed in the runtime environment.
            MAX_CHUNK_CHARS = 2000  # Reduced to avoid token limit
            MIN_CHUNK_CHARS = 50
            has_headers = re.search(r'^#{1,6}\s+', text, re.MULTILINE)
            if not has_headers:
                # No headers, but check for table-like pages first. If a
                # table is present, prefer page chunking to preserve
                # table/row integrity; otherwise, use fixed chunking.
                if self._is_table_like(text):
                    return [text], "page"
                return self._create_chunks(text)

            chunks = []
            header_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
            sections = re.split(header_pattern, text)

            current_chunk = ""
            for section in sections:
                if not section.strip():
                    continue
                if header_pattern.match(section):
                    if current_chunk.strip() and len(current_chunk) > MIN_CHUNK_CHARS:
                        chunks.append(current_chunk.strip())
                    current_chunk = section + "\n"
                else:
                    # Check if adding this section would exceed MAX_CHUNK_CHARS
                    if len(current_chunk) + len(section) > MAX_CHUNK_CHARS and current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        current_chunk = section
                    else:
                        current_chunk += section

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Safety: split any oversized chunks
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > MAX_CHUNK_CHARS:
                    # Use fixed chunking for oversized chunks
                    sub_chunks, _ = self._create_chunks(chunk)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            # Table-aware semantic chunking: if a markdown-like table appears
            # on the page we prefer to return the whole page as a single
            # "page" chunk to preserve table context.
            if self._is_table_like(text):
                # Same logic as above for fallback path
                if len(text) <= MAX_CHUNK_CHARS * 3:
                    return [text], "page"
                splitted = self._split_table_into_chunks(text, rows_per_chunk=8)
                if splitted:
                    return splitted, "page"
                return self._create_chunks(text)

            return final_chunks, "semantic"

    def _is_table_like(self, text: str) -> bool:
        """Heuristic: detect a markdown-style table or pipe-delimited table.

        We look for multiple pipe characters forming rows and an optional
        separation row containing dashes. If detected, we prefer page-level
        chunking for that page to avoid breaking the table.
        """
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False

        # Count lines that appear to be pipe-delimited
        pipe_lines = [l for l in lines if '|' in l and len(l.strip()) > 3]
        if len(pipe_lines) >= 3:
            # If we see a header separator (| --- |) then it's likely a table
            for l in pipe_lines:
                if re.search(r'\|\s*-{3,}\s*\|', l):
                    return True
            # Otherwise, many pipe lines is a good signal of a table
            return True

        # Also look for a typical CSV-looking row with many commas but not
        # too many punctuation characters, which could also be a table.
        comma_lines = [l for l in lines if ',' in l and len(l.split(',')) >= 3]
        if len(comma_lines) >= 3:
            return True

        # Check for simple tabular English patterns like 'per day' + currency
        # pair repeated (S$ or $ repeated) which often indicates a table
        if re.search(r'(S\$|\$)\s*\d', text) and ('|' in text or ',' in text):
            return True

        return False

    def _split_table_into_chunks(self, text: str, rows_per_chunk: int = 10) -> List[str]:
        """Split a large table-like page into smaller chunks while preserving header.

        This preserves the header row for each sub-chunk so the LLM has the
        table structure and meaning for each partial table. This is used as a
        fallback when a page contains a very large table that would exceed
        embedding model token limits when kept as a single chunk.
        """
        lines = text.splitlines()
        # Find all lines that look like table rows
        pipe_lines = [l for l in lines if '|' in l and len(l.strip()) > 3]
        if not pipe_lines:
            return []

        # Identify the first pipe-line to use as header candidate
        header_idx = next((i for i, l in enumerate(lines) if '|' in l and len(l.strip()) > 3), None)
        if header_idx is None:
            return []

        # Optionally include the separator line if present
        separator_idx = header_idx + 1 if header_idx + 1 < len(lines) and re.search(r"\|\s*-{3,}", lines[header_idx + 1]) else None

        # Collect the table rows (we maintain the order in the original document)
        table_row_indices = [i for i, l in enumerate(lines) if '|' in l and len(l.strip()) > 3]
        table_rows = [lines[i] for i in table_row_indices]

        # Partition table rows into reasonably sized groups
        groups = [table_rows[i:i + rows_per_chunk] for i in range(0, len(table_rows), rows_per_chunk)]

        chunks = []
        # Include some minimal pre-table context to help the LLM (the few lines before header)
        preamble_window = max(0, header_idx - 3)
        preamble = '\n'.join(lines[preamble_window:header_idx]).strip()

        header_lines = []
        header_lines.append(lines[header_idx])
        if separator_idx is not None:
            header_lines.append(lines[separator_idx])

        header_text = '\n'.join(header_lines).strip()

        for group in groups:
            chunk_parts = []
            if preamble:
                chunk_parts.append(preamble)
            if header_text:
                chunk_parts.append(header_text)
            chunk_parts.extend(group)
            chunk_text = '\n'.join(chunk_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # If we couldn't split correctly, return empty to signal fallback
        return chunks if chunks else []

    def _create_chunks(self, text: str) -> Tuple[List[str], str]:
        if len(text) <= self.chunk_size:
            return [text], "fixed"
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
            if start + self.chunk_size > len(text) and start < len(text):
                chunks.append(text[start:])
                break
        return [c.strip() for c in chunks if c.strip()], "fixed"

    def _extract_pdf_text_pymupdf4llm(self, file_path: Path) -> List[Dict]:
        try:
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(str(file_path))
            if not md_text or not md_text.strip():
                print(f"  [WARN] pymupdf4llm returned empty text for {file_path.name}")
                return []
            pages = md_text.split("\n-----\n") 
            return [{"page_num": i+1, "text": self._clean_text(p)} for i, p in enumerate(pages) if p.strip()]
        except Exception as e:
            print(f"  [ERROR] pymupdf4llm extraction failed: {e}")
            return []

    def _extract_docx_text(self, file_path: Path) -> List[Dict]:
        try:
            doc = Document(str(file_path))
            text = "\n".join([p.text for p in doc.paragraphs])
            return [{"page_num": 1, "text": self._clean_text(text)}]
        except: return []

    def _extract_text_file(self, file_path: Path) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [{"page_num": 1, "text": self._clean_text(f.read())}]
        except: return []

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()

    def _extract_year_from_filename(self, filename: str) -> Optional[int]:
        match = re.search(r"(\d{4})", filename)
        return int(match.group(1)) if match else None
