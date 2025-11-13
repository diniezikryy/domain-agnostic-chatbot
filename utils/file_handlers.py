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
        MAX_CHUNK_CHARS = 6000 
        MIN_CHUNK_CHARS = 50
        
        has_headers = re.search(r'^#{1,6}\s+', text, re.MULTILINE)
        if not has_headers:
            return [text], "fixed"

        chunks = []
        header_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
        sections = re.split(header_pattern, text)
        
        current_chunk = ""
        for section in sections:
            if not section.strip(): continue
            if header_pattern.match(section):
                if current_chunk.strip() and len(current_chunk) > MIN_CHUNK_CHARS:
                    chunks.append(current_chunk.strip())
                current_chunk = section + "\n"
            else:
                current_chunk += section
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks, "semantic"

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
            pages = md_text.split("\n-----\n") 
            return [{"page_num": i+1, "text": self._clean_text(p)} for i, p in enumerate(pages)]
        except: return []

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
