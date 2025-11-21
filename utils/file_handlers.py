# utils/file_handlers.py

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
import json

from docx import Document
from dotenv import load_dotenv

from .pdf_extractor import (
    extract_pages_with_azure,
    extract_pages_local,
    _AZURE_AVAILABLE,
)
from config.optimization_settings import optimization_settings

load_dotenv()


class FileHandler:
    """Handles processing of various document formats."""

    def __init__(self, **kwargs):
        # Load Azure Document Intelligence credentials
        self.azure_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.azure_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        # Use Azure Document Intelligence if credentials are present and SDK is available.
        # Otherwise fall back to a local PyMuPDF/pdfplumber extractor for free re-indexing.
        self.use_azure = bool(self.azure_endpoint and self.azure_key and _AZURE_AVAILABLE)
        if not self.use_azure:
            print(
                "[INFO] Azure Document Intelligence not enabled or unavailable. Using local PDF extraction fallback (no paid credits required)."
            )

        # Optionally load OpenAI key here, or inside the LLM method
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_document(
        self,
        file_path: str,
        enrich_with_llm: bool = False,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a document and return:
        - list of chunk texts
        - list of corresponding metadata dicts

        If enrich_with_llm is True, each chunk's metadata is augmented with:
        - llm_summary
        - llm_key_entities
        - llm_likely_questions
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return [], []

        try:
            # 1) Extract page-level text for the document
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                if self.use_azure:
                    page_texts = extract_pages_with_azure(
                        str(file_path), self.azure_endpoint, self.azure_key
                    )
                else:
                    # Use a local, free extractor so users can reindex without Azure.
                    page_texts = extract_pages_local(str(file_path))
            elif suffix == ".docx":
                page_texts = self._extract_docx_text(file_path)
            elif suffix in {".txt", ".md"}:
                page_texts = self._extract_text_file(file_path)
            else:
                print(f"Unsupported file format: {suffix}")
                return [], []

            if not page_texts:
                print(f"No text extracted from {file_path.name}")
                return [], []

            # 2) Insurance-aware plan detection based on filename
            INSURER_PLAN_KEYWORDS: Dict[str, re.Pattern] = {
                "GREAT_SupremeHealth_Benefits.pdf": re.compile(
                    r"\b(P PLUS|P PRIME|A PLUS|B PLUS|STANDARD|GREAT TotalCare)\b",
                    re.IGNORECASE,
                ),
                "SINGLIFE_TRAVEL_POLICY.pdf": re.compile(
                    r"\b(Prestige|Plus|Lite)\b", re.IGNORECASE
                ),
                "Manulife_Policy_Illustration_REDACTED.pdf": re.compile(
                    r"(Critical Care Enhancer|Total and Permanent Disability Plus Rider)",
                    re.IGNORECASE,
                ),
            }

            filename = file_path.name
            plan_regex = INSURER_PLAN_KEYWORDS.get(filename)

            all_chunks: List[str] = []
            all_metadata: List[Dict[str, Any]] = []

            # 3) Iterate over pages and split into sub-chunks
            # Build a full document text used for parent document context
            document_text = "\n\n".join(p["text"] for p in page_texts)
            for page_info in page_texts:
                page_num = page_info["page_num"]
                page_content = page_info["text"]

                if not page_content or not page_content.strip():
                    continue

                # Derive a page heading from markdown header, table header, or first line
                heading_match = re.search(
                    r"^\s*(#{1,3})\s*(.+)$", page_content, re.MULTILINE
                )
                table_header_match = re.search(
                    r"^\s*\|(.+)\|", page_content, re.MULTILINE
                )

                if heading_match:
                    page_heading = heading_match.group(2).strip()
                elif table_header_match:
                    page_heading = table_header_match.group(1).split("|")[0].strip()
                else:
                    first_line = next(
                        (line for line in page_content.splitlines() if line.strip()),
                        f"Page {page_num}",
                    )
                    page_heading = first_line.strip()

                # Detect any plan names on this page
                found_plans: List[str] = []
                if plan_regex:
                    raw_matches = plan_regex.findall(page_content)
                    if raw_matches:
                        # findall may return tuples if there are groups in the regex
                        if isinstance(raw_matches[0], tuple):
                            for t in raw_matches:
                                found_plans.append(" ".join(x for x in t if x))
                        else:
                            found_plans.extend(raw_matches)
                    # de-duplicate
                    found_plans = list(dict.fromkeys(found_plans))

                normalized_plans = self._normalize_plans(found_plans)

                # Split this page into smaller semantic chunks
                sub_chunks = self._split_page_into_chunks(page_content, page_num)

                # Pre-compute section-style headings for parent section mapping
                page_headings = self._extract_section_headers(page_content)

                # Pre-compute document-level summary (do once per file) for the optimized pipeline
                doc_summary = None
                if enrich_with_llm and optimization_settings.ENABLE_OPTIMIZED_PIPELINE:
                    try:
                        doc_summary = self._generate_document_summary_llm(document_text)
                    except Exception as e:
                        print(f"[WARN] Document summary generation failed: {e}")

                for sub in sub_chunks:
                    chunk_text = sub["chunk_text"]
                    chunk_index = sub["chunk_index"]

                    if not chunk_text.strip():
                        continue

                    chunk_id = f"p{page_num}-{chunk_index}"

                    metadata: Dict[str, Any] = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "page_number": page_num,
                        "page_heading": page_heading,
                        "plan_context": normalized_plans,
                        "chunk_id": chunk_id,
                        "chunk_size": len(chunk_text),
                        "file_type": suffix,
                    }

                    # Add parent document and parent section metadata
                    # Add parent document metadata only for optimized pipeline
                    if optimization_settings.ENABLE_OPTIMIZED_PIPELINE:
                        # parent_document_id is simply the absolute path - unique for each file
                        metadata["parent_document_id"] = str(file_path)
                        # Small document preview (first 3k chars) to keep metadata compact
                        metadata["parent_document_text"] = (
                            document_text[:3000] if len(document_text) > 3000 else document_text
                        )

                    # map chunk to nearest page heading (if any)
                    if optimization_settings.ENABLE_OPTIMIZED_PIPELINE and sub.get("start_idx") is not None:
                        section = self._get_parent_section_for_offset(page_headings, sub["start_idx"]) or {}
                        metadata["parent_section_heading"] = section.get("heading") or page_heading
                        metadata["parent_section_id"] = (
                            f"{file_path.name}-p{page_num}-s{section.get('index')}"
                            if section
                            else f"{file_path.name}-p{page_num}-s0"
                        )
                        metadata["parent_section_text"] = (
                            self._get_section_text(page_content, section)
                            if section
                            else page_content[:1500]
                        )

                    # Optional LLM enrichment
                    if enrich_with_llm:
                        try:
                            extra = self._generate_chunk_metadata_llm(chunk_text)
                            if extra:
                                metadata.update(extra)
                            # Add precomputed per-document summary to each chunk's metadata
                            if doc_summary:
                                metadata["document_summary"] = doc_summary
                        except Exception as e:
                            # Don't blow up ingestion if enrichment fails
                            print(f"[WARN] LLM enrichment failed for {chunk_id}: {e}")

                    all_chunks.append(chunk_text)
                    all_metadata.append(metadata)

            return all_chunks, all_metadata

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            import traceback

            traceback.print_exc()
            return [], []

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _extract_docx_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from a DOCX file as a single 'page'."""
        page_texts: List[Dict[str, Any]] = []
        try:
            doc = Document(file_path)
            full_text_lines: List[str] = []
            for paragraph in doc.paragraphs:
                full_text_lines.append(paragraph.text)
            full_text = "\n".join(full_text_lines)
            page_texts.append({"page_num": 1, "text": self._clean_text(full_text)})
            return page_texts
        except Exception as e:
            print(f"Error reading DOCX {file_path.name}: {e}")
            return []

    def _extract_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from a TXT or MD file as a single 'page'."""
        page_texts: List[Dict[str, Any]] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            page_texts.append({"page_num": 1, "text": self._clean_text(text)})
            return page_texts
        except Exception as e:
            print(f"Error reading text file {file_path.name}: {e}")
            return []

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Collapse excessive blank lines
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Normalize spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)
        # Remove repeated markdown emphasis artifacts
        text = re.sub(r"\*\*\s*\*\*", "", text)
        text = re.sub(r"_{2,}", "", text)
        # Strip common email signatures (for non-PDF sources that might be emails)
        text = self._remove_email_signature(text)
        return text.strip()

    def _remove_email_signature(self, text: str) -> str:
        """Remove simple email signature closings if present."""
        signature_regex = re.compile(
            r"(?m)(?:\n|\A)\s*"
            r"(?:Best regards,|Best Regards,|Regards,|Sincerely,|Thank you,|Thanks,)"
            r"\s*(?:\n.*){0,3}$",
            flags=re.IGNORECASE,
        )
        new_text = signature_regex.sub("\n", text)
        # Remove placeholder names
        new_text = re.sub(r"(?m)^\s*\[?Your Name\]?\s*$", "\n", new_text)
        return new_text.rstrip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def _split_page_into_chunks(
        self,
        page_content: str,
        page_num: int,
        max_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Split a page into multiple smaller chunks based on paragraphs
        and a sliding window over characters.

        Returns a list of dicts:
            [{"chunk_text": str, "chunk_index": int}, ...]
        """
        # Normalize line breaks
        text = page_content.replace("\r\n", "\n").replace("\r", "\n")

        # Split into paragraphs using double newlines as separators
        raw_paragraphs = [p for p in text.split("\n\n")]
        # Preserve paragraph spans so we can compute chunk offsets for parent section lookup
        paragraph_spans: List[Dict[str, Any]] = []
        ptr = 0
        for para in raw_paragraphs:
            if not para or not para.strip():
                ptr += len(para) + 2  # account for the split separator
                continue
            # Find the next occurrence of this paragraph after the pointer
            start_idx = text.find(para, ptr)
            if start_idx == -1:
                start_idx = ptr
            end_idx = start_idx + len(para)
            paragraph_spans.append({"text": para.strip(), "start": start_idx, "end": end_idx})
            ptr = end_idx

        # If no paragraphs, treat whole page as one chunk
        if not paragraph_spans:
            return [{"chunk_text": page_content.strip(), "chunk_index": 0, "start_idx": 0, "end_idx": len(page_content)}]

        chunks: List[Dict[str, Any]] = []
        current = ""
        chunk_index = 0
        current_paragraphs: List[Dict[str, Any]] = []

        for para in paragraph_spans:
            # Always ensure paragraphs are separated
            candidate = (current + "\n\n" + para["text"]).strip() if current else para["text"]

            if len(candidate) <= max_chars:
                # initialize start index of this chunk if not set
                # (we track paragraph list instead of a separate variable)
                current = candidate
                current_paragraphs.append(para)
                continue

            # If adding this paragraph would exceed max_chars, flush current chunk
            if current:
                # Determine start and end indices for the current chunk from paragraphs
                start_idx = current_paragraphs[0]["start"] if current_paragraphs else 0
                end_idx = current_paragraphs[-1]["end"] if current_paragraphs else 0
                chunk_obj = {
                    "chunk_text": current.strip(),
                    "chunk_index": chunk_index,
                }
                # Only attach start/end indices if optimized pipeline enabled
                if optimization_settings.ENABLE_OPTIMIZED_PIPELINE:
                    chunk_obj["start_idx"] = start_idx
                    chunk_obj["end_idx"] = end_idx

                chunks.append(chunk_obj)
                chunk_index += 1

                # Start new chunk with some overlap from the end of the previous chunk
                if len(current) > overlap_chars:
                    overlap = current[-overlap_chars:]
                    current = (overlap + "\n\n" + para["text"]).strip()
                else:
                    current = para["text"]
                current_paragraphs = [para]
            else:
                # Single paragraph longer than max_chars: hard-cut into pieces
                remaining_text = para["text"]
                while len(remaining_text) > max_chars:
                    piece = remaining_text[:max_chars]
                    chunk_obj = {
                        "chunk_text": piece.strip(),
                        "chunk_index": chunk_index,
                    }
                    # Only add indices for optimized pipeline
                    if optimization_settings.ENABLE_OPTIMIZED_PIPELINE:
                        chunk_obj["start_idx"] = 0
                        chunk_obj["end_idx"] = 0

                    chunks.append(chunk_obj)
                    chunk_index += 1
                    remaining_text = remaining_text[max_chars:]
                current = remaining_text
                current_paragraphs = [para]

        # Flush any remaining text as the last chunk
        if current and current.strip():
            start_idx = current_paragraphs[0]["start"] if current_paragraphs else 0
            end_idx = current_paragraphs[-1]["end"] if current_paragraphs else 0
            chunk_obj = {
                "chunk_text": current.strip(),
                "chunk_index": chunk_index,
            }
            if optimization_settings.ENABLE_OPTIMIZED_PIPELINE:
                chunk_obj["start_idx"] = start_idx
                chunk_obj["end_idx"] = end_idx

            chunks.append(chunk_obj)

        if not chunks:
            base = {"chunk_text": page_content.strip(), "chunk_index": 0}
            if optimization_settings.ENABLE_OPTIMIZED_PIPELINE:
                base["start_idx"] = 0
                base["end_idx"] = len(page_content)
            return [base]

        return chunks

    # ------------------------------------------------------------------
    # Plan normalization
    # ------------------------------------------------------------------
    def _normalize_plans(self, found_plans: List[str]) -> List[str]:
        """Normalise raw plan strings into canonical labels."""
        PLAN_NORMALIZATION_RULES = [
            (re.compile(r"\bp plus\b", re.IGNORECASE), "P PLUS"),
            (re.compile(r"\bp prime\b", re.IGNORECASE), "P PRIME"),
            (re.compile(r"\ba plus\b", re.IGNORECASE), "A PLUS"),
            (re.compile(r"\bb plus\b", re.IGNORECASE), "B PLUS"),
            (re.compile(r"\bgreat totalcare\b", re.IGNORECASE), "GREAT TotalCare"),
            (re.compile(r"\bprestige\b", re.IGNORECASE), "Prestige"),
            (re.compile(r"\blite\b", re.IGNORECASE), "Lite"),
            (re.compile(r"\bstandard\b", re.IGNORECASE), "STANDARD"),
            (re.compile(r"critical care", re.IGNORECASE), "Critical Care Enhancer"),
            (
                re.compile(r"total and permanent disability", re.IGNORECASE),
                "Total and Permanent Disability Plus Rider",
            ),
        ]

        normalised: List[str] = []

        for raw in found_plans:
            text = raw.strip()
            if not text:
                continue

            lowered = text.lower()
            matched = False
            for pattern, canonical in PLAN_NORMALIZATION_RULES:
                if pattern.search(lowered):
                    normalised.append(canonical)
                    matched = True
                    break

            if not matched:
                normalised.append(text)

        # De-duplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for p in normalised:
            if p not in seen:
                seen.add(p)
                deduped.append(p)

        return deduped

    # ------------------------------------------------------------------
    # Section extraction helpers
    # ------------------------------------------------------------------
    def _extract_section_headers(self, page_content: str) -> List[Dict[str, Any]]:
        """Extract markdown-like headings and return a list with positions.

        Returns a list of dicts: [{"index": int, "heading": str, "start": int, "end": int}]
        """
        headings = []
        for idx, m in enumerate(re.finditer(r"^\s*(#{1,3})\s*(.+)$", page_content, re.MULTILINE)):
            headings.append({"index": idx + 1, "heading": m.group(2).strip(), "start": m.start(), "end": None})

        # If none found, return empty list; we'll fallback to page heading
        if not headings:
            return []

        # compute end offsets
        for i in range(len(headings)):
            if i < len(headings) - 1:
                headings[i]["end"] = headings[i + 1]["start"]
            else:
                headings[i]["end"] = len(page_content)

        return headings

    def _get_parent_section_for_offset(self, headings: List[Dict[str, Any]], offset: int) -> Dict[str, Any]:
        """Return the last heading whose start is <= offset, or None."""
        if not headings:
            return None

        candidate = None
        for h in headings:
            if h["start"] <= offset:
                candidate = h
            else:
                break

        return candidate

    def _get_section_text(self, page_content: str, section: Dict[str, Any]) -> str:
        if not section:
            return ""
        s = section.get("start", 0)
        e = section.get("end", len(page_content))
        text = page_content[s:e].strip()
        return text if len(text) <= 3000 else text[:3000]

    # ------------------------------------------------------------------
    # LLM enrichment
    # ------------------------------------------------------------------
    def _generate_chunk_metadata_llm(self, chunk_text: str) -> Dict[str, Any]:
        """
        Call a lighter LLM to get:
        - llm_summary: str
        - llm_key_entities: List[str]
        - llm_likely_questions: List[str]

        Returns a dict suitable for merging into the chunk metadata.
        """
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        # Local import so this module can be used without OpenAI installed (if needed)
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)

        system_prompt = (
            "You are helping index an insurance policy document for a RAG system. "
            "Given a chunk of text, extract a concise summary, key entities, and "
            "1-3 natural-language questions this chunk can help answer. "
            "Respond as valid JSON with keys: summary, key_entities, likely_questions. "
            "key_entities should be a list of short strings. "
            "likely_questions should be a list of short user-style questions."
        )

        user_prompt = f"Chunk:\n\n{chunk_text}"

        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # choose a small, cheap model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM metadata JSON: {e}")

        summary = (data.get("summary") or "").strip()
        key_entities = data.get("key_entities") or []
        likely_questions = data.get("likely_questions") or []

        # Normalise list fields
        if isinstance(key_entities, str):
            key_entities = [key_entities]
        if isinstance(likely_questions, str):
            likely_questions = [likely_questions]

        return {
            "llm_summary": summary,
            "llm_key_entities": [
                str(x).strip() for x in key_entities if str(x).strip()
            ],
            "llm_likely_questions": [
                str(x).strip() for x in likely_questions if str(x).strip()
            ],
        }

    def _generate_document_summary_llm(self, document_text: str) -> str:
        """Return a concise document-level summary for inclusion in metadata."""
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        # Local import to avoid requiring openai at module import time
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)

        system_prompt = (
            "You are a summarization assistant for insurance policy documents. "
            "Provide a short 1-3 sentence summary describing the WHAT and WHO of the document, "
            "focusing on plan types and coverage aspects. Return only the summary text."
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document:\n\n{document_text[:5000]}"},
            ],
            response_format={"type": "text"},
        )

        try:
            text = response.choices[0].message.content
            return str(text).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate document summary: {e}")
