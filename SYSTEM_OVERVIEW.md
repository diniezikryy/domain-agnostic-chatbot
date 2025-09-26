# Domain-Agnostic Chatbot - System Overview

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Component Breakdown](#component-breakdown)
3. [End-to-End Flow](#end-to-end-flow)
4. [Data Flow Diagram](#data-flow-diagram)
5. [File Structure](#file-structure)
6. [Key Algorithms](#key-algorithms)

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │    Batch        │    │    Query        │
│  Processing     │    │  Management     │    │  Processing     │
│                 │    │                 │    │                 │
│ PDF/DOCX/TXT    │───▶│ Switch Domains  │◀───│ User Questions  │
│ → Chunks        │    │ Load Indexes    │    │ → AI Responses  │
│ → FAISS Index   │    │ Registry        │    │                 │
│ → BM25 Index    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Breakdown

### 1. Main CLI Interface (`main.py`)
**Purpose**: Entry point for user interactions
**Key Functions**:
- Parse command-line arguments
- Handle batch switching (`--batch`, `--list-batches`)
- Route queries to QueryProcessor
- Display responses to user

**Code Structure**:
```python
def main():
    # Parse arguments (query, --batch, --list-batches, etc.)
    # Initialize BatchManager and QueryProcessor
    # Handle batch operations or process query
    # Display results
```

### 2. Batch Manager (`batch_manager.py`)
**Purpose**: Manages different document domains and switching between them
**Key Functions**:
- Load/save batch registry (`batch_registry.json`)
- Switch between document batches
- Track batch metadata (doc count, creation date)
- Provide file paths for FAISS/BM25 indexes

**Data Structure**:
```json
{
  "batches": {
    "insurance": {
      "id": "insurance",
      "name": "Insurance Policies",
      "doc_count": 4,
      "faiss_path": "batches/insurance/faiss_index",
      "bm25_path": "batches/insurance/bm25_index.pkl"
    }
  },
  "default_batch": "insurance"
}
```

### 3. Document Processor (`document_processor.py`)
**Purpose**: Converts raw documents into searchable indexes
**Process**:
1. Read documents (PDF, DOCX, TXT, MD)
2. Extract and clean text
3. Split into chunks (800 chars with 100 overlap)
4. Generate embeddings via OpenAI API
5. Build FAISS and BM25 indexes
6. Save indexes and metadata

### 4. Query Processor (`query_processor.py`)
**Purpose**: Handles user questions and generates responses
**Process**:
1. Load appropriate batch indexes
2. Preprocess user query
3. Perform hybrid search (FAISS + BM25)
4. Generate conversational response via GPT-4o-mini
5. Return formatted answer

### 5. Search Engine (`utils/search.py`)
**Purpose**: Hybrid FAISS + BM25 search implementation
**Components**:
- **SearchIndexBuilder**: Creates indexes during batch setup
- **HybridSearchEngine**: Performs real-time search queries

### 6. File Handlers (`utils/file_handlers.py`)
**Purpose**: Extract text from different document formats
**Supported Formats**: PDF (PyPDF2), DOCX (python-docx), TXT, MD

### 7. Embedding Generator (`utils/embeddings.py`)
**Purpose**: Generate vector embeddings for semantic search
**API**: OpenAI text-embedding-3-small (1536 dimensions)

## End-to-End Flow

### Phase 1: Document Setup (One-time per domain)

```
1. User drops documents into documents/insurance/
   ├── FWD_Critical_Illness_Plus.pdf
   ├── AIA_Ultimate_Critical_Cover.pdf
   └── MSIG_CriticalCarePlus.pdf

2. User runs: python setup_batch.py insurance

3. Document Processing Flow:
   ┌─────────────┐
   │ PDF Files   │
   └─────┬───────┘
         ▼
   ┌─────────────┐
   │FileHandler  │ Extract text, clean, chunk
   │.process_doc │ → 182 chunks (800 chars each)
   └─────┬───────┘
         ▼
   ┌─────────────┐
   │Embedding    │ OpenAI API call
   │Generator    │ → 182 vectors (1536 dims)
   └─────┬───────┘
         ▼
   ┌─────────────┐
   │SearchIndex  │
   │Builder      │
   └─────┬───────┘
         ├─ FAISS Index (semantic search)
         └─ BM25 Index (keyword search)

4. Batch Registration:
   └─ Updates batch_registry.json
   └─ Creates batches/insurance/ directory
```

### Phase 2: Query Processing (Real-time)

```
User: python main.py --batch insurance "What cancers are covered?"

1. CLI Processing (main.py):
   ├─ Parse arguments
   ├─ Initialize BatchManager
   └─ Initialize QueryProcessor

2. Batch Loading (batch_manager.py):
   ├─ Switch to 'insurance' batch
   ├─ Get file paths from registry
   └─ Return paths to QueryProcessor

3. Index Loading (utils/search.py):
   ├─ Load FAISS index (182 vectors)
   ├─ Load BM25 index (182 tokenized chunks)
   └─ Initialize HybridSearchEngine

4. Query Processing (query_processor.py):

   Step 4.1: Query Preprocessing
   ┌─────────────────────────────────────────┐
   │ "What cancers are covered?"             │
   │           ↓                             │
   │ Remove stopwords: "cancers covered"     │
   └─────────────────────────────────────────┘

   Step 4.2: Hybrid Search
   ┌─────────────────┐    ┌─────────────────┐
   │ FAISS Search    │    │ BM25 Search     │
   │                 │    │                 │
   │ 1. Generate     │    │ 1. Tokenize     │
   │    query vector │    │    query        │
   │ 2. Cosine       │    │ 2. TF-IDF       │
   │    similarity   │    │    scoring      │
   │ 3. Top 10       │    │ 3. Top 10       │
   │    results      │    │    results      │
   └─────────┬───────┘    └─────────┬───────┘
             │                      │
             └──────┬─────────┬─────┘
                    ▼         ▼
   ┌─────────────────────────────────────────┐
   │ Result Fusion                           │
   │ • Normalize scores                      │
   │ • Weight: 60% FAISS + 40% BM25         │
   │ • Remove duplicates                     │
   │ • Return top 5 chunks                  │
   └─────────────────────────────────────────┘

   Step 4.3: Response Generation
   ┌─────────────────────────────────────────┐
   │ GPT-4o-mini API Call                    │
   │                                         │
   │ System: "You are an insurance expert"   │
   │ User: Context + Question + Instructions │
   │                                         │
   │ Context: Top 5 policy chunks           │
   │ Question: "What cancers are covered?"   │
   │ Instructions: "Give clear answer..."    │
   └─────────────────────────────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────────────┐
   │ "Based on the policy documents, the     │
   │ following cancers are covered:          │
   │ • Major cancer (any malignant tumor)    │
   │ • Early stage cancers like carcinoma-   │
   │   in-situ and T1N0M0 classifications   │
   │ • Specific coverage for breast, lung... │
   │                                         │
   │ However, all cancers in the presence of │
   │ HIV infection are excluded..."          │
   └─────────────────────────────────────────┘

5. Response Display:
   └─ Formatted output to terminal
```

## Data Flow Diagram

```
┌─────────────┐
│ documents/  │
│ insurance/  │ 4 PDF files
└─────┬───────┘
      │ setup_batch.py
      ▼
┌─────────────┐
│ Text        │ 182 chunks
│ Extraction  │ (800 chars each)
└─────┬───────┘
      │ OpenAI API
      ▼
┌─────────────┐
│ Embeddings  │ 182 vectors
│ Generation  │ (1536 dims)
└─────┬───────┘
      │
      ▼
┌─────────────┐     ┌─────────────┐
│ FAISS       │     │ BM25        │
│ Index       │     │ Index       │
│ (semantic)  │     │ (keyword)   │
└─────┬───────┘     └─────┬───────┘
      │                   │
      │ Query Time        │
      │                   │
      ▼                   ▼
┌─────────────────────────────────┐
│ Hybrid Search Engine            │
│ • Load both indexes             │
│ • Process user query            │
│ • Combine results               │
└─────────────┬───────────────────┘
              │ Top 5 chunks
              ▼
┌─────────────────────────────────┐
│ GPT-4o-mini Response Generation │
│ • Context: Retrieved chunks     │
│ • Query: User question          │
│ • Output: Conversational answer │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ Final Response to User          │
└─────────────────────────────────┘
```

## File Structure & Responsibilities

```
domain-agnostic-chatbot/
├── main.py                    # CLI entry point & argument parsing
├── setup_batch.py             # Document processing orchestrator
├── batch_manager.py           # Domain switching & registry management
├── document_processor.py      # Coordinates document → index pipeline
├── query_processor.py         # Query → response pipeline
├── utils/
│   ├── file_handlers.py       # PDF/DOCX/TXT text extraction
│   ├── embeddings.py          # OpenAI embedding generation
│   └── search.py              # FAISS + BM25 hybrid search
├── config/
│   └── settings.py            # Configuration & environment
├── documents/                 # User input files
│   └── insurance/             # Domain-specific folders
├── batches/                   # Generated indexes & metadata
│   ├── batch_registry.json    # Central batch tracking
│   └── insurance/             # Per-batch storage
│       ├── faiss_index/       # FAISS vector index
│       ├── bm25_index.pkl     # BM25 keyword index
│       └── metadata.json      # Batch info & statistics
```

## Key Algorithms

### 1. Hybrid Search Scoring
```python
# Normalize scores
faiss_normalized = faiss_score / max_faiss_score
bm25_normalized = bm25_score / max_bm25_score

# Weighted combination
final_score = (faiss_normalized * 0.6) + (bm25_normalized * 0.4)

# Sort by final_score, return top K
```

### 2. Text Chunking Strategy
```python
chunk_size = 800 characters
overlap = 100 characters

# Smart chunking: break at sentence boundaries when possible
if not at_end_of_text:
    find_sentence_ending_within_last_100_chars()
    break_at_sentence_if_found()
```

### 3. Document Processing Pipeline
```python
def create_batch(documents):
    chunks = []
    for doc in documents:
        text = extract_text(doc)           # PDF → text
        doc_chunks = create_chunks(text)   # text → 800-char chunks
        chunks.extend(doc_chunks)

    embeddings = generate_embeddings(chunks)  # OpenAI API
    faiss_index = build_faiss(embeddings)     # Vector index
    bm25_index = build_bm25(chunks)           # Keyword index

    save_indexes(faiss_index, bm25_index)
```

## Performance Characteristics

- **Index Loading**: ~0.5s (182 chunks)
- **Query Processing**: 1-3s (embedding generation + search + LLM)
- **Memory Usage**: ~100MB per batch in memory
- **Storage**: ~50MB per batch on disk
- **Accuracy**: Hybrid search provides both semantic and exact keyword matches

## API Dependencies

1. **OpenAI Embeddings**: text-embedding-3-small (1536 dimensions)
2. **OpenAI Chat**: gpt-4o-mini for response generation
3. **Local Libraries**: FAISS (similarity search), BM25 (keyword search)

This system provides domain-agnostic document querying with professional conversational responses, optimized for insurance policy analysis but extensible to any document domain.