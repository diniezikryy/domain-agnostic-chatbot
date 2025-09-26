# Domain-Agnostic Chatbot

A pure Python CLI application for querying documents across different domains using hybrid FAISS + BM25 search.

## Overview

This chatbot system can work with different document sets (insurance policies, legal contracts, technical manuals, etc.) by implementing a batch management system. Users can easily switch between different document domains and ask questions specific to each domain.

## Features

- **Multi-format support**: PDF, DOCX, TXT, MD files
- **Hybrid search**: FAISS vector search + BM25 keyword search
- **Batch management**: Organize documents by domain
- **CLI interface**: Simple command-line interaction
- **Domain-agnostic**: No hardcoded domain-specific logic

## Quick Start

### 1. Setup Environment

```bash
# Clone or create project directory
cd domain-agnostic-chatbot

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Add Documents

```bash
# Create domain directories and add your documents
mkdir -p documents/insurance
cp your_insurance_docs/*.pdf documents/insurance/

mkdir -p documents/legal
cp your_legal_docs/*.docx documents/legal/
```

### 3. Create Document Batches

```bash
# Process insurance documents
python setup_batch.py insurance

# Process legal documents
python setup_batch.py legal
```

### 4. Query Documents

```bash
# Ask questions using default batch
python main.py "What medical conditions are covered?"

# Ask questions using specific batch
python main.py --batch insurance "Can I claim for diabetes?"
python main.py --batch legal "What are the termination clauses?"

# List available batches
python main.py --list-batches

# Get batch information
python main.py --batch-info insurance
```

## CLI Commands

### Core Commands

```bash
# Ask a question (uses default batch)
python main.py "Your question here"

# Ask with specific batch
python main.py --batch <batch_name> "Your question here"

# List available batches
python main.py --list-batches

# Get batch information
python main.py --batch-info <batch_name>

# Set default batch
python main.py --set-default <batch_name>
```

### Batch Management Commands

```bash
# Create new batch from documents
python setup_batch.py <batch_name>

# Recreate existing batch
python setup_batch.py <batch_name> --rebuild

# Delete batch
python setup_batch.py <batch_name> --delete

# Use custom source directory
python setup_batch.py <batch_name> --source /path/to/documents
```

## Directory Structure

```
domain-agnostic-chatbot/
├── main.py                    # CLI entry point
├── setup_batch.py             # Batch creation script
├── batch_manager.py           # Core batch management
├── document_processor.py      # Document processing
├── query_processor.py         # Query processing
├── utils/
│   ├── file_handlers.py       # PDF, DOCX, TXT processors
│   ├── embeddings.py          # Text embedding utilities
│   └── search.py              # FAISS + BM25 search logic
├── config/
│   └── settings.py            # Configuration management
├── documents/                 # User document input
│   ├── insurance/             # Insurance documents
│   ├── legal/                 # Legal documents
│   └── technical/             # Technical manuals
├── batches/                   # Generated document batches
│   ├── batch_registry.json    # Central batch registry
│   ├── insurance/             # Processed insurance batch
│   └── legal/                 # Processed legal batch
└── requirements.txt           # Dependencies
```

## How It Works

1. **Document Processing**: Documents are processed into text chunks with metadata
2. **Index Creation**: FAISS (semantic) and BM25 (keyword) indexes are built for each batch
3. **Batch Management**: Users can switch between different document domains
4. **Hybrid Search**: Queries use both FAISS and BM25 for comprehensive results
5. **Response Generation**: Results are combined and formatted for the user

## Configuration

Environment variables:
- `OPENAI_API_KEY`: Required for embeddings and response generation

Settings can be modified in `config/settings.py`:
- Chunk size and overlap
- Search weights (FAISS vs BM25)
- Model names
- API parameters

## Performance

Target performance metrics:
- Response time: <6s average
- Memory usage: <500MB per query
- Batch switching: <2s
- Document processing: <5 minutes per 100 documents

## Examples

### Insurance Policy Queries
```bash
python main.py --batch insurance "What is the waiting period for cancer coverage?"
python main.py --batch insurance "How do I file a claim?"
python main.py --batch insurance "What conditions are excluded?"
```

### Legal Contract Queries
```bash
python main.py --batch legal "What are the termination clauses?"
python main.py --batch legal "What are the payment terms?"
python main.py --batch legal "Who owns intellectual property?"
```

### Technical Manual Queries
```bash
python main.py --batch technical "How do I install the software?"
python main.py --batch technical "What are the system requirements?"
python main.py --batch technical "How do I troubleshoot errors?"
```

## Roadmap

**Phase 2**: Web Integration
- FastAPI server for programmatic access
- RESTful APIs for batch management
- Web-based document upload

**Phase 3**: MCP Integration
- MCP server exposing domain switching tools
- Integration with Claude Code
- Enhanced tool descriptions

**Phase 4**: Advanced Features
- Real-time document updates
- Analytics and usage tracking
- Multi-user support
- Cloud deployment

## Troubleshooting

### Common Issues

1. **No embeddings generated**
   - Check OPENAI_API_KEY is set
   - Verify internet connection
   - Check API quota

2. **Document processing fails**
   - Ensure file format is supported (PDF, DOCX, TXT, MD)
   - Check file permissions
   - Verify file is not corrupted

3. **Batch not found**
   - Use `--list-batches` to see available batches
   - Ensure batch was created successfully
   - Check `batches/batch_registry.json`

4. **Slow performance**
   - Check available memory
   - Reduce batch size
   - Optimize chunk settings

## Contributing

This is a foundation for building domain-agnostic document query systems. The modular architecture allows for easy extension and customization.

Key extension points:
- Add new file format handlers in `utils/file_handlers.py`
- Customize search logic in `utils/search.py`
- Modify query processing in `query_processor.py`
- Add new CLI commands in `main.py`