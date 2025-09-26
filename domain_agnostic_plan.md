# Multi-Domain Chatbot Implementation Plan

## Overview

Transform the current SIT-specific chatbot into a domain-agnostic system that can work with different document sets (insurance policies, legal contracts, technical manuals, etc.) using a pure Python CLI implementation with batch management.

## Implementation Strategy

**Phase 1: Pure Python CLI Application**
- Build core functionality without web framework complexity
- Prove domain-agnostic architecture works
- Simple file-based document management
- CLI interface for testing and validation

**Phase 2: Web Integration (Future)**
- Add FastAPI server layer
- Implement MCP server integration
- Maintain backward compatibility

## Current System Analysis

**Existing Architecture:**
- Hybrid retrieval: FAISS vector search + BM25 keyword search
- Query processing: GPT-4o-mini for rewriting and response generation
- FastAPI server with OpenAI-compatible endpoints
- Performance: ~5.7s average response time, 100% success rate

**SIT-Specific Components to Remove:**
- `SIT_KEYWORDS` list and university-specific bias
- Academic advisor persona in prompts
- SIT-specific query expansions and preprocessing
- Hard-coded university terminology injection

## Architecture Design

### CLI Application Flow
```
User Drops Files → Document Processing → Batch Creation → Query Processing → Response
      ↓                     ↓                 ↓              ↓             ↓
documents/domain/    DocumentProcessor   BatchManager   QueryProcessor   CLI Output
```

### Directory Structure
```
domain-agnostic-chatbot/
├── main.py                        # CLI entry point
├── setup_batch.py                 # Batch creation script
├── batch_manager.py               # Core batch management
├── document_processor.py          # Generic document processing
├── query_processor.py             # Domain-agnostic query processing
├── utils/
│   ├── file_handlers.py           # PDF, DOCX, TXT processors
│   ├── embeddings.py              # Text embedding utilities
│   └── search.py                  # FAISS + BM25 search logic
├── config/
│   └── settings.py                # Configuration management
├── documents/                     # User document input
│   ├── insurance/                 # Insurance documents
│   ├── legal/                     # Legal documents
│   ├── technical/                 # Technical manuals
│   └── medical/                   # Medical documents
├── batches/                       # Generated document batches
│   ├── batch_registry.json        # Central batch registry
│   ├── insurance/                 # Processed insurance batch
│   │   ├── faiss_index/
│   │   ├── bm25_index.pkl
│   │   └── metadata.json
│   ├── legal/                     # Processed legal batch
│   └── technical/                 # Processed technical batch
├── scripts/
│   ├── create_batch.py           # CLI tool for batch creation
│   ├── migrate_sit_data.py       # Migration script
│   └── test_batches.py           # Testing utilities
├── server.py                     # Updated FastAPI server
├── query_test.py                 # Modified query processing
└── requirements.txt              # Updated dependencies
```

├── tests/                          # Test suite
│   ├── test_document_processor.py
│   ├── test_batch_manager.py
│   ├── test_query_processor.py
│   └── test_integration.py
├── requirements.txt                # Dependencies
└── README.md                       # Usage instructions
```

## Requirements

### Functional Requirements

**FR1: Multi-Format Document Processing**
- Support PDF, DOCX, TXT, and MD file formats
- Extract text content and preserve structure
- Handle documents of varying sizes (1KB to 50MB)
- Generate consistent chunking across formats

**FR2: Domain-Agnostic Batch Management**
- Create document batches from directory contents
- Switch between different document domains
- Maintain separate indexes per batch
- List available batches with metadata

**FR3: Hybrid Search System**
- Implement FAISS vector search for semantic similarity
- Implement BM25 keyword search for exact matches
- Combine search results with weighted scoring
- Maintain performance: <6s average response time

**FR4: Query Processing**
- Remove domain-specific bias and keywords
- Generic query rewriting for better retrieval
- Context-aware response generation
- Preserve abbreviation handling capabilities

**FR5: CLI Interface**
- Simple command to ask questions: `python main.py "question"`
- Batch setup command: `python setup_batch.py domain_name`
- Batch switching: `python main.py --batch domain_name "question"`
- Status and listing commands

### Non-Functional Requirements

**NFR1: Performance**
- Response time: <6s average (maintain current 5.7s benchmark)
- Memory usage: <500MB per query
- Support batch sizes up to 1000 documents
- Index creation: <5 minutes per 100 documents

**NFR2: Reliability**
- 95% query success rate
- Graceful error handling for corrupted files
- Recovery from index corruption
- Consistent results across runs

**NFR3: Maintainability**
- Modular component design
- Clear separation of concerns
- Comprehensive logging
- Unit test coverage >80%

**NFR4: Usability**
- Simple file-drop workflow
- Clear error messages
- Progress indicators for batch creation
- Minimal configuration required

## Implementation Timeline

### Phase 1: Core CLI Application (Week 1-2)

#### Week 1: Foundation & Document Processing
**Objectives:**
- Set up pure Python CLI project structure
- Implement document processing pipeline
- Create basic batch management

**Tasks:**
- [ ] Create directory structure and core files
- [ ] Implement `DocumentProcessor` class
- [ ] Add multi-format file handlers (PDF, DOCX, TXT, MD)
- [ ] Create text chunking and embedding logic
- [ ] Design batch registry format (JSON)
- [ ] Implement basic CLI entry point

**Deliverables:**
- Working document processing pipeline
- Basic CLI interface
- Directory structure with file handlers

#### Week 2: Search & Query Processing
**Objectives:**
- Port existing search logic to new architecture
- Remove SIT-specific components
- Implement batch switching

**Tasks:**
- [ ] Port FAISS + BM25 hybrid search logic
- [ ] Create generic query processing (remove SIT bias)
- [ ] Implement `BatchManager` class
- [ ] Add batch creation script (`setup_batch.py`)
- [ ] Create CLI query interface
- [ ] Basic error handling and logging

**Deliverables:**
- Working hybrid search system
- Functional batch switching
- Complete CLI interface

### Phase 2: Testing & Optimization (Week 3-4)

#### Week 3: Integration Testing
**Objectives:**
- Test with multiple document domains
- Performance optimization
- Error handling improvements

**Tasks:**
- [ ] Create test document sets (insurance, legal, technical)
- [ ] Performance testing and optimization
- [ ] Memory usage optimization
- [ ] Comprehensive error handling
- [ ] Logging and debugging improvements
- [ ] Cross-domain query testing

**Deliverables:**
- Tested multi-domain system
- Performance benchmarks
- Robust error handling

#### Week 4: Migration & Documentation
**Objectives:**
- Migrate existing SIT data
- Complete documentation
- Prepare for Phase 2 (web integration)

**Tasks:**
- [ ] Create migration script for existing SIT data
- [ ] Write comprehensive documentation
- [ ] Create usage examples and tutorials
- [ ] Prepare codebase for web framework integration
- [ ] Final testing and validation

**Deliverables:**
- Migration scripts
- Complete documentation
- Production-ready CLI application

## Acceptance Tests

### AT1: Document Processing Tests

**AT1.1: Multi-Format Support**
```bash
# Setup: Place test files in documents/test/
documents/test/
├── sample.pdf      # 5MB insurance policy
├── contract.docx   # 2MB legal contract
├── manual.txt      # 1MB technical manual
└── guide.md        # 500KB markdown guide

# Test: Process all formats
python setup_batch.py test

# Expected: All files processed successfully
✓ 4 documents processed
✓ FAISS index created (test/faiss_index/)
✓ BM25 index created (test/bm25_index.pkl)
✓ Metadata file generated (test/metadata.json)
```

**AT1.2: Large Document Handling**
```bash
# Setup: 50MB PDF file
# Test: Process without memory issues
python setup_batch.py large_docs

# Expected: Successful processing within memory limits
✓ Large document chunked appropriately
✓ Memory usage < 500MB during processing
✓ Processing time < 5 minutes
```

### AT2: Batch Management Tests

**AT2.1: Batch Creation and Switching**
```bash
# Test: Create multiple batches
python setup_batch.py insurance
python setup_batch.py legal
python setup_batch.py technical

# Test: List available batches
python main.py --list-batches

# Expected output:
Available batches:
- insurance (150 documents, created 2025-01-15)
- legal (89 documents, created 2025-01-15)
- technical (45 documents, created 2025-01-15)
```

**AT2.2: Batch Switching Performance**
```bash
# Test: Switch between batches quickly
python main.py --batch insurance "What is covered?"
python main.py --batch legal "What are termination clauses?"
python main.py --batch technical "How to install?"

# Expected: Each switch < 2s, correct domain responses
```

### AT3: Query Processing Tests

**AT3.1: Domain-Agnostic Responses**
```bash
# Test: Same question across domains
python main.py --batch insurance "What are the benefits?"
python main.py --batch legal "What are the benefits?"

# Expected: Different, domain-appropriate responses
# Insurance: Lists medical benefits, coverage details
# Legal: Lists employment benefits, contract terms
```

**AT3.2: Performance Benchmarks**
```bash
# Test: Response time consistency
python main.py --batch insurance "Can I claim for cancer?" # Run 10 times

# Expected:
✓ Average response time < 6s
✓ 95% of queries succeed
✓ Consistent response quality
✓ Memory usage < 500MB per query
```

### AT4: Error Handling Tests

**AT4.1: Corrupted File Handling**
```bash
# Setup: Add corrupted PDF to documents/test/
# Test: Process batch with corrupted file
python setup_batch.py test

# Expected: Graceful error handling
✓ Process other files successfully
✓ Log error for corrupted file
✓ Continue batch creation
✓ Clear error message to user
```

**AT4.2: Missing Batch Handling**
```bash
# Test: Query non-existent batch
python main.py --batch nonexistent "test question"

# Expected:
Error: Batch 'nonexistent' not found.
Available batches: insurance, legal, technical
```

### AT5: Integration Tests

**AT5.1: End-to-End Workflow**
```bash
# Complete user workflow test:

# 1. User drops insurance documents
cp ~/insurance_docs/* documents/insurance/

# 2. User creates batch
python setup_batch.py insurance

# 3. User asks questions
python main.py --batch insurance "Can I claim for stage 2 breast cancer?"
python main.py --batch insurance "What is the waiting period?"
python main.py --batch insurance "How do I file a claim?"

# Expected: All steps complete successfully with accurate responses
```

**AT5.2: Migration Test**
```bash
# Test: Migrate existing SIT data
python scripts/migrate_sit_data.py

# Expected:
✓ SIT batch created in batches/sit/
✓ Existing FAISS index copied
✓ Existing BM25 index copied
✓ Metadata generated
✓ Batch registry updated
✓ Queries work with migrated data
```

## Usage Examples

### Basic Usage
```bash
# 1. Drop documents into domain folder
cp *.pdf documents/insurance/

# 2. Create document batch
python setup_batch.py insurance

# 3. Ask questions
python main.py "What medical conditions are covered?"
python main.py --batch insurance "Can I claim for diabetes?"
```

### Advanced Usage
```bash
# List all batches
python main.py --list-batches

# Get batch information
python main.py --batch-info insurance

# Switch default batch
python main.py --set-default insurance

# Ask question with specific batch
python main.py --batch legal "What are the termination clauses?"
```

## CLI Command Reference

### Core Commands
```bash
# Ask a question (uses default batch)
python main.py "What are the coverage details?"

# Ask with specific batch
python main.py --batch insurance "Can I claim for cancer?"

# List available batches
python main.py --list-batches

# Get batch information
python main.py --batch-info insurance

# Set default batch
python main.py --set-default insurance
```

### Batch Management Commands
```bash
# Create new batch from documents
python setup_batch.py batch_name

# Recreate existing batch
python setup_batch.py batch_name --rebuild

# Delete batch
python setup_batch.py batch_name --delete

# Migrate existing SIT data
python scripts/migrate_sit_data.py
```

## Configuration Files

### Batch Registry Format (`batches/batch_registry.json`)
```json
{
  "batches": {
    "insurance": {
      "id": "insurance",
      "name": "Insurance Policies",
      "description": "FWD Critical Illness policies and documentation",
      "doc_count": 150,
      "created_at": "2025-01-15T10:00:00Z",
      "faiss_path": "batches/insurance/faiss_index",
      "bm25_path": "batches/insurance/bm25_index.pkl",
      "metadata_path": "batches/insurance/metadata.json"
    }
  },
  "default_batch": "insurance",
  "last_modified": "2025-01-15T10:00:00Z"
}
```

### Individual Batch Metadata (`batches/{batch_id}/metadata.json`)
```json
{
  "batch_id": "insurance",
  "name": "Insurance Policies",
  "description": "FWD Critical Illness policies",
  "created_at": "2025-01-15T10:00:00Z",
  "documents": [
    {
      "filename": "FWD_Critical_Illness_Plus.pdf",
      "file_path": "documents/insurance/FWD_Critical_Illness_Plus.pdf",
      "processed_at": "2025-01-15T10:05:00Z",
      "chunk_count": 156,
      "file_size_mb": 1.2
    }
  ],
  "statistics": {
    "total_documents": 1,
    "total_chunks": 156,
    "avg_chunk_size": 800
  }
}
```

## Performance Considerations

### Current Benchmarks (to maintain)
- **Average response time**: ~5.7s
- **Success rate**: 100%
- **Memory usage**: ~357MB average per query
- **Document retrieval**: ~13 docs average per query

### CLI Performance Targets
- **Batch switching**: <2s
- **Document processing**: <5 minutes per 100 documents
- **Query response**: <6s average
- **Memory usage**: <500MB per query

### Optimization Strategies
- **Lazy loading**: Load indexes only when batch is active
- **Memory management**: Clear unused batch data
- **Index optimization**: Compress large FAISS indexes
- **Caching**: Cache frequently accessed chunks

## Migration Strategy

### Existing Data Migration
```bash
# Migrate current SIT data to new batch system
python scripts/migrate_sit_data.py

# This script will:
# 1. Copy combined_faiss to batches/sit/faiss_index/
# 2. Copy bm25_index.pkl to batches/sit/bm25_index.pkl
# 3. Create metadata.json for SIT batch
# 4. Update batch registry with SIT batch
# 5. Set SIT as default batch
# 6. Test query compatibility
```

### File Mapping
```
Current System          →  New System
combined_faiss/         →  batches/sit/faiss_index/
bm25_index.pkl         →  batches/sit/bm25_index.pkl
abbreviation_cache.json →  config/abbreviations.json
query_test.py          →  query_processor.py (refactored)
```

## Success Metrics

### Technical Metrics
- [ ] Response time maintains < 6s average (current: 5.7s)
- [ ] Memory usage stays < 500MB per query (current: 357MB)
- [ ] Support 4+ file formats (PDF, DOCX, TXT, MD)
- [ ] Zero data loss during SIT migration
- [ ] Unit test coverage > 80%

### Functional Metrics
- [ ] Support 3+ different document domains successfully
- [ ] Batch switching < 2s response time
- [ ] Document processing < 5 minutes per 100 documents
- [ ] 95%+ query success rate across all domains
- [ ] End-to-end workflow completion in <10 minutes

### User Experience Metrics
- [ ] Simple 3-step workflow: drop files → create batch → ask questions
- [ ] Clear error messages with actionable guidance
- [ ] Intuitive CLI commands requiring minimal learning
- [ ] Consistent response quality across domains

## Quality Gates

### Phase 1 Completion Criteria
- [ ] All acceptance tests (AT1-AT5) pass
- [ ] Performance benchmarks met
- [ ] SIT data migration successful
- [ ] 3 different document domains working

### Ready for Phase 2 (Web Integration)
- [ ] CLI system stable and tested
- [ ] Clean modular architecture in place
- [ ] Documentation complete
- [ ] Performance optimized

## Future Roadmap

### Phase 2: Web Integration (Future)
- Add FastAPI server layer for programmatic access
- Implement RESTful APIs for batch management
- Add web-based document upload interface

### Phase 3: MCP Integration (Future)
- Create MCP server that exposes domain switching tools
- Integrate with Claude Code for seamless document querying
- Add Claude-friendly tool descriptions and parameters

### Phase 4: Advanced Features (Future)
- Real-time document updates and re-indexing
- Advanced analytics and usage tracking
- Multi-user support with access controls
- Cloud deployment options

## Conclusion

This implementation plan transforms the SIT-specific chatbot into a flexible, domain-agnostic system using a pragmatic approach:

**Phase 1 Benefits:**
- ✅ Pure Python CLI - no web framework complexity
- ✅ Preserves working 5.7s response time optimization
- ✅ Simple file-drop workflow for users
- ✅ Comprehensive testing and validation
- ✅ Foundation for future web/MCP integration

**Key Success Factors:**
1. **Incremental approach** - CLI first, then web layers
2. **Performance preservation** - maintain current benchmarks
3. **Comprehensive testing** - acceptance tests ensure quality
4. **Clear requirements** - functional and non-functional specifications
5. **Migration strategy** - seamless transition from SIT system

The resulting CLI application will provide a solid foundation for building web APIs and MCP integration while ensuring the core domain-agnostic architecture is proven and optimized.