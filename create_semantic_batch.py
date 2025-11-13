import sys, os
sys.path.insert(0, os.getcwd())
from document_processor import DocumentProcessor

doc_processor = DocumentProcessor()

# Using available critical illness documents for testing
doc_paths = [
    'documents/critical_illness/aia-absolute-critical-cover-brochure.pdf',
    'documents/critical_illness/aia-prime-critical-cover.pdf',
    'documents/critical_illness/Contract - Complete Critical Protect (CCN1).pdf'
]

print('Creating semantic chunking batch with critical illness documents...')
success = doc_processor.create_batch(
    batch_id='my_policies_semantic',
    document_paths=doc_paths,
    batch_name='My Policies (Semantic Chunking)',
    description='Test batch with semantic chunking using critical illness documents',
    chunking_strategy='semantic'
)
result = 'SUCCESS' if success else 'FAILED'
print('Batch creation:', result)