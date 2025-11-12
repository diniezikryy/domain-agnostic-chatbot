import sys, os
sys.path.insert(0, os.getcwd())
from document_processor import DocumentProcessor

doc_processor = DocumentProcessor()
doc_paths = [
    'documents/my_policies/GREAT_SupremeHealth_Benefits.pdf',
    'documents/my_policies/GREAT_TravelCare.pdf',
    'documents/my_policies/Manulife_Policy_Illustration_REDACTED.pdf'
]

print('Creating semantic chunking batch...')
success = doc_processor.create_batch(
    batch_id='my_policies_semantic',
    document_paths=doc_paths,
    batch_name='My Policies (Semantic Chunking)',
    description='Test batch with semantic chunking',
    chunking_strategy='semantic'
)
result = 'SUCCESS' if success else 'FAILED'
print('Batch creation:', result)