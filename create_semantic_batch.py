import sys, os
sys.path.insert(0, os.getcwd())
import argparse
from document_processor import DocumentProcessor
from batch_manager import BatchManager

doc_processor = DocumentProcessor()
doc_paths = [
    'documents/my_policies/GREAT_SupremeHealth_Benefits.pdf',
    'documents/my_policies/GREAT_TravelCare.pdf',
    'documents/my_policies/Manulife_Policy_Illustration_REDACTED.pdf'
]

parser = argparse.ArgumentParser(description='Create a semantic-chunked batch for evaluation')
parser.add_argument('--batch_id', type=str, default='my_policies_semantic', help='Batch id to create')
parser.add_argument('--force', action='store_true', default=False, help='Overwrite an existing batch')
args = parser.parse_args()

doc_processor = DocumentProcessor()
batch_manager = BatchManager()

if batch_manager.get_batch_info(args.batch_id) and not args.force:
    print(f"Semantic batch '{args.batch_id}' already exists. Use --force to recreate.")
    sys.exit(0)

print('Creating semantic chunking batch...')
success = doc_processor.create_batch(
    batch_id=args.batch_id,
    document_paths=doc_paths,
    batch_name='My Policies (Semantic Chunking)',
    description='Test batch with semantic chunking',
    chunking_strategy='semantic'
)
result = 'SUCCESS' if success else 'FAILED'
print('Batch creation:', result)