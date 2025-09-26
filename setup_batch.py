#!/usr/bin/env python3
"""
Batch Setup Script
Creates document batches from files in the documents directory.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from batch_manager import BatchManager

def main():
    parser = argparse.ArgumentParser(description="Create document batches for the chatbot")

    parser.add_argument("batch_name", help="Name of the batch to create")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild existing batch")
    parser.add_argument("--delete", action="store_true", help="Delete existing batch")
    parser.add_argument("--source", help="Source directory (default: documents/{batch_name})")

    args = parser.parse_args()

    try:
        batch_manager = BatchManager()

        # Handle deletion
        if args.delete:
            if batch_manager.delete_batch(args.batch_name):
                print(f"Batch '{args.batch_name}' deleted successfully.")
            else:
                print(f"Failed to delete batch '{args.batch_name}' or batch does not exist.")
            return

        # Determine source directory
        source_dir = args.source or f"documents/{args.batch_name}"
        source_path = Path(source_dir)

        if not source_path.exists():
            print(f"Source directory '{source_dir}' does not exist.")
            print(f"Create it and add your documents, then run this script again.")
            return

        # Check for documents
        document_files = []
        for ext in ['*.pdf', '*.docx', '*.txt', '*.md']:
            document_files.extend(source_path.glob(ext))

        if not document_files:
            print(f"No documents found in '{source_dir}'.")
            print("Supported formats: PDF, DOCX, TXT, MD")
            return

        print(f"Found {len(document_files)} documents to process:")
        for doc in document_files:
            print(f"  - {doc.name}")

        # Check if batch exists
        existing_batch = batch_manager.get_batch_info(args.batch_name)
        if existing_batch and not args.rebuild:
            print(f"Batch '{args.batch_name}' already exists.")
            print("Use --rebuild to recreate it.")
            return

        # Process documents
        print(f"\nCreating batch '{args.batch_name}'...")
        document_processor = DocumentProcessor()

        success = document_processor.create_batch(
            batch_id=args.batch_name,
            document_paths=[str(doc) for doc in document_files],
            batch_name=args.batch_name.replace('_', ' ').title(),
            description=f"Document batch created from {source_dir}"
        )

        if success:
            print(f"✅ Batch '{args.batch_name}' created successfully!")

            # Show batch info
            info = batch_manager.get_batch_info(args.batch_name)
            if info:
                print(f"Documents processed: {info.get('doc_count', 0)}")
                print(f"Total chunks: {info.get('statistics', {}).get('total_chunks', 0)}")

            print(f"\nYou can now query this batch with:")
            print(f"  python main.py --batch {args.batch_name} \"your question here\"")
        else:
            print(f"❌ Failed to create batch '{args.batch_name}'")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()