import pickle
import sys
from pathlib import Path
import json


def inspect_batch_chunks(batch_id: str):
    """
    Loads and inspects the chunks and metadata from a specified batch.
    """
    print(f"--- Inspecting Batch: {batch_id} ---")

    # Path to the FAISS index.pkl file, based on search.py
    # We only need to read one, and the FAISS pkl is a simple dictionary
    index_file = Path(f"batches/{batch_id}/faiss_index/index.pkl")

    if not index_file.exists():
        print(f"Error: index.pkl not found for batch '{batch_id}'")
        print(f"Looked at path: {index_file.resolve()}")
        return

    try:
        # Load the data from the pickle file
        with open(index_file, "rb") as f:
            data = pickle.load(f)

        chunks = data.get("chunks")
        metadata_list = data.get("metadata")

        if not chunks or not metadata_list:
            print("Error: 'chunks' or 'metadata' not found in the index file.")
            return

        print(f"Found {len(chunks)} chunks. Displaying details:\n")

        # Loop through each chunk and its metadata
        for i, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
            print("=" * 80)
            print(f"CHUNK {i + 1} / {len(chunks)}")
            print("=" * 80)

            # Print the metadata in a readable format
            print(f"METADATA:")
            print(f"  - Filename: {metadata.get('filename')}")
            print(f"  - Page: {metadata.get('page_number')}")
            print(f"  - Heading: {metadata.get('page_heading')}")
            # extraction_method was added by the hybrid FileHandler
            print(f"  - Extraction Method: {metadata.get('extraction_method')}")

            # This is the "smart metadata" we've been working on
            print(f"  - Plan Context: {metadata.get('plan_context')}")

            # Print the actual chunk content
            print("\nCHUNK CONTENT:")
            print(chunk)
            print("\n\n")

    except Exception as e:
        print(f"An error occurred while reading the batch file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide a batch_id as an argument.")
        print("Usage: python inspect_batch.py <batch_id>")
        sys.exit(1)

    batch_to_inspect = sys.argv[1]
    inspect_batch_chunks(batch_to_inspect)
