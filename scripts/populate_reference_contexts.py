"""
Populate reference_contexts list with the actual chunk text from FAISS index
for evaluation_dataset_custom_user_questions.json.

Usage:
  python scripts/populate_reference_contexts.py --dataset test_data/evaluation_dataset_custom_user_questions.json --batch_id my_policies --out test_data/evaluation_dataset_custom_user_questions.refined.json
"""
import argparse
import json
import pickle
from pathlib import Path


def load_faiss_chunks(batch_id: str):
    index_file = Path("batches") / batch_id / "faiss_index" / "index.pkl"
    if not index_file.exists():
        raise FileNotFoundError(index_file)
    with open(index_file, "rb") as f:
        data = pickle.load(f)
    return data.get("chunks", []), data.get("metadata", [])


def find_chunk_text(chunks, metadata, filename, page_number):
    for idx, meta in enumerate(metadata):
        if meta.get("filename") == filename and meta.get("page_number") == page_number:
            return chunks[idx]
    return None


def find_chunk_id(metadata, filename, page_number):
    """Return the internal chunk_id from batch metadata for a given file/page."""
    for meta in metadata:
        if meta.get("filename") == filename and meta.get("page_number") == page_number:
            return meta.get("chunk_id")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="test_data/evaluation_dataset_custom_user_questions.json")
    parser.add_argument("--batch_id", default="my_policies")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    in_path = Path(args.dataset)
    out_path = Path(args.out) if args.out else in_path

    chunks, metadata = load_faiss_chunks(args.batch_id)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for q in data:
        refs = q.get("reference_contexts", []) or []
        new_refs = []
        chunk_ids = []
        # If references are strings already, keep them
        # If they are dict with filename & page_number, convert
        for r in refs:
            if isinstance(r, str):
                new_refs.append(r)
            elif isinstance(r, dict):
                filename = r.get("filename")
                page = r.get("page_number") or r.get("page")
                if filename and page:
                    text = find_chunk_text(chunks, metadata, filename, page)
                    if text:
                        # Add found chunk text
                        new_refs.append(text)
                        # Also add the underlying chunk_id (if present in index)
                        cid = find_chunk_id(metadata, filename, page)
                        if cid:
                            chunk_ids.append(cid)
                        changed = True
                    else:
                        # fallback to storing a human-readable note
                        new_refs.append(f"[Missing chunk text for {filename} page {page}]")
                        changed = True
                else:
                    # unknown format - store as string representation
                    new_refs.append(str(r))
            else:
                new_refs.append(str(r))
        q["reference_contexts"] = new_refs
        if chunk_ids:
            # Add a canonicalized list of chunk ids to the dataset to enable
            # stable matching during evaluation. These chunk ids correspond to
            # the per-batch internal metadata (e.g., "p19-0").
            q["reference_chunk_ids"] = chunk_ids

    if changed:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {out_path} (references populated with chunk text)")
    else:
        print("[OK] No changes made; dataset already uses chunk text for references")


if __name__ == '__main__':
    main()
