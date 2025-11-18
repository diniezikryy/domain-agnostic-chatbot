import pickle, json, re
from pathlib import Path

faiss_pkl = Path('batches') / 'my_policies_semantic' / 'faiss_index' / 'index.pkl'
if not faiss_pkl.exists():
    print(json.dumps({'error': f'File not found: {faiss_pkl}'}))
    raise SystemExit(1)

with open(faiss_pkl, 'rb') as f:
    data = pickle.load(f)

chunks = data.get('chunks', [])
metadata = data.get('metadata', [])

# Search for common variants that might represent "S$800 per day"
patterns = [r'S\$\s*800', r'\$800', r'800\s*per\s*day', r'800\s*/\s*day']
regex = re.compile('|'.join(patterns), re.IGNORECASE)

matches = []
for i, (c, m) in enumerate(zip(chunks, metadata)):
    if regex.search(c):
        matches.append({
            'index': i,
            'chunk_id': m.get('chunk_id'),
            'filename': m.get('filename'),
            'page_number': m.get('page_number'),
            'chunk_size': m.get('chunk_size'),
            'chunking_strategy': m.get('chunking_strategy'),
            'extraction_method': m.get('extraction_method'),
            'page_heading': m.get('page_heading'),
            'preview': c[:400].replace('\n', '\\n')
        })

# If no direct matches, look for 'rehabilitation' and inspect nearby chunks
if not matches:
    rehab_matches = []
    for i, (c, m) in enumerate(zip(chunks, metadata)):
        if 'rehabilitat' in c.lower() or 'rehabilitation' in c.lower():
            rehab_matches.append({
                'index': i,
                'chunk_id': m.get('chunk_id'),
                'filename': m.get('filename'),
                'page_number': m.get('page_number'),
                'chunk_size': m.get('chunk_size'),
                'chunking_strategy': m.get('chunking_strategy'),
                'extraction_method': m.get('extraction_method'),
                'page_heading': m.get('page_heading'),
                'preview': c[:400].replace('\n', '\\n')
            })
    matches = rehab_matches

output = {
    'total_chunks': len(chunks),
    'matches_found': len(matches),
    'matches': matches[:50]
}

print(json.dumps(output, ensure_ascii=False, indent=2))
