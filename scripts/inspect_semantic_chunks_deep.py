import pickle,json,re
from pathlib import Path
pkl = Path('batches') / 'my_policies_semantic' / 'faiss_index' / 'index.pkl'
if not pkl.exists():
    print('index.pkl not found'); raise SystemExit(1)
with open(pkl,'rb') as f:
    d = pickle.load(f)
chunks = d.get('chunks',[])
meta = d.get('metadata',[])

matches = []
for i,(c,m) in enumerate(zip(chunks,meta)):
    lower = c.lower()
    if '800' in lower or 'rehabilitat' in lower:
        matches.append({
            'i':i,
            'chunk_id': m.get('chunk_id'),
            'page': m.get('page_number'),
            'filename': m.get('filename'),
            'chunking_strategy': m.get('chunking_strategy'),
            'preview': c[:800].replace('\n','\\n')
        })
print(json.dumps({'total_chunks': len(chunks), 'hits': len(matches), 'matches': matches}, ensure_ascii=False, indent=2))
