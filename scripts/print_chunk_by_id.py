import pickle, json
from pathlib import Path

semantic_pkl = Path('batches') / 'my_policies_semantic' / 'faiss_index' / 'index.pkl'
baseline_pkl = Path('batches') / 'my_policies' / 'faiss_index' / 'index.pkl'

def load_index(p):
    with open(p,'rb') as f:
        d = pickle.load(f)
    return d.get('chunks',[]), d.get('metadata',[])

sc_chunks, sc_meta = load_index(semantic_pkl)
b_chunks, b_meta = load_index(baseline_pkl)

interest_sem = ['p1-13','p1-14','p1-10','p1-31','p1-32']
interest_base = ['p4-0']

out = {'semantic':[], 'baseline':[]}
for cid in interest_sem:
    found = False
    for c,m in zip(sc_chunks, sc_meta):
        if m.get('chunk_id') == cid:
            out['semantic'].append({'chunk_id': cid, 'filename': m.get('filename'), 'page': m.get('page_number'), 'chunking_strategy': m.get('chunking_strategy'), 'extraction_method': m.get('extraction_method'), 'content': c[:2000]})
            found = True
            break
    if not found:
        out['semantic'].append({'chunk_id': cid, 'found': False})

for cid in interest_base:
    found = False
    for c,m in zip(b_chunks, b_meta):
        if m.get('chunk_id') == cid:
            out['baseline'].append({'chunk_id': cid, 'filename': m.get('filename'), 'page': m.get('page_number'), 'chunking_strategy': m.get('chunking_strategy'), 'extraction_method': m.get('extraction_method'), 'content': c[:2000]})
            found = True
            break
    if not found:
        out['baseline'].append({'chunk_id': cid, 'found': False})

with open('evaluation/results/chunk_contents_inspection.json','w',encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('Wrote evaluation/results/chunk_contents_inspection.json')
