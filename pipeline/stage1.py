"""
Stage 1 — Multi-Strategy Deduplication Pipeline
----------------------------------------------

This script performs *balanced, multi-stage deduplication* on a Stage-0 cleaned
JSONL dataset. The goal is to remove exact, near-duplicate, template-based, and
semantic duplicates while keeping the highest-quality representative example
from each detected cluster.

It uses 5 complementary strategies:

1. **Exact Deduplication**
   - Normalize text (lowercase, remove punctuation, collapse whitespace)
   - Compute SHA-256 hash for each normalized instruction
   - All identical hashes are clustered; one representative is kept, rest removed.

2. **Near-Duplicate Deduplication (Shingling + Jaccard)**
   - Convert instructions into k-shingles (default: 5-grams)
   - Use MinHash + LSH (if datasketch is installed) OR fallback to brute-force
   - Cluster items with Jaccard similarity above threshold (default: 0.80–0.85)

3. **Template-Level Deduplication**
   - Mask numbers (“<NUM>”) and other template-like patterns
   - Group instructions that share the same masked template
   - This captures variations such as:
       “Write a summary for report 1234”
       “Write a summary for report 8348”
     which represent the same instruction pattern.

4. **Semantic Deduplication (Embedding-based)**
   - Encode each instruction using Sentence-Transformers
   - Compute cosine similarities using Faiss, sklearn, or brute-force
   - Cluster instructions whose embeddings exceed a similarity threshold
     (default: 0.92-0.95)
   - Keeps only the most informative/longest output.

5. **Cluster Resolution**
   - For every cluster produced above, pick a representative example based on:
       - Longest `output` field (proxy for information richness)
   - All others are removed unless already removed by stronger criteria.

Outputs created in `--outdir`:
--------------------------------
cleaned_stage1.jsonl             → final deduplicated dataset  
stage1_clusters_exact.json       → exact-duplicate clusters  
stage1_clusters_near.json        → near-duplicate clusters  
stage1_clusters_template.json    → template-based clusters  
stage1_clusters_semantic.json    → semantic-similar clusters  
dedup_report.json                → summary (kept/removed/cluster counts)

How to run:
--------------------------------
python pipeline/stage1_dedup_minhash.py \
    --input outputs/stage0_clean.jsonl \
    --outdir outputs/stage1_dedup \
    --lsh-threshold 0.80 \
    --semantic-threshold 0.92

Notes:
--------------------------------
- Works without optional libraries (datasketch / sentence-transformers /
  faiss / sklearn), gracefully degrading to simpler methods when necessary.
- The script is optimized for large datasets and avoids O(n²) operations
  unless no accelerated backends are available.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except Exception:
    _HAS_DATASKETCH = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# -------------------- Normalization utils --------------------
NON_ALPHANUM_RE = re.compile(r'[^\w\s]')
MULTI_WS_RE = re.compile(r'\s+')
NUM_RE = re.compile(r"\b\d{1,4}(?:[-,\/]?\d{1,4})*\b")

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = NON_ALPHANUM_RE.sub(' ', s)
    s = MULTI_WS_RE.sub(' ', s)
    s = s.strip()
    return s

def mask_template(s: str) -> str:
    # replace numbers with <NUM> and sequences of uppercase words (likely orgs) with <ORG>
    t = NUM_RE.sub('<NUM>', s)
    # crude org masking: Capitalized tokens separated by spaces in original string
    # For mask we operate on original string
    return t

# Exact dedupe
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

# Shingling & Jaccard
def shingles(text: str, k: int = 5) -> set:
    tokens = text.split()
    if len(tokens) < k:
        return set(tokens)
    return set(' '.join(tokens[i:i+k]) for i in range(len(tokens)-k+1))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = a.intersection(b)
    union = a.union(b)
    return len(inter) / len(union)
# MinHash helpers (datasketch)
def make_minhash_from_shingles(shs: set, num_perm: int = 128) -> 'MinHash':
    m = MinHash(num_perm=num_perm)
    for sh in shs:
        m.update(sh.encode('utf-8'))
    return m

# Semantic helpers
def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    if not _HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError('sentence-transformers is not installed')
    return SentenceTransformer(model_name)

# Clustering & Resolution
def pick_representative(cluster_indices: List[int], records: List[Dict[str, Any]]) -> int:
    # default: pick index with longest output string
    best_idx = cluster_indices[0]
    best_len = len(records[best_idx]['output'] if records[best_idx].get('output') else '')
    for idx in cluster_indices[1:]:
        l = len(records[idx].get('output',''))
        if l > best_len:
            best_idx = idx
            best_len = l
    return best_idx

# Main dedupe pipeline
def stage1_dedupe(
    records: List[Dict[str, Any]],
    outdir: Path,
    lsh_jaccard_threshold: float = 0.85,
    semantic_sim_threshold: float = 0.95,
    model_name: str = 'all-MiniLM-L6-v2',
    shingle_k: int = 5,
    minhash_perm: int = 128,
    verbose: bool = True
):
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    norm_instr = [normalize_text(r.get('instruction','')) for r in records]
    # Exact dedupe by normalized instruction sha
    exact_map = defaultdict(list)
    exact_hashes = []
    for i, t in enumerate(norm_instr):
        h = sha256_hex(t)
        exact_map[h].append(i)
        exact_hashes.append(h)
    exact_clusters = [v for v in exact_map.values() if len(v) > 1]
    # Mark exact duplicates
    is_removed = [False]*n
    for cluster in exact_clusters:
        # keep one representative
        rep = pick_representative(cluster, records)
        for idx in cluster:
            if idx != rep:
                is_removed[idx] = True

    # Near-duplicate via MinHash+LSH (if available), else brute-force Jaccard on shingles
    shingle_sets = [shingles(t, k=shingle_k) for t in norm_instr]
    near_clusters = []
    if _HAS_DATASKETCH:
        if verbose: print('Using datasketch MinHash + LSH')
        lsh = MinHashLSH(threshold=lsh_jaccard_threshold, num_perm=minhash_perm)
        minhashes = []
        for i, sh in enumerate(shingle_sets):
            m = make_minhash_from_shingles(sh, num_perm=minhash_perm)
            minhashes.append(m)
            lsh.insert(str(i), m)
        visited = set()
        for i, m in enumerate(minhashes):
            if i in visited: continue
            result = lsh.query(m)
            idxs = sorted([int(x) for x in result])
            if len(idxs) > 1:
                near_clusters.append(idxs)
                visited.update(idxs)
    else:
        if verbose: print('datasketch not installed; using brute-force Jaccard (O(n^2))')
        seen = set()
        for i in range(n):
            if i in seen: continue
            cluster = [i]
            for j in range(i+1, n):
                if j in seen: continue
                sim = jaccard(shingle_sets[i], shingle_sets[j])
                if sim >= lsh_jaccard_threshold:
                    cluster.append(j)
                    seen.add(j)
            if len(cluster) > 1:
                near_clusters.append(cluster)
                seen.update(cluster)
    # For near clusters, remove non-representative unless already removed by exact
    for cluster in near_clusters:
        rep = pick_representative(cluster, records)
        for idx in cluster:
            if idx != rep:
                is_removed[idx] = True
    # Template-based clusters (mask numbers and short tokens)
    templates = defaultdict(list)
    for i, orig in enumerate(records):
        templ = mask_template(orig.get('instruction',''))
        templates[templ].append(i)
    template_clusters = [v for v in templates.values() if len(v) > 1]
    for cluster in template_clusters:
        rep = pick_representative(cluster, records)
        for idx in cluster:
            if idx != rep:
                # only mark if not already removed by stronger checks
                if not is_removed[idx]:
                    is_removed[idx] = True
    # Semantic dedupe using embeddings
    semantic_clusters = []
    if _HAS_SENTENCE_TRANSFORMERS:
        if verbose: print('Computing embeddings for semantic dedupe...')
        model = load_embedding_model(model_name)
        texts = [records[i].get('instruction','') for i in range(n)]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if _HAS_FAISS:
            if verbose: print('Using faiss for approximate NN search')
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            _, nn_idx = index.search(embeddings, 10)  # top-10
            visited = set()
            for i in range(n):
                if i in visited: continue
                cluster = [i]
                for j in nn_idx[i]:
                    if j != i:
                        # cosine similarity since normalized
                        score = float((embeddings[i] @ embeddings[j]) / ( (embeddings[i].sum()*0+1) ))
                        # note: since embeddings normalized, dot product is cosine
                        score = float(np.dot(embeddings[i], embeddings[j])) if False else float((embeddings[i] @ embeddings[j]))
                        if score >= semantic_sim_threshold:
                            cluster.append(int(j))
                if len(cluster) > 1:
                    semantic_clusters.append(sorted(set(cluster)))
                    visited.update(cluster)
        elif _HAS_SKLEARN:
            if verbose: print('Using sklearn NearestNeighbors for semantic dedupe')
            nn = NearestNeighbors(n_neighbors=min(10, n), metric='cosine').fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)
            visited = set()
            for i in range(n):
                if i in visited: continue
                cluster = [i]
                for dist, j in zip(distances[i], indices[i]):
                    # cosine distance -> similarity = 1 - dist
                    sim = 1.0 - dist
                    if sim >= semantic_sim_threshold:
                        cluster.append(int(j))
                if len(cluster) > 1:
                    semantic_clusters.append(sorted(set(cluster)))
                    visited.update(cluster)
        else:
            if verbose: print('No sklearn/faiss available: running brute-force embedding similarity (O(n^2))')
            import numpy as np
            emb = embeddings
            visited = set()
            for i in range(n):
                if i in visited: continue
                cluster = [i]
                for j in range(i+1, n):
                    sim = float(np.dot(emb[i], emb[j]) / ( (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j])) ))
                    if sim >= semantic_sim_threshold:
                        cluster.append(j)
                        visited.add(j)
                if len(cluster) > 1:
                    semantic_clusters.append(cluster)
                    visited.update(cluster)
        # mark semantic duplicates for removal (if not already removed)
        for cluster in semantic_clusters:
            rep = pick_representative(cluster, records)
            for idx in cluster:
                if idx != rep and not is_removed[idx]:
                    is_removed[idx] = True
    else:
        if verbose: print('sentence-transformers not installed; skipping semantic dedupe')
    # Prepare outputs
    cleaned = []
    idx_map = []
    removed_idxs = []
    for i, rec in enumerate(records):
        if not is_removed[i]:
            cleaned.append(rec)
            idx_map.append(i)
        else:
            removed_idxs.append(i)
    # Save cleaned dataset
    cleaned_path = outdir / 'cleaned_stage1.jsonl'
    with cleaned_path.open('w', encoding='utf-8') as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    def save_clusters(path: Path, clusters: List[List[int]]):
        with path.open('w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=2)
    save_clusters(outdir / 'stage1_clusters_exact.json', exact_clusters)
    save_clusters(outdir / 'stage1_clusters_near.json', near_clusters)
    save_clusters(outdir / 'stage1_clusters_template.json', template_clusters)
    save_clusters(outdir / 'stage1_clusters_semantic.json', semantic_clusters)
    report = {
        'total': n,
        'kept': len(cleaned),
        'removed': len(removed_idxs),
        'exact_clusters': len(exact_clusters),
        'near_clusters': len(near_clusters),
        'template_clusters': len(template_clusters),
        'semantic_clusters': len(semantic_clusters)
    }
    with (outdir / 'dedup_report.json').open('w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2)
    return {
        'cleaned_path': str(cleaned_path),
        'report': report,
        'removed_indices': removed_idxs
    }
# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description='Stage 1 Deduplication (Balanced)')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file (from Stage 0 pass)')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory for stage1')
    parser.add_argument('--lsh-threshold', type=float, default=0.80, help='Jaccard threshold for LSH/shingle dedupe')
    parser.add_argument('--semantic-threshold', type=float, default=0.92, help='cosine threshold for semantic dedupe')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='sentence-transformers model name')
    parser.add_argument('--max-rows', type=int, default=None, help='optional: process only N rows')
    args = parser.parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    if not input_path.exists():
        print('Input file not found:', input_path)
        return
    records = []
    with input_path.open('r', encoding='utf-8') as inf:
        for i, line in enumerate(inf):
            if args.max_rows and i >= args.max_rows:
                break
            line = line.strip()
            if not line: continue
            try:
                j = json.loads(line)
                # keep only instruction,input,output keys (and preserve extra)
                records.append({'instruction': j.get('instruction',''), 'input': j.get('input',''), 'output': j.get('output',''), **{k:v for k,v in j.items() if k not in ['instruction','input','output']}})
            except Exception as e:
                print('Skipping invalid JSON line', i, e)
    result = stage1_dedupe(records, outdir, lsh_jaccard_threshold=args.lsh_threshold, semantic_sim_threshold=args.semantic_threshold, model_name=args.model)
    print('Stage 1 complete')
    print(result['report'])
if __name__ == '__main__':
    main()











