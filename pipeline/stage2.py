"""
Stage 2 — Semantic Quality & Label-Issue Detection
--------------------------------------------------

This stage performs semantic filtering on Stage-1 output to remove:
- semantic outliers
- noisy / low-quality samples
- mislabeled or contradictory records
- out-of-distribution (OOD) examples

Three evidence signals are used:

1. KNN Embedding Outlier Detection
   - Embeddings computed for (instruction + output)
   - Local neighborhood similarity used to score outliers
   - Produces knn_score ∈ [0,1]; higher = more anomalous

2. Cluster-Based Pseudo-Labels + Cleanlab
   - Agglomerative clustering → pseudo-labels
   - Classifier predicts cluster IDs
   - Cleanlab flags label issues and cluster inconsistencies

3. Cleanlab OOD Detection
   - Uses embedding distance to detect far-off, noisy, or irrelevant samples

Removal Rule
------------
Each record contributes to a combined removal_score:
  +0.5 strong KNN outlier
  +0.3 moderate outlier
  +0.4 Cleanlab label issue
  +0.3 OOD > 95th percentile

Record removed if: removal_score ≥ 0.7

Outputs (written to --outdir)
-----------------------------
- stage2_clean.jsonl        → cleaned dataset
- stage2_flagged.csv        → all flagged entries
- stage2_report.json        → summary stats
- (optional) embeddings.npy → if --save-embeddings used

Run Example
-----------
python pipeline/stage2_cleanlab.py \
  --input outputs/stage1_dedup/cleaned_stage1.jsonl \
  --outdir outputs/stage2_semantic \
  --save-embeddings \
  --remove-threshold 0.85
"""


from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    import cleanlab
    from cleanlab.filter import find_label_issues
    from cleanlab.outlier import OutOfDistribution
    _HAS_CLEANLAB = True
except Exception:
    _HAS_CLEANLAB = False

import math


def load_jsonl(path: Path, max_rows: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def compute_embeddings(records: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    if not _HAS_ST:
        raise RuntimeError('sentence-transformers not installed')
    model = SentenceTransformer(model_name)
    texts = [(r.get('instruction', '') + ' ' + (r.get('output', '') or '')).strip() for r in records]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def knn_outlier_scores(emb: np.ndarray, k: int = 5) -> np.ndarray:
    if not _HAS_SK:
        sims = cosine_similarity(emb, emb)
        np.fill_diagonal(sims, -1)
        mean_topk = np.mean(np.sort(sims, axis=1)[:, -k:], axis=1)
        scores = 1.0 - ((mean_topk - mean_topk.min()) /
                        max(1e-12, (mean_topk.max() - mean_topk.min())))
        return scores

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(emb)), metric='cosine')
    nn.fit(emb)
    distances, _ = nn.kneighbors(emb)
    sims = 1.0 - distances[:, 1:]
    mean_sim = sims.mean(axis=1)
    minv, maxv = float(mean_sim.min()), float(mean_sim.max())
    if maxv - minv < 1e-12:
        return np.zeros(len(mean_sim))
    return 1.0 - (mean_sim - minv) / (maxv - minv)


def pseudo_labels_from_clusters(emb: np.ndarray, n_clusters: int = 30) -> np.ndarray:
    if not _HAS_SK:
        n = emb.shape[0]
        rng = np.random.default_rng(42)
        return rng.integers(0, min(n_clusters, max(2, n // 10)), size=n)

    n = emb.shape[0]
    k = min(max(2, n_clusters), max(2, n // 5))
    clustering = AgglomerativeClustering(n_clusters=k)
    return clustering.fit_predict(emb)


def find_label_issues_with_cleanlab(emb: np.ndarray, labels: np.ndarray) -> List[int]:
    if not _HAS_CLEANLAB:
        return []
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        pred_probs = cross_val_predict(
            clf, emb, labels,
            cv=min(5, max(2, len(labels) // 50)),
            method='predict_proba'
        )
    except Exception:
        pred_probs = cross_val_predict(clf, emb, labels, cv=2, method='predict_proba')

    try:
        return list(find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        ))
    except Exception:
        try:
            issues = find_label_issues(labels=labels, pred_probs=pred_probs)
            return list(issues)
        except Exception:
            return []


def cleanlab_ood_scores(emb: np.ndarray) -> np.ndarray:
    if not _HAS_CLEANLAB:
        return np.zeros(emb.shape[0])
    try:
        ood = OutOfDistribution()
        return np.array(ood.fit_score(features=emb))
    except Exception:
        return np.zeros(emb.shape[0])


def stage2_process(records: List[Dict[str, Any]], outdir: Path,
                   max_rows: int | None = None, save_embeddings: bool = False,
                   remove_threshold: float = 0.85):

    outdir.mkdir(parents=True, exist_ok=True)
    n = len(records)

    if not _HAS_ST:
        raise RuntimeError('sentence-transformers required')

    emb = compute_embeddings(records)
    if save_embeddings:
        np.save(outdir / 'stage2_embeddings.npy', emb)

    knn_scores = knn_outlier_scores(emb, k=5)
    labels = pseudo_labels_from_clusters(emb, n_clusters=max(10, min(100, n // 10)))

    if _HAS_CLEANLAB and _HAS_SK:
        try:
            label_issue_idxs = find_label_issues_with_cleanlab(emb, labels)
        except Exception:
            label_issue_idxs = []
    else:
        label_issue_idxs = []

    ood_scores = cleanlab_ood_scores(emb) if _HAS_CLEANLAB else np.zeros(n)

    flagged_df_rows = []
    for i in range(n):
        reasons = []
        if knn_scores[i] > 0.9:
            reasons.append('knn_outlier')
        elif knn_scores[i] > 0.75:
            reasons.append('knn_suspicious')
        if i in label_issue_idxs:
            reasons.append('label_issue_cleanlab')
        if ood_scores[i] > np.quantile(ood_scores, 0.9) and ood_scores[i] > 0:
            reasons.append('ood_high')
        if reasons:
            flagged_df_rows.append({
                'index': i,
                'instruction': records[i].get('instruction', '')[:200],
                'output': records[i].get('output', '')[:200],
                'knn_score': float(knn_scores[i]),
                'ood_score': float(ood_scores[i]),
                'label_issue': i in label_issue_idxs,
                'reasons': ';'.join(reasons)
            })

    # to_remove = set()
    # for r in flagged_df_rows:
    #     if r['label_issue'] or r['knn_score'] > remove_threshold:
    #         to_remove.add(r['index'])
    to_remove = set()
    for r in flagged_df_rows:
        removal_score = 0.0

        if r['knn_score'] > 0.90:
            removal_score += 0.5
        elif r['knn_score'] > 0.80:
            removal_score += 0.3

        if r['label_issue']:
            removal_score += 0.4

        if r['ood_score'] > np.quantile(ood_scores, 0.95):
            removal_score += 0.3

        # Remove if combined evidence is strong
        if removal_score >= 0.7:  # Requires strong multi-signal agreement
            to_remove.add(r['index'])

    cleaned = [rec for i, rec in enumerate(records) if i not in to_remove]

    save_jsonl(cleaned, outdir / 'stage2_clean.jsonl')
    pd.DataFrame(flagged_df_rows).to_csv(outdir / 'stage2_flagged.csv', index=False)

    report = {
        'total': n,
        'flagged_total': len(flagged_df_rows),
        'removed_final': len(to_remove),
        'kept': len(cleaned)
    }

    with (outdir / 'stage2_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return {
        'report': report,
        'flagged_indices': [r['index'] for r in flagged_df_rows],
        'removed_indices': sorted(list(to_remove))
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 2 - Cleanlab Semantic Noise & Label-Issue Detection')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file from Stage1 cleaned dataset')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory for Stage2')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--save-embeddings', action='store_true')
    parser.add_argument('--remove-threshold', type=float, default=0.85)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    if not input_path.exists():
        print('Input not found:', input_path)
        return

    records = load_jsonl(input_path, max_rows=args.max_rows)
    print('Loaded', len(records), 'records')

    res = stage2_process(
        records,
        outdir,
        max_rows=args.max_rows,
        save_embeddings=args.save_embeddings,
        remove_threshold=args.remove_threshold
    )
    print('Stage 2 complete')
    print(res['report'])


if __name__ == '__main__':
    main()
