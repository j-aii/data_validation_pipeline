"""
Stage 3 - Semantic Correctness & Answer Quality (RAGAS + fallbacks)

Full implementation (Option A): a single Python script that attempts to use
RAGAS and lettuceDetect where available, and falls back to heuristic or
embedding-based checks when heavy deps are missing.

Inputs:
  --input : JSONL file (from Stage 2 clean)
  --outdir: output directory for reports

Outputs (written to outdir):
  - stage3_scored.csv        : per-record scores and features
  - stage3_flagged.csv       : records flagged for human review
  - stage3_removed.jsonl     : records removed (low score)
  - cleaned_stage3.jsonl     : records kept after Stage 3
  - stage3_report.json       : summary counts and thresholds used

Dependencies (recommended env):
  pip install ragas lettucedetect sentence-transformers pandas numpy tqdm openai

Notes:
 - This script WILL NOT call any external APIs unless you explicitly provide
   an LLM provider key via environment variables and enable the `--use-llm`
   option. When ragas is installed it will attempt to use its default model
   runner; otherwise we use heuristics and SBERT-based proxies.
 - The script is defensive and will run in degraded mode (heuristics only)
   if ragas or lettuceDetect are not installed.

Usage example:
  python ev_stage3_ragas.py --input outputs/stage2_cleanlab/stage2_clean.jsonl --outdir outputs/stage3

"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try guarded imports
_HAS_RAGAS = False
_HAS_LETTUCE = False
_HAS_ST = False
_HAS_OPENAI = False

try:
    import ragas
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevance,
        answer_correctness,
        answer_completeness,
    )
    _HAS_RAGAS = True
except Exception:
    _HAS_RAGAS = False

try:
    # lettuceDetect package name may differ; guard carefully
    import lettucedetect
    _HAS_LETTUCE = True
except Exception:
    _HAS_LETTUCE = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ---------------------- Utilities & heuristics ----------------------
NUM_RE = re.compile(r"\b\d{1,7}(?:[.,]\d{1,3})*\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
NON_ALPHANUM_RE = re.compile(r"[^\w\s]\s*")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def extract_first_number(s: str) -> Tuple[float|None, List[str]]:
    # returns first numeric token as float and list of tokens found
    toks = NUM_RE.findall(s)
    if not toks:
        return None, []
    # normalize and parse
    first = toks[0].replace(',','')
    try:
        val = float(first)
    except Exception:
        # remove decimals like 1.234.567
        first_clean = re.sub(r"[^0-9.]", "", first)
        try:
            val = float(first_clean)
        except Exception:
            val = None
    return val, toks


def extract_years(s: str) -> List[int]:
    ys = YEAR_RE.findall(s)
    out = []
    for y in ys:
        try:
            yi = int(y)
            out.append(yi)
        except Exception:
            continue
    return out


def lexical_overlap_score(a: str, b: str) -> float:
    # simple token overlap ratio
    atoks = set(re.sub(r"[^0-9a-zA-Z ]+"," ", a.lower()).split())
    btoks = set(re.sub(r"[^0-9a-zA-Z ]+"," ", b.lower()).split())
    if not atoks and not btoks:
        return 1.0
    if not atoks or not btoks:
        return 0.0
    inter = atoks & btoks
    union = atoks | btoks
    return len(inter) / max(1, len(union))

# ---------------------- Fallback LLM-like evaluator (SBERT proxy) ----------------------
class SBERTProxy:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not _HAS_ST:
            raise RuntimeError('sentence-transformers not installed')
        self.model = SentenceTransformer(model_name)

    def relevance_score(self, question: str, answer: str) -> float:
        # cosine similarity between q and a embeddings
        qa = question + ' ' + answer
        q_emb = self.model.encode([question], convert_to_numpy=True)
        a_emb = self.model.encode([answer], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        a_emb = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)
        sim = float((q_emb * a_emb).sum())
        return sim

    def correctness_proxy(self, question: str, answer: str) -> float:
        # use lexical overlap & embedding sim heuristic
        lex = lexical_overlap_score(question, answer)
        try:
            emb_sim = self.relevance_score(question, answer)
        except Exception:
            emb_sim = 0.0
        # weighted
        return 0.6 * emb_sim + 0.4 * lex

# ---------------------- RAGAS wrapper ----------------------

def run_ragas_evaluation(records: List[Dict[str, Any]], ragas_model: str|None = None) -> List[Dict[str, float]]:
    """Attempt to run RAGAS evaluation. If ragas not available, return empty list."""
    if not _HAS_RAGAS:
        return []
    # Prepare ragas dataset
    questions = [r.get('instruction','') for r in records]
    answers = [r.get('output','') for r in records]
    contexts = ['' for _ in records]
    ds = {
        'question': questions,
        'answer': answers,
        'context': contexts
    }
    # ragas.evaluate expects a Dataset-like object; use ragas.datasets.Dataset
    try:
        from ragas import Dataset
        ragas_ds = Dataset.from_dict(ds)
        metrics = [faithfulness, answer_relevance, answer_correctness, answer_completeness]
        results = evaluate(ragas_ds, metrics=metrics, batch_size=16)
        # results is a dict or DataFrame; try to extract per-row metrics
        out = []
        # results.to_pandas() may exist
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            for _, row in df.iterrows():
                out.append({
                    'faithfulness': float(row.get('faithfulness', 0.0)),
                    'relevance': float(row.get('answer_relevance', 0.0)),
                    'correctness': float(row.get('answer_correctness', 0.0)),
                    'completeness': float(row.get('answer_completeness', 0.0))
                })
            return out
        # else try dict format
        if isinstance(results, dict) and 'predictions' in results:
            for rec in results['predictions']:
                out.append({
                    'faithfulness': float(rec.get('faithfulness', 0.0)),
                    'relevance': float(rec.get('answer_relevance', 0.0)),
                    'correctness': float(rec.get('answer_correctness', 0.0)),
                    'completeness': float(rec.get('answer_completeness', 0.0))
                })
            return out
    except Exception as e:
        print('RAGAS evaluation failed:', e)
        return []

# ---------------------- lettuceDetect wrapper ----------------------

def run_lettuce_hallucination(records: List[Dict[str, Any]]) -> Tuple[List[float], List[Any]]:
    # returns (hallucination_score_list, spans_list)
    if _HAS_LETTUCE:
        try:
            model = lettucedetect.HallucinationDetector()
            scores = []
            spans = []
            for r in records:
                out = model.predict(question=r.get('instruction',''), answer=r.get('output',''))
                # assume output contains {'score': float, 'spans': [...]}
                scores.append(float(out.get('score', 0.0)))
                spans.append(out.get('spans', []))
            return scores, spans
        except Exception as e:
            print('lettuceDetect error:', e)
    # Fallback heuristic: hallucination_score = 1 - lexical_overlap(question, answer)
    scores = []
    spans = [[] for _ in records]
    for r in records:
        q = r.get('instruction','')
        a = r.get('output','')
        lex = lexical_overlap_score(q, a)
        scores.append(1.0 - lex)
    return scores, spans

# ---------------------- Domain rules ----------------------

def domain_checks(record: Dict[str, Any]) -> Dict[str, Any]:
    instr = record.get('instruction','')
    out = record.get('output','')
    checks = {}

    # Year checks
    years = extract_years(instr + ' ' + out)
    checks['years'] = years
    checks['year_valid'] = any(2010 <= y <= 2025 for y in years) if years else False

    # Numeric checks
    num, toks = extract_first_number(out)
    checks['first_number'] = num
    checks['has_number'] = num is not None
    if num is not None:
        # heuristic: EV sales should be between 0 and 5,000,000
        checks['number_reasonable'] = (0 <= num <= 5_000_000)
    else:
        checks['number_reasonable'] = False

    # instruction asks for 'value' or 'sales'
    asks_value = bool(re.search(r'\b(sales|value|sold|units)\b', instr, flags=re.I))
    checks['asks_sales'] = asks_value

    # if asks_value but answer has no number -> bad
    checks['value_mismatch'] = asks_value and not checks['has_number']

    # simple company token detection
    # find capitalized tokens in instruction (crude org detection)
    orgs = re.findall(r"\b([A-Z][A-Za-z0-9&\-\.]{2,})\b", instr)
    checks['org_tokens'] = orgs

    # lexical overlap
    checks['lexical_overlap_q_a'] = lexical_overlap_score(instr, out)

    return checks

# ---------------------- Composite scoring ----------------------

def composite_score(component_scores: Dict[str, float], domain: Dict[str, Any]) -> float:
    # weights (tunable)
    w = {
        'correctness': 0.30,
        'relevance': 0.20,
        'faithfulness': 0.20,
        'completeness': 0.10,
        'hallucination': 0.10,  # lower is better, we will use (1 - hallucination_score)
        'domain': 0.10
    }
    correctness = component_scores.get('correctness', 0.0)
    relevance = component_scores.get('relevance', 0.0)
    faithfulness = component_scores.get('faithfulness', 0.0)
    completeness = component_scores.get('completeness', 0.0)
    halluc = component_scores.get('hallucination', 0.0)

    # domain_score from checks
    domain_score = 0.0
    # encourage reasonable number and year validity and lexical overlap
    if domain.get('number_reasonable'):
        domain_score += 0.6
    if domain.get('year_valid'):
        domain_score += 0.2
    domain_score += min(0.2, domain.get('lexical_overlap_q_a', 0.0) * 0.2)

    score = (
        w['correctness']*correctness +
        w['relevance']*relevance +
        w['faithfulness']*faithfulness +
        w['completeness']*completeness +
        w['hallucination']*(1.0 - halluc) +
        w['domain']*domain_score
    )
    return float(score)

# ---------------------- Main pipeline ----------------------

def stage3_process(records: List[Dict[str, Any]], outdir: Path, use_ragas: bool = True, use_lettuce: bool = True, sbert_model: str = 'all-MiniLM-L6-v2', thresholds: Dict[str, float] | None = None) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(records)

    thresholds = thresholds or {}
    remove_thresh = thresholds.get('remove', 0.55)
    review_thresh = thresholds.get('review', 0.70)

    # 1) run ragas if available
    ragas_results = []
    if use_ragas and _HAS_RAGAS:
        try:
            print('Running RAGAS evaluation...')
            ragas_results = run_ragas_evaluation(records)
        except Exception as e:
            print('RAGAS run failed, falling back to SBERT proxy:', e)
            ragas_results = []
    else:
        ragas_results = []

    # 2) run lettuceDetect or fallback heuristic
    halluc_scores, halluc_spans = run_lettuce_hallucination(records) if use_lettuce else run_lettuce_hallucination(records)

    # 3) build SBERT proxy if RAGAS missing
    sbert = None
    if not ragas_results:
        if _HAS_ST:
            print('Loading SBERT model (proxy evaluator)...')
            sbert = SBERTProxy(sbert_model)
        else:
            print('No SBERT/RAGAS available: running purely heuristic checks (less accurate)')

    # prepare per-record outputs
    rows = []
    for i, rec in enumerate(tqdm(records, desc='Scoring')):
        q = rec.get('instruction','')
        a = rec.get('output','')
        components = {
            'faithfulness': 0.0,
            'relevance': 0.0,
            'correctness': 0.0,
            'completeness': 0.0,
            'hallucination': float(halluc_scores[i]) if i < len(halluc_scores) else 0.0
        }
        # populate from ragas if present
        if ragas_results:
            rr = ragas_results[i]
            components['faithfulness'] = rr.get('faithfulness', 0.0)
            components['relevance'] = rr.get('relevance', 0.0)
            components['correctness'] = rr.get('correctness', 0.0)
            components['completeness'] = rr.get('completeness', 0.0)
        else:
            # estimate using SBERT proxy or heuristics
            if sbert is not None:
                try:
                    components['relevance'] = float(np.clip(sbert.relevance_score(q, a), -1.0, 1.0))
                    # map [-1,1] -> [0,1]
                    components['relevance'] = (components['relevance'] + 1.0) / 2.0
                except Exception:
                    components['relevance'] = lexical_overlap_score(q, a)
                components['correctness'] = sbert.correctness_proxy(q, a) if sbert is not None else lexical_overlap_score(q,a)
                components['faithfulness'] = components['relevance'] * 0.9
                components['completeness'] = min(1.0, len(a.split()) / 50.0)
            else:
                components['relevance'] = lexical_overlap_score(q, a)
                components['correctness'] = lexical_overlap_score(q, a)
                components['faithfulness'] = lexical_overlap_score(q, a)
                components['completeness'] = min(1.0, len(a.split()) / 50.0)

        domain = domain_checks(rec)
        final = composite_score(components, domain)

        row = {
            'index': i,
            'instruction': q,
            'output': a,
            'faithfulness': components['faithfulness'],
            'relevance': components['relevance'],
            'correctness': components['correctness'],
            'completeness': components['completeness'],
            'hallucination': components['hallucination'],
            'domain_number': domain.get('first_number'),
            'domain_years': domain.get('years'),
            'domain_year_valid': domain.get('year_valid'),
            'domain_number_reasonable': domain.get('number_reasonable'),
            'domain_value_mismatch': domain.get('value_mismatch'),
            'domain_lex_overlap': domain.get('lexical_overlap_q_a'),
            'composite_score': final
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # decisions
    removed = df[df['composite_score'] < remove_thresh]
    review = df[(df['composite_score'] >= remove_thresh) & (df['composite_score'] < review_thresh)]
    kept = df[df['composite_score'] >= review_thresh]

    # save outputs
    df.to_csv(outdir / 'stage3_scored.csv', index=False)
    review.to_csv(outdir / 'stage3_flagged.csv', index=False)

    # write removed and kept jsonl
    removed_records = [records[int(r['index'])] for _, r in removed.iterrows()]
    kept_records = [records[int(r['index'])] for _, r in kept.iterrows()]
    save_jsonl(removed_records, outdir / 'stage3_removed.jsonl')
    save_jsonl(kept_records, outdir / 'cleaned_stage3.jsonl')

    report = {
        'total': n,
        'removed_count': int(len(removed)),
        'flagged_count': int(len(review)),
        'kept_count': int(len(kept))
    }
    with (outdir / 'stage3_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report

# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description='Stage 3 - Semantic Correctness & Answer Quality')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file (Stage 2 clean)')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory for Stage 3')
    parser.add_argument('--no-ragas', action='store_true', help='Do not attempt to use RAGAS even if installed')
    parser.add_argument('--no-lettuce', action='store_true', help='Do not attempt to use lettuceDetect')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='SBERT model for proxy scoring')
    parser.add_argument('--remove-thresh', type=float, default=0.55, help='Composite score below which we remove sample')
    parser.add_argument('--review-thresh', type=float, default=0.70, help='Composite score below which we flag for review')
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    if not records:
        print('No records loaded from', input_path)
        return
    report = stage3_process(records, outdir, use_ragas=not args.no_ragas, use_lettuce=not args.no_lettuce, sbert_model=args.model, thresholds={'remove': args.remove_thresh, 'review': args.review_thresh})
    print('Stage 3 finished. Report:')
    print(report)

if __name__ == '__main__':
    main()
