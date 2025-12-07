"""
Stage 4 - Task-Specific Validation for EV Dataset

Purpose:
 - Validate domain-specific correctness for instruction->output QA pairs about EV .
 - Rule-based checks + embedding semantic checks + optional LLM checks.
 - Balanced mode: remove clear errors, flag borderline cases.

Inputs:
 - JSONL (instruction,input,output) - typically stage3 cleaned file

Outputs (written to --outdir):
 - stage4_scored.csv      (per-row scores & flags)
 - stage4_clean.jsonl     (kept rows)
 - stage4_flagged.jsonl   (need human review)
 - stage4_removed.jsonl   (removed rows)
 - stage4_report.json     (summary)

Usage:
 python stage4_ev_validate.py --input path/to/stage3_clean.jsonl --outdir outputs/stage4

Dependencies (recommended):
 pip install sentence-transformers python-dateutil pandas tqdm

Notes:
 - Optional LLM verification is supported via --use-llm but you must provide OPENAI_API_KEY (or change to your LLM client).
 - The script is defensive when optional libs are missing (it will fall back to lexical heuristics).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import math
import difflib
import statistics

# optional imports
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from dateutil import parser as dateparser
    _HAS_DATEUTIL = True
except Exception:
    _HAS_DATEUTIL = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------- Configurable canonical categories --------------------------
CANONICAL_CATEGORIES = [
    'LIGHT PASSENGER VEHICLE', 'LIGHT GOODS VEHICLE', 'LIGHT MOTOR VEHICLE',
    'HEAVY MOTOR VEHICLES', 'MEDIUM MOTOR VEHICLE', 'MEDIUM PASSENGER VEHICLE',
    'HEAVY PASSENGER VEHICLE', 'THREE WHEELER (NT)', 'THREE WHEELER (T)',
    'TWO WHEELER (NT)', 'TWO WHEELER (INVALID CARRIAGE)', 'FOUR WHEELER (INVALID CARRIAGE)',
    'OTHER THAN MENTIONED ABOVE'
]

# fuzzy match helper
def fuzzy_match_category(cat: str, choices: List[str], cutoff: float = 0.7) -> Tuple[str, float]:
    if not cat:
        return ('', 0.0)
    # normalize
    c = re.sub(r"[^A-Za-z0-9 ]+", ' ', cat).strip().upper()
    # exact
    if c in choices:
        return (c, 1.0)
    # difflib quick ratio
    ratios = [(choice, difflib.SequenceMatcher(None, c, choice).ratio()) for choice in choices]
    best, score = max(ratios, key=lambda x: x[1])
    if score >= cutoff:
        return (best, score)
    # try token overlap
    tokens = set(c.split())
    best2, best_score2 = '', 0.0
    for choice in choices:
        t = set(choice.split())
        ov = len(tokens & t) / max(1, len(tokens | t))
        if ov > best_score2:
            best2, best_score2 = choice, ov
    if best_score2 >= cutoff:
        return (best2, best_score2)
    return ('', max(score, best_score2))

# -------------------------- Regex and parsing utils --------------------------
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
NUM_RE = re.compile(r"\b\d{1,7}(?:[.,]\d{1,3})?\b")
DATE_LIKE_RE = re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b")

def extract_years(text: str) -> List[int]:
    ys = YEAR_RE.findall(text)
    out = []
    for y in ys:
        try:
            out.append(int(y))
        except Exception:
            continue
    return out

def extract_first_int(text: str) -> Tuple[int|None, List[str]]:
    toks = NUM_RE.findall(text)
    if not toks:
        return (None, [])
    # take first token, strip commas
    t = toks[0].replace(',', '')
    try:
        val = int(float(t))
        return (val, toks)
    except Exception:
        return (None, toks)

def try_parse_date(text: str) -> Tuple[str|None, bool]:
    # returns (ISO-ish string, success)
    if _HAS_DATEUTIL:
        try:
            dt = dateparser.parse(text, default=None)
            if dt:
                return (dt.date().isoformat(), True)
        except Exception:
            pass
    # fallback: look for dd/mm/yy like patterns
    m = DATE_LIKE_RE.search(text)
    if m:
        return (m.group(1), True)
    return (None, False)

# -------------------------- Heuristic checks --------------------------

def is_number_reasonable(n: int, category: str, max_global: int = 5_000_000) -> bool:
    if n is None:
        return False
    if n < 0:
        return False
    if n > max_global:
        return False
    # category-specific heuristics (light vehicles may be large, heavy small)
    cat = category or ''
    cat = cat.upper()
    if 'TWO WHEELER' in cat or 'TWO-WHEELER' in cat or 'TWO WHEELER' in cat:
        # two-wheeler counts can be high
        return n <= 500_0000  # very lenient
    if 'LIGHT' in cat or 'PASSENGER' in cat:
        return n <= 1_000_000
    # default
    return n <= 500_000

# lexical overlap
def lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(re.sub(r"[^0-9A-Za-z ]+", ' ', a.lower()).split())
    b_tokens = set(re.sub(r"[^0-9A-Za-z ]+", ' ', b.lower()).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

# -------------------------- Embedding similarity --------------------------
class EmbeddingScorer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if not _HAS_ST:
            raise RuntimeError('sentence-transformers not installed')
        self.model = SentenceTransformer(model_name)

    def similarity(self, a: str, b: str) -> float:
        v = self.model.encode([a, b], convert_to_numpy=True)
        a_v, b_v = v[0], v[1]
        # cosine
        denom = (np.linalg.norm(a_v) * np.linalg.norm(b_v))
        if denom == 0:
            return 0.0
        return float(np.dot(a_v, b_v) / denom)

# -------------------------- Optional LLM verifier --------------------------
def llm_verify(instruction: str, output: str, openai_model: str = 'gpt-4o-mini') -> Tuple[float, str]:
    """Return (score 0-1, reason). Requires OPENAI_API_KEY in env and openai installed."""
    if not _HAS_OPENAI:
        return (0.5, 'openai not available')
    prompt = (
        "You are a validator. Given a question and an answer about EV registration counts,\n"
        "return a JSON object with keys: verdict (PASS/FAIL/REVIEW), score (0-1), reason (brief).\n\n"
        f"QUESTION: {instruction}\nANSWER: {output}\n\nRespond strictly with JSON."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        text = resp['choices'][0]['message']['content']
        # try parse json from text
        import json as _json
        parsed = _json.loads(text)
        score = float(parsed.get('score', 0.5))
        reason = parsed.get('reason', '')
        return (score, reason)
    except Exception as e:
        return (0.5, f'llm error: {e}')

# -------------------------- Composite scoring & decision logic --------------------------

def compute_row_score(record: Dict[str,Any], emb_scorer: EmbeddingScorer|None, config: Dict[str, Any]) -> Dict[str, Any]:
    instr = record.get('instruction','') or ''
    out = record.get('output','') or ''

    # category extraction (attempt to parse from instruction)
    # heuristics: look for uppercase words sequences in instruction
    cat_match = re.findall(r"([A-Z][A-Z\s\-\(\)0-9]{3,})", instr)
    suggested_cat = ''
    suggested_score = 0.0
    if cat_match:
        # take the longest match
        cand = max(cat_match, key=len)
        suggested_cat, suggested_score = fuzzy_match_category(cand.strip(), CANONICAL_CATEGORIES, cutoff=0.6)

    # fallback: try to find any canonical token inside instruction
    if not suggested_cat:
        for choice in CANONICAL_CATEGORIES:
            if choice.lower() in instr.lower():
                suggested_cat = choice
                suggested_score = 1.0
                break

    # numeric extraction
    num, toks = extract_first_int(out)
    years = extract_years(instr + ' ' + out)
    date_text, date_ok = try_parse_date(instr + ' ' + out)

    # lexical overlap
    lex = lexical_overlap(instr, out)

    # embedding similarity
    emb_sim = None
    if emb_scorer is not None:
        try:
            emb_sim = emb_scorer.similarity(instr, out)
        except Exception:
            emb_sim = None

    # domain checks
    year_valid = any(1990 <= y <= 2030 for y in years) if years else False
    number_reasonable = is_number_reasonable(num, suggested_cat if suggested_cat else '')

    # rule-based verdicts
    rule_flags = {
        'missing_number': num is None,
        'multiple_numbers': len(toks) > 1 if toks else False,
        'malformed_year': any(len(str(y)) != 4 for y in years) if years else False,
        'bad_date_parse': not date_ok,
        'unknown_category': suggested_cat == ''
    }

    # compute component scores (0..1)
    comp = {}
    comp['lexical'] = float(lex)
    comp['emb_sim'] = float(emb_sim) if emb_sim is not None else comp['lexical']
    comp['year_valid'] = 1.0 if year_valid else 0.0
    comp['number_reasonable'] = 1.0 if number_reasonable else 0.0
    comp['category_confidence'] = float(suggested_score)

    # composite scoring weights for Balanced mode
    w = config.get('weights', {
        'emb_sim': 0.35,
        'lexical': 0.10,
        'year_valid': 0.20,
        'number_reasonable': 0.20,
        'category_confidence': 0.15
    })

    score = (
        w['emb_sim'] * comp['emb_sim'] +
        w['lexical'] * comp['lexical'] +
        w['year_valid'] * comp['year_valid'] +
        w['number_reasonable'] * comp['number_reasonable'] +
        w['category_confidence'] * comp['category_confidence']
    )

    # tighten: if missing number but instruction clearly asks for count -> penalize
    if rule_flags['missing_number'] and re.search(r'\b(number|count|how many|value|registrat)', instr, flags=re.I):
        score *= 0.35

    # if instruction asks for date but output gives number -> penalize
    if re.search(r'\b(on what date|what date|when)\b', instr, flags=re.I) and num is not None:
        score *= 0.5

    # final normalized score
    final_score = float(max(0.0, min(1.0, score)))

    return {
        'suggested_category': suggested_cat,
        'suggested_cat_score': suggested_score,
        'num': num,
        'num_tokens': toks,
        'years': years,
        'date_parsed': date_text,
        'rule_flags': rule_flags,
        'components': comp,
        'final_score': final_score
    }

# -------------------------- Main pipeline --------------------------

def stage4_process(records: List[Dict[str,Any]], outdir: Path, use_llm: bool = False, model_name: str = 'all-MiniLM-L6-v2', thresholds: Dict[str, float] | None = None) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(records)

    thresholds = thresholds or {'remove': 0.45, 'flag': 0.70}  # Balanced: remove <0.45, flag 0.45-0.70, keep >=0.70

    emb = None
    if _HAS_ST:
        try:
            emb = EmbeddingScorer(model_name)
        except Exception:
            emb = None

    rows = []
    for i, rec in enumerate(tqdm(records, desc='Stage4 validating')):
        instr = rec.get('instruction','')
        out = rec.get('output','')
        meta = compute_row_score(rec, emb, config={})

        llm_score = None
        llm_reason = ''
        if use_llm:
            s, r = llm_verify(instr, out)
            llm_score, llm_reason = s, r
            # blend LLM signal moderately
            meta['final_score'] = 0.7 * meta['final_score'] + 0.3 * llm_score

        verdict = 'keep'
        if meta['final_score'] < thresholds['remove']:
            verdict = 'remove'
        elif meta['final_score'] < thresholds['flag']:
            verdict = 'flag'

        row = {
            'index': i,
            'instruction': instr,
            'output': out,
            'suggested_category': meta['suggested_category'],
            'suggested_cat_score': meta['suggested_cat_score'],
            'num': meta['num'],
            'years': meta['years'],
            'date_parsed': meta['date_parsed'],
            'rule_flags': meta['rule_flags'],
            'components': meta['components'],
            'final_score': meta['final_score'],
            'llm_score': llm_score,
            'llm_reason': llm_reason,
            'verdict': verdict
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    removed_df = df[df['verdict'] == 'remove']
    flagged_df = df[df['verdict'] == 'flag']
    kept_df = df[df['verdict'] == 'keep']

    # save outputs
    df.to_csv(outdir / 'stage4_scored.csv', index=False)

    def save_jsonl_from_df(dframe: pd.DataFrame, path: Path):
        items = []
        for _, r in dframe.iterrows():
            # original record index maps to position in input
            items.append(records[int(r['index'])])
        with path.open('w', encoding='utf-8') as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + '\n')

    save_jsonl_from_df(kept_df, outdir / 'stage4_clean.jsonl')
    save_jsonl_from_df(flagged_df, outdir / 'stage4_flagged.jsonl')
    save_jsonl_from_df(removed_df, outdir / 'stage4_removed.jsonl')

    report = {
        'total': n,
        'kept': int(len(kept_df)),
        'flagged': int(len(flagged_df)),
        'removed': int(len(removed_df)),
        'thresholds': thresholds
    }
    with (outdir / 'stage4_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report

# -------------------------- CLI --------------------------

def main():
    parser = argparse.ArgumentParser(description='Stage 4 - EV domain specific validation (Balanced)')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL (stage3 cleaned)')
    parser.add_argument('--outdir', '-o', required=True, help='Output directory for stage4')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='sentence-transformers model name')
    parser.add_argument('--use-llm', action='store_true', help='Use OpenAI LLM for extra verification (requires API key)')
    parser.add_argument('--remove-thresh', type=float, default=0.45, help='Remove threshold (default balanced)')
    parser.add_argument('--flag-thresh', type=float, default=0.70, help='Flag threshold (default balanced)')
    args = parser.parse_args()

    inp = Path(args.input)
    outd = Path(args.outdir)
    outd.mkdir(parents=True, exist_ok=True)

    # load jsonl
    records = []
    with inp.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    report = stage4_process(records, outd, use_llm=args.use_llm, model_name=args.model, thresholds={'remove': args.remove_thresh, 'flag': args.flag_thresh})
    print('Stage 4 completed. Report:')
    print(report)

if __name__ == '__main__':
    main()
