# **EV LLM Data Validation Pipeline — README**

A modular, multi-stage pipeline for validating, cleaning, deduplicating, scoring, and anonymizing datasets for EV (Electric Vehicle) domain LLM training.

This pipeline is intentionally **defensive**, **domain-aware**, and designed for **production-grade dataset preparation**.

---

# **Pipeline Overview (Stage 0 → Stage 5)**

The full pipeline performs:

1. **Stage 0 – Structural Schema Validation**
2. **Stage 1 – Multi-Strategy Deduplication**
3. **Stage 2 – Semantic Quality & Label-Issue Detection**
4. **Stage 3 – Semantic Correctness & Answer Quality (RAGAS + fallback)**
5. **Stage 4 – EV Domain-Specific QA Validation**
6. **Stage 5 – PII Detection & Anonymization (Presidio Custom Config)**

Each stage writes **clean**, **flagged**, **report**, and **embedding** outputs (as applicable).



---

# **Stage 0 — Schema & Structure Validation**

Validates the **form** of every JSONL record.

### **What This Stage Does**

Ensures every line is valid JSON
Enforces required fields:

* `instruction`
* `output`
* (optional) `input`
  Removes empty, malformed, or broken entries
  Logs every failure
  Outputs a cleaned file suitable for further processing

### **Usage**

```bash
python pipeline/stage0.py --input data/raw.jsonl --outdir outputs/stage0_clean.jsonl
```

### **Outputs**

* `stage0_clean.jsonl`
* Console summary of valid/invalid counts



---

# **Stage 1 — Multi-Strategy Deduplication Pipeline**

Balanced deduplication using **5 strategies**:

### **1. Exact Deduplication**

* Normalize text
* Hash with SHA-256
* Remove perfect duplicates

### **2. Near-Duplicate (Shingling + Jaccard + MinHash)**

* k-shingles
* MinHash + LSH (if installed)
* Jaccard threshold default 0.80–0.85

### **3. Template-Level Deduplication**

* Mask: numbers, IDs, patterns
* Groups templated variants like:

  > “Write a report for vehicle 1234”
  > “Write a report for vehicle 9982”

### **4. Semantic Deduplication**

* Sentence-transformer embeddings
* Cosine similarity threshold ~0.92–0.95
* FAISS/sklearn if available, else brute

### **5. Cluster Resolution**

* Keep the best representative
* Based on longest / most informative output

### **Usage**

```bash
python pipeline/stage1_dedup_minhash.py \
  --input outputs/stage0_clean.jsonl \
  --outdir outputs/stage1_dedup \
  --lsh-threshold 0.80 \
  --semantic-threshold 0.92
```

### **Outputs**

* `cleaned_stage1.jsonl`
* Cluster files:

  * `stage1_clusters_exact.json`
  * `stage1_clusters_near.json`
  * `stage1_clusters_template.json`
  * `stage1_clusters_semantic.json`
* `dedup_report.json`



---

# **Stage 2 — Semantic Quality & Label-Issue Detection**

Removes semantically inconsistent, noisy, mislabeled, and OOD samples using **three signals**.

### **1. KNN Embedding Outlier Detection**

* Uses whole-record embedding
* Computes neighbor similarity
* Outlier scoring ∈ [0,1]

### **2. Cleanlab Label-Issue Detection**

* Auto-generated pseudo-labels (via clusters)
* Cleanlab flags overlapping/contradictory samples

### **3. Cleanlab OOD Score**

* High-distance embeddings identify off-topic samples

### **Removal Rule**

Each sample gets a **removal_score**:

| Condition             | Score |
| --------------------- | ----- |
| Strong KNN outlier    | +0.5  |
| Moderate outlier      | +0.3  |
| Cleanlab label issue  | +0.4  |
| OOD > 95th percentile | +0.3  |

Removed if: **removal_score ≥ 0.7**

### **Usage**

```bash
python pipeline/stage2_cleanlab.py \
  --input outputs/stage1_dedup/cleaned_stage1.jsonl \
  --outdir outputs/stage2_semantic \
  --save-embeddings \
  --remove-threshold 0.85
```

### **Outputs**

* `stage2_clean.jsonl`
* `stage2_flagged.csv`
* `stage2_report.json`
* `embeddings.npy` (optional)



---

# **Stage 3 — Semantic Correctness & Answer Quality**

*RAGAS + fallback heuristics + lettuceDetect where possible*

Evaluates if **answers are correct, grounded, and non-hallucinatory**.

### **Quality Signals**

RAGAS — faithfulness, correctness, answer relevance
lettuceDetect — hallucination scoring
SBERT similarity
Heuristic fallback rules when heavy dependencies are missing

### **Usage**

```bash
python ev_stage3_ragas.py \
  --input outputs/stage2_semantic/stage2_clean.jsonl \
  --outdir outputs/stage3
```

### **Outputs**

* `stage3_scored.csv`
* `stage3_flagged.csv`
* `stage3_removed.jsonl`
* `cleaned_stage3.jsonl`
* `stage3_report.json`



---

# **Stage 4 — EV Domain-Specific QA Validation**

Domain-specialized correctness rules for EV instructions & outputs.

### **Checks Performed**

EV terminology validation
Technical factual correctness (battery, motor, charging, BMS, range, CAN, etc.)
Rule-based correctness filters
Red-flag checks:

* nonsensical specs
* impossible EV ranges
* incorrect charging curves
  Optional LLM-based validation (if enabled)

### **Usage**

```bash
python stage4_ev_validate.py \
  --input outputs/stage3/cleaned_stage3.jsonl \
  --outdir outputs/stage4
```

### **Outputs**

* `stage4_scored.csv`
* `stage4_clean.jsonl`
* `stage4_flagged.jsonl`
* `stage4_removed.jsonl`
* `stage4_report.json`



---

# **Stage 5 — PII Detection & Anonymization (EV-Specific Presidio)**

## **Prerequisite for Stage 5 – spaCy Model Installation**

Before running PII detection:

```bash
pip install spacy
python -m spacy download en_core_web_lg
python -m spacy validate
```

Custom Presidio configuration detecting ONLY EV-relevant identifiers:

### **Detected Entities**

**VIN** (strict 17-character standard)
**Indian license plates**
**PAN**
**Aadhaar** (#### #### #### format)

All mapped to one entity type: **EV_PII**

### **How It Works**

* Custom Presidio AnalyzerEngine (all default recognizers disabled)
* Adds one PatternRecognizer built from EV patterns
* Scans instruction + output
* If PII found:

  * anonymize
  * store original + anonymized versions in `flagged`
* If no PII → kept as is

### **Usage**

```bash
python pipeline/stage5_pii.py \
  --input outputs/stage4/stage4_clean.jsonl \
  --outdir outputs/stage5_pii_filtered
```

### **Outputs**

* `stage5_pii_clean.jsonl`
* `stage5_pii_flagged.jsonl`
* `stage5_pii_report.csv`



---

# **Environment Setup**

We are using **two environments**:

### **env1 (dedup + cleanlab + embeddings)**

Include:

```
pandas
numpy
tqdm
sentence-transformers
scikit-learn
datasketch
faiss-cpu
cleanlab
```

### **env2 (ragas + hallucination + presidio)**

Include:

```
pandas
numpy
ragas
lettucedetect
sentence-transformers
presidio-analyzer
presidio-anonymizer
spacy
```

---

### **spaCy Setup (Required for Stage 5 – PII Detection)**

Presidio requires a spaCy NLP backend.

Install spaCy:

```bash
pip install spacy
```

Download the large English model:

```bash
python -m spacy download en_core_web_lg
```

---

# **Running Entire Pipeline**

```bash
python stage0.py
python stage1_dedup_minhash.py
python stage2_cleanlab.py
python ev_stage3_ragas.py
python stage4_ev_validate.py
python stage5_pii.py
```

---

# **Folder Structure**

```
data/
pipeline/
    stage0.py
    stage1_dedup_minhash.py
    stage2_cleanlab.py
    ev_stage3_ragas.py
    stage4_ev_validate.py
    stage5_pii.py
outputs/
    stage0_clean.jsonl
    stage1_dedup/
    stage2_semantic/
    stage3/
    stage4/
    stage5_pii/
```

---

# **To run**

While in the ev_llm_pipeline directory

```bash
./run_pipeline.ps1
```
---







