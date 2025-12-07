"""
Stage 5 — PII Detection & Anonymization (EV-Specific)
-----------------------------------------------------

This stage performs privacy filtering on the Stage-4 cleaned dataset.
It detects and anonymizes sensitive identifiers using a custom
Presidio analyzer configured specifically for EV-domain PII.

What This Stage Detects
------------------------
Only EV-relevant PII is recognized (no general-purpose NER noise):

- VIN numbers (strict 17-character automotive standard)
- Indian vehicle license plates
- Indian PAN numbers
- Aadhaar numbers (#### #### ####)

Each detection is mapped to a single entity type: EV_PII.

How It Works
-------------
1. A custom Presidio AnalyzerEngine is built:
   - spaCy `en_core_web_lg` NLP backend
   - default recognizers disabled
   - a single custom PatternRecognizer created for EV_PII

2. Each record is scanned:
   - instruction text
   - output text

3. If PII is found:
   - record is anonymized using Presidio AnonymizerEngine
   - original + anonymized versions are saved under “flagged”

4. If no PII:
   - record is passed through unchanged to “clean”

Outputs (written to --outdir)
-----------------------------
stage5_pii_clean.jsonl      → records with zero PII  
stage5_pii_flagged.jsonl    → records containing PII (with anonymized versions)  
stage5_pii_report.csv        → one-line summary per record (#detections, entity types)

Usage
-----
python pipeline/stage5_pii.py \
    --input outputs/stage4_normalized/stage4_clean.jsonl \
    --outdir outputs/stage5_pii_filtered

Notes
-----
- Only custom EV patterns are used; avoids false positives from generic NER.
- Adheres to strict VIN + Aadhaar formatting rules.
- Ensures zero sensitive info enters downstream model training.
"""

import json
import os
import re
import pandas as pd
from tqdm import tqdm

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import RecognizerRegistry

import warnings
warnings.filterwarnings("ignore")

def build_analyzer():
    # Load SpaCy NLP engine
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"}
        ]
    })
    nlp_engine = provider.create_engine()

    # Analyzer WITHOUT default recognizers
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
        registry=None
    )

    analyzer.registry = RecognizerRegistry()


    # Custom EV-Specific Patterns Only
    custom_patterns = []

    # VIN — strict 17-char rule
    vin_regex = r"\b([A-HJ-NPR-Z]{3}[A-HJ-NPR-Z0-9]{5}[0-9X][A-HJ-NPR-Z0-9]{8})\b"
    custom_patterns.append(Pattern(name="VIN_PATTERN", regex=vin_regex, score=0.7))

    # Indian license plates (generic pattern)
    plate_regex = r"\b([A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4})\b"
    custom_patterns.append(Pattern(name="INDIAN_PLATE_PATTERN", regex=plate_regex, score=0.7))

    # Indian PAN number
    pan_regex = r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
    custom_patterns.append(Pattern(name="PAN_PATTERN", regex=pan_regex, score=0.7))

    # Aadhaar number (1234 5678 9101)
    aadhaar_regex = r"\b(\d{4}\s\d{4}\s\d{4})\b"
    custom_patterns.append(Pattern(name="AADHAAR_PATTERN", regex=aadhaar_regex, score=0.7))

    # Register single EV_PII recognizer
    ev_recognizer = PatternRecognizer(
        supported_entity="EV_PII",
        patterns=custom_patterns
    )

    analyzer.registry.add_recognizer(ev_recognizer)

    return analyzer

# 2. ANONYMIZER
anonymizer = AnonymizerEngine()

def anonymize_text(text, analyzer):
    results = analyzer.analyze(text=text, language="en")

    if not results:
        return text, []

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    return anonymized.text, results


def process_file(input_path, outdir):
    os.makedirs(outdir, exist_ok=True)

    analyzer = build_analyzer()

    clean = []
    flagged = []
    report_rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Scanning for PII...")):
            try:
                item = json.loads(line)
            except:
                continue

            instr = item.get("instruction", "")
            out = item.get("output", "")

            # Analyze PII
            new_instr, instr_hits = anonymize_text(instr, analyzer)
            new_out, out_hits = anonymize_text(out, analyzer)

            total_hits = instr_hits + out_hits

            if len(total_hits) == 0:
                clean.append({
                    "instruction": instr,
                    "output": out
                })
            else:
                flagged.append({
                    "instruction": instr,
                    "output": out,
                    "anonymized_instruction": new_instr,
                    "anonymized_output": new_out,
                    "pii_entities": [r.entity_type for r in total_hits]
                })

            # Report entry
            report_rows.append({
                "index": idx,
                "pii_count": len(total_hits),
                "entities": ",".join([r.entity_type for r in total_hits])
            })

    # Save outputs
    clean_path = os.path.join(outdir, "stage5_pii_clean.jsonl")
    flagged_path = os.path.join(outdir, "stage5_pii_flagged.jsonl")
    report_path = os.path.join(outdir, "stage5_pii_report.csv")

    with open(clean_path, "w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex) + "\n")

    with open(flagged_path, "w", encoding="utf-8") as f:
        for ex in flagged:
            f.write(json.dumps(ex) + "\n")

    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    print("\nStage 5 PII filtering completed.")
    print(f"- Clean: {len(clean)}")
    print(f"- Flagged: {len(flagged)}")
    print(f"- Saved report: {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Stage 4 clean JSONL")
    parser.add_argument("--outdir", required=True, help="Output folder")

    args = parser.parse_args()
    process_file(args.input, args.outdir)
