"""
Stage 0 - Schema & Structure Validation

This script performs the first step of the EV LLM Validation Pipeline.
It validates each JSONL record to ensure it follows the expected QA schema
and removes malformed, empty, or structurally invalid samples.

What this stage does:
- Ensures each line is valid JSON.
- Validates required fields: instruction, output, (optional input).
- Enforces minimum length & non-empty constraints.
- Logs validation errors with helpful messages.
- Writes a cleaned JSONL file containing only valid entries.

Usage:
    python pipeline/stage0.py --input data/ev_trends_llm.jsonl --outdir outputs/stage0_clean.jsonl

Inputs:
    --input   Path to the raw dataset (.jsonl)
    --outdir  Path to save the cleaned output file

Outputs:
    A cleaned JSONL file containing only schema-valid QA pairs.
    Also prints counts of valid and invalid samples.

This stage ensures that only structurally correct data proceeds to Stage 1.
"""

import json
import argparse
from pydantic import BaseModel, ValidationError, field_validator
from rich.console import Console
from rich.progress import track

console = Console()

class QAPair(BaseModel):
    instruction: str
    output: str
    input: str | None = ""

    @field_validator("instruction")
    def validate_instruction(cls, v):
        if not v or not v.strip():
            raise ValueError("Instruction cannot be empty")
        if len(v) < 5:
            raise ValueError("Instruction too short")
        return v

    @field_validator("output")
    def validate_output(cls, v):
        if not v or not v.strip():
            raise ValueError("Output cannot be empty")
        if len(v) < 2:
            raise ValueError("Output too short")
        return v


def validate_jsonl(input_file, output_file):
    valid = 0
    invalid = 0
    cleaned = []

    console.print("\n[bold cyan]Stage 0: Schema Validation Started[/bold cyan]\n")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(track(lines, description="Validating...")):
        line = line.strip()
        if not line:
            invalid += 1
            continue

        try:
            raw = json.loads(line)
        except:
            console.print(f"[red]JSON Error at line {idx+1}[/red]")
            invalid += 1
            continue

        try:
            obj = QAPair(**raw)
            cleaned.append(obj.model_dump())
            valid += 1
        except ValidationError as e:
            console.print(f"[yellow]Line {idx+1} skipped: {e}[/yellow]")
            invalid += 1

    with open(output_file, "w", encoding="utf-8") as wf:
        for item in cleaned:
            wf.write(json.dumps(item) + "\n")

    console.print("\nStage 0 Complete")
    console.print(f"Valid: {valid}")
    console.print(f"Invalid: {invalid}")
    console.print(f"Saved → {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    validate_jsonl(args.input, args.outdir)
