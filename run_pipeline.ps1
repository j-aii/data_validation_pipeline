param(
    [string]$infile = "data/ev_trends_llm.jsonl",
    [string]$outdir = "outputs"
)

Write-Host ""
Write-Host "=== Running EV Validation Pipeline ==="
Write-Host ""


#  STAGE 0,1,2 (env)
Write-Host "Starting stage 0-2"

& "$PSScriptRoot\env\Scripts\Activate.ps1"

# Stage 0
python "$PSScriptRoot\pipeline\stage0.py" `
    --input (Join-Path $PSScriptRoot $infile) `
    --outdir (Join-Path $PSScriptRoot "$outdir\stage0_clean.jsonl")

# Stage 1
python "$PSScriptRoot\pipeline\stage1.py" `
    --input (Join-Path $PSScriptRoot "$outdir\stage0_clean.jsonl") `
    --outdir (Join-Path $PSScriptRoot "$outdir\stage1")

# Stage 2
python "$PSScriptRoot\pipeline\stage2.py" `
    --input (Join-Path $PSScriptRoot "$outdir\stage1\cleaned_stage1.jsonl") `
    --outdir (Join-Path $PSScriptRoot "$outdir\stage2")

deactivate


#  STAGE 3,4,5 (env2)
Write-Host ""
Write-Host "Starting stage 3-5"

& "$PSScriptRoot\evn2\Scripts\Activate.ps1"

python "$PSScriptRoot\pipeline\stage3.py" `
    --input (Join-Path $PSScriptRoot "$outdir\stage2\stage2_clean.jsonl") `
    --outdir (Join-Path $PSScriptRoot "$outdir\stage3")

python "$PSScriptRoot\pipeline\stage4.py" `
    --input (Join-Path $PSScriptRoot "$outdir\stage3\cleaned_stage3.jsonl") `
    --outdir (Join-Path $PSScriptRoot "$outdir\stage4")

python "$PSScriptRoot\pipeline\stage5.py" `
    --input (Join-Path $PSScriptRoot "$outdir\stage4\stage4_clean.jsonl") `
    --outdir (Join-Path $PSScriptRoot "$outdir\stage5")

deactivate

Write-Host "Pipeline Completed Successfully"

