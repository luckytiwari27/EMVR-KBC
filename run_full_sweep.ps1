# run_full_sweep.ps1 -- FULL pipeline for a dataset+LLM combo you haven't
# proposed rules for yet: extractor -> proposer -> reasoner (vanilla + EMVR,
# multiple seeds) -> comparison report, saved to a file.
#
# Use this for any (Dataset, LlmName) pair where you DON'T already have
# proposed rules cached. If you already have them (like your gpt35 UMLS run),
# use run_reasoner_only_sweep.ps1 instead -- it skips straight to the
# reasoner stage and costs zero API calls.
#
# USAGE:
#   .\run_full_sweep.ps1 -Dataset UMLs -LlmName gemini15 -ApiKey "your-key" `
#       -Prefix umls_gemini15 -Seeds 5,42,123
#
# RESUMING AFTER A RATE LIMIT / QUOTA EXHAUSTION (e.g. switching Gmail
# accounts on Gemini's free tier): the proposer stage is resumable per
# relation. If this script stops partway through proposing (you'll see a
# clear "quota exhausted, switch keys" message from proposer.py), just
# re-run the EXACT SAME command with a new -ApiKey. It will skip every
# relation already proposed and continue from where it stopped -- no
# wasted API calls, no need to restart extraction.

param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][ValidateSet("gpt35","gpt40","gemini15","llama3")][string]$LlmName,
    [Parameter(Mandatory=$true)][string]$ApiKey,
    [Parameter(Mandatory=$true)][string]$Prefix,
    [Parameter(Mandatory=$true)][int[]]$Seeds,
    [double]$WeightDecay = 0.1
)

$sharedDir = "runs\${Prefix}_shared"

# ---------------- Step 1: extractor (no API call, deterministic given the ----
# ---------------- same --rand_seed, safe to (re)run cheaply if missing) ------
if (-not (Test-Path "$sharedDir\subgraphs")) {
    Write-Host "=== Extractor (dataset=$Dataset) ==="
    python lesr.py --run_name "${Prefix}_shared" --dataset $Dataset --run_extractor --rand_seed 5
    if ($LASTEXITCODE -ne 0) { Write-Error "Extractor failed."; exit 1 }
} else {
    Write-Host "Subgraphs already exist at $sharedDir\subgraphs -- skipping extractor."
}

# ---------------- Step 2: proposer (the ONLY step that calls the LLM) -------
if (-not (Test-Path "$sharedDir\proposed")) {
    Write-Host "=== Proposer (llm=$LlmName) ==="
} else {
    Write-Host "=== Proposer (llm=$LlmName) -- resuming, some relations may already be done ==="
}
python lesr.py --run_name "${Prefix}_shared" --dataset $Dataset --run_proposer --llm_name $LlmName --llm_api_key $ApiKey
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Proposer stopped (see message above -- likely a quota/rate limit)."
    Write-Host "Re-run this EXACT command with a different -ApiKey to resume:"
    Write-Host "  .\run_full_sweep.ps1 -Dataset $Dataset -LlmName $LlmName -ApiKey <NEW_KEY> -Prefix $Prefix -Seeds $($Seeds -join ',')"
    Write-Host "Already-proposed relations will be skipped automatically -- nothing is lost."
    exit 1
}

# ---------------- Step 3: reasoner, vanilla + EMVR, per seed ----------------
$vanillaDirs = @()
$emvrDirs = @()

foreach ($seed in $Seeds) {
    $vanilla = "${Prefix}_seed${seed}_vanilla"
    $emvr = "${Prefix}_seed${seed}_emvr"

    Remove-Item -Recurse -Force "runs\$vanilla" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "runs\$emvr" -ErrorAction SilentlyContinue
    Copy-Item -Recurse $sharedDir "runs\$vanilla"
    Copy-Item -Recurse $sharedDir "runs\$emvr"
    Remove-Item -Recurse -Force "runs\$vanilla\reasoner" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "runs\$emvr\reasoner" -ErrorAction SilentlyContinue

    Write-Host "=== Seed $seed : vanilla LeSR reasoner (weight_decay=$WeightDecay, no API call) ==="
    python lesr.py --run_name $vanilla --dataset $Dataset --run_reasoner --weight_decay $WeightDecay --rand_seed $seed
    if ($LASTEXITCODE -ne 0) { Write-Error "vanilla run failed for seed $seed"; exit 1 }

    Write-Host "=== Seed $seed : EMVR-KBC reasoner (weight_decay=$WeightDecay, no API call) ==="
    python lesr.py --run_name $emvr --dataset $Dataset --run_reasoner --weight_decay $WeightDecay `
        --use_emvr --use_emvr_warmstart --emvr_save_stats `
        --emvr_sample_size 500 --emvr_beta 0.7 --emvr_tau_min 0.1 --emvr_tau_max 0.5 --rand_seed $seed
    if ($LASTEXITCODE -ne 0) { Write-Error "EMVR run failed for seed $seed"; exit 1 }

    $vanillaDirs += "runs\$vanilla"
    $emvrDirs += "runs\$emvr"
}

# ---------------- Step 4: comparison report, printed + saved ----------------
$vanillaJoined = $vanillaDirs -join ","
$emvrJoined = $emvrDirs -join ","

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "reports" | Out-Null
$reportPath = "reports\${Prefix}_${timestamp}.txt"

Write-Host ""
Write-Host "=== All seeds done. Consolidated report (also saving to $reportPath) ==="
python compare_runs.py --vanilla_run_dir $vanillaJoined --emvr_run_dir $emvrJoined --dataset $Dataset 2>&1 | Tee-Object -FilePath $reportPath

Write-Host ""
Write-Host "Report saved to: $reportPath"