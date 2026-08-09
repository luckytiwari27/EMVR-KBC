# run_reasoner_only_sweep.ps1 -- reruns ONLY the reasoner stage (vanilla + EMVR)
# across multiple seeds, reusing an EXISTING shared extractor/proposer run dir.
# No LLM API calls. Prints the comparison report AND saves it to a timestamped
# .txt file so it isn't lost when you close the terminal.
#
# USAGE:
#   .\run_reasoner_only_sweep.ps1 -Dataset UMLs -SharedDir runs\a_simple_run -Prefix umls_wd01 -Seeds 5,42,123

param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$SharedDir,
    [Parameter(Mandatory=$true)][string]$Prefix,
    [Parameter(Mandatory=$true)][int[]]$Seeds
)

if (-not (Test-Path "$SharedDir\proposed")) {
    Write-Error "ERROR: $SharedDir has no proposed\ subfolder -- is this a valid shared run dir?"
    exit 1
}

$vanillaDirs = @()
$emvrDirs = @()

foreach ($seed in $Seeds) {
    $vanilla = "${Prefix}_seed${seed}_vanilla"
    $emvr = "${Prefix}_seed${seed}_emvr"

    Remove-Item -Recurse -Force "runs\$vanilla" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "runs\$emvr" -ErrorAction SilentlyContinue
    Copy-Item -Recurse $SharedDir "runs\$vanilla"
    Copy-Item -Recurse $SharedDir "runs\$emvr"
    # drop any stale reasoner/ output that came along in the copy -- otherwise
    # lesr.py sees an existing learned_weights.pt and SKIPS retraining, silently
    # reusing old results instead of applying the corrected weight_decay.
    Remove-Item -Recurse -Force "runs\$vanilla\reasoner" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "runs\$emvr\reasoner" -ErrorAction SilentlyContinue

    Write-Host "=== Seed $seed : vanilla LeSR reasoner (weight_decay=0.1, no API call) ==="
    python lesr.py --run_name $vanilla --dataset $Dataset --run_reasoner --weight_decay 0.1 --rand_seed $seed
    if ($LASTEXITCODE -ne 0) { Write-Error "vanilla run failed for seed $seed"; exit 1 }

    Write-Host "=== Seed $seed : EMVR-KBC reasoner (weight_decay=0.1, no API call) ==="
    python lesr.py --run_name $emvr --dataset $Dataset --run_reasoner --weight_decay 0.1 `
        --use_emvr --use_emvr_warmstart --emvr_save_stats `
        --emvr_sample_size 500 --emvr_beta 0.7 --emvr_tau_min 0.1 --emvr_tau_max 0.5 --rand_seed $seed
    if ($LASTEXITCODE -ne 0) { Write-Error "EMVR run failed for seed $seed"; exit 1 }

    $vanillaDirs += "runs\$vanilla"
    $emvrDirs += "runs\$emvr"
}

$vanillaJoined = $vanillaDirs -join ","
$emvrJoined = $emvrDirs -join ","

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = "reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$reportPath = "$reportDir\${Prefix}_${timestamp}.txt"

Write-Host ""
Write-Host "=== All seeds done. Consolidated report (also saving to $reportPath) ==="
python compare_runs.py --vanilla_run_dir $vanillaJoined --emvr_run_dir $emvrJoined --dataset $Dataset 2>&1 | Tee-Object -FilePath $reportPath

Write-Host ""
Write-Host "Report saved to: $reportPath"