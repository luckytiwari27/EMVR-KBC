# run_emvr_wl.ps1 -- reasoner-only sweep WITH working weight learning.
#
# Runs three arms off ONE shared extractor/proposer directory (no LLM API
# calls, so the LLM cost is paid once and never again):
#
#   <Prefix>_vanilla       LeSR baseline           (needed for Filtering Recall)
#   <Prefix>_emvr_nowarm   --use_emvr only         (isolates the filtering effect)
#   <Prefix>_emvr_warm     --use_emvr + warm-start (full EMVR-KBC)
#
# WHY THE HYPERPARAMETERS CHANGE FROM run_reasoner_only_sweep.ps1
# --------------------------------------------------------------
# train_loop() takes ONE full-batch AdamW step per epoch, and Adam moves each
# parameter by about `lr` per step. With lr=1e-3 and StepLR(step=100,
# gamma=0.1), the total achievable logit displacement is roughly
#   100*1e-3 + 100*1e-4 + 100*1e-5 + ... ~ 0.11
# so softmax(raw_weights) never leaves uniform and the run silently reproduces
# the base paper's "w/o weight learning" ablation. -ConstantLr (the default
# here) sets scheduler_gamma=1 so the learning rate does not collapse, and
# raises lr / epochs / patience so the weights can actually separate.
#
# NOTE: lesr.py declares --scheduler_gamma as type=int. Integer 1 is fine, but
# if you want a fractional gamma, change that line in lesr.py to type=float.
#
# Seeds are deliberately NOT swept: ReasonerModel initialises raw_weights to
# torch.zeros, so with a shared extractor/proposer dir the vanilla reasoner is
# fully deterministic and a "3-seed sweep" reports one run three times.
#
# USAGE:
#   .\run_emvr_wl.ps1 -Dataset UMLs -SharedDir runs\a_simple_run -Prefix umls_wl
#   .\run_emvr_wl.ps1 -Dataset UMLs -SharedDir runs\a_simple_run -Prefix umls_wl_plus -ReasonerPlus
#   .\run_emvr_wl.ps1 -Dataset UMLs -SharedDir runs\a_simple_run -Prefix umls_wl -SkipVanilla

param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$SharedDir,
    [Parameter(Mandatory=$true)][string]$Prefix,

    # ---- weight-learning settings (the actual fix) ----
    [double]$Lr = 0.01,          # was 0.001 -- too small for 1 step/epoch
    [int]$Epochs = 3000,         # was 1000
    [int]$Patience = 300,        # was 30 -- fired long before convergence
    [double]$WeightDecay = 0.1,  # paper value; keep for fidelity
    [int]$SchedStep = 100000,    # effectively disables the LR staircase
    [int]$SchedGamma = 1,        # 1 = constant LR (int, see note above)

    # ---- model / arm selection ----
    [switch]$ReasonerPlus,       # use ReasonerModelPlus (explicit alpha, matches Eq. 2)
    [switch]$SkipVanilla,        # skip the baseline arm (loses Filtering Recall)
    [switch]$SkipNoWarm,         # skip the filtering-only arm (loses the ablation)

    # ---- EMVR settings ----
    [int]$SampleSize = 500,
    [double]$Beta = 0.7,
    [double]$TauMin = 0.1,
    [double]$TauMax = 0.5,
    [int]$KgeBsize = 32
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "$SharedDir\proposed")) {
    Write-Error "ERROR: $SharedDir has no proposed\ subfolder -- run --run_extractor and --run_proposer first."
}

$plusFlag = @()
if ($ReasonerPlus) { $plusFlag = @("--use_reasoner_plus") }

$wlFlags = @(
    "--initial_lr", $Lr,
    "--num_epochs", $Epochs,
    "--early_stop_patience", $Patience,
    "--weight_decay", $WeightDecay,
    "--scheduler_step", $SchedStep,
    "--scheduler_gamma", $SchedGamma,
    "--kge_bsize", $KgeBsize
) + $plusFlag

function New-Arm([string]$name) {
    # Fresh copy of the shared extractor/proposer output. reasoner/ and
    # grounding/ are dropped so lesr.py re-grounds and re-trains instead of
    # silently reusing stale artifacts (_rules_all.json is regenerated from
    # proposed/ at the start of every --run_reasoner call, so deleting the
    # whole reasoner/ folder is safe). semantic_similarities.pt is preserved
    # because recomputing it means reloading the sentence LM.
    $dir = "runs\$name"
    Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    Copy-Item -Recurse $SharedDir $dir

    $sem = "$dir\reasoner\semantic_similarities.pt"
    $tmp = $null
    if (Test-Path $sem) { $tmp = "$dir\_semsim.tmp"; Move-Item $sem $tmp }

    Remove-Item -Recurse -Force "$dir\reasoner"  -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "$dir\grounding" -ErrorAction SilentlyContinue

    if ($tmp) {
        New-Item -ItemType Directory -Force -Path "$dir\reasoner" | Out-Null
        Move-Item $tmp $sem
    }
    return $dir
}

$compareArgs = @("--dataset", $Dataset)

# ---------------------------------------------------------------- arm 1
if (-not $SkipVanilla) {
    $name = "${Prefix}_vanilla"
    Write-Host "`n=== ARM 1/3: vanilla LeSR, weight learning ON (no API call) ===" -ForegroundColor Cyan
    $dir = New-Arm $name
    python lesr.py --run_name $name --dataset $Dataset --run_reasoner @wlFlags
    if ($LASTEXITCODE -ne 0) { Write-Error "vanilla arm failed" }
    $compareArgs += @("--vanilla_run_dir", $dir)
} else {
    Write-Host "`n[skipping vanilla arm -- Filtering Recall will report N/A, since the" -ForegroundColor Yellow
    Write-Host " weights of the rules EMVR discarded only exist in a run that grounded them]" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- arm 2
if (-not $SkipNoWarm) {
    $name = "${Prefix}_emvr_nowarm"
    Write-Host "`n=== ARM 2/3: EMVR-KBC, verification only, NO warm-start ===" -ForegroundColor Cyan
    $dir = New-Arm $name
    python lesr.py --run_name $name --dataset $Dataset --run_reasoner @wlFlags `
        --use_emvr --emvr_save_stats `
        --emvr_sample_size $SampleSize --emvr_beta $Beta `
        --emvr_tau_min $TauMin --emvr_tau_max $TauMax
    if ($LASTEXITCODE -ne 0) { Write-Error "EMVR no-warm arm failed" }
    $compareArgs += @("--emvr_nowarm_run_dir", $dir)
}

# ---------------------------------------------------------------- arm 3
$name = "${Prefix}_emvr_warm"
Write-Host "`n=== ARM 3/3: EMVR-KBC, verification + warm-start (full framework) ===" -ForegroundColor Cyan
$dir = New-Arm $name
python lesr.py --run_name $name --dataset $Dataset --run_reasoner @wlFlags `
    --use_emvr --emvr_save_stats --use_emvr_warmstart `
    --emvr_sample_size $SampleSize --emvr_beta $Beta `
    --emvr_tau_min $TauMin --emvr_tau_max $TauMax
if ($LASTEXITCODE -ne 0) { Write-Error "EMVR warm arm failed" }
$compareArgs += @("--emvr_warm_run_dir", $dir)

# ---------------------------------------------------------------- report
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path "reports" | Out-Null
$report = "reports\${Prefix}_${stamp}.txt"

Write-Host "`n=== COMPARISON REPORT ===" -ForegroundColor Cyan
python compare_emvr.py @compareArgs --sensitivity | Tee-Object -FilePath $report
Write-Host "`nSaved to $report" -ForegroundColor Green