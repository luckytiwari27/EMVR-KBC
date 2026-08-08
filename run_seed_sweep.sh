#!/usr/bin/env bash
# run_seed_sweep.sh -- run vanilla LeSR + EMVR-KBC across multiple seeds on one
# dataset. The extractor and proposer (the ONLY stages that call the LLM API)
# run exactly ONCE, regardless of how many seeds you pass -- seeds only vary
# the reasoner stage's training stochasticity (weight init, optimizer), which
# is the actual thing EMVR-KBC's warm-start claim needs seed coverage on.
# This keeps API cost fixed at "one dataset's worth of proposals" no matter
# how many seeds you sweep.
#
# USAGE:
#   ./run_seed_sweep.sh <dataset> <run_prefix> <seed1> [seed2] [seed3] ...
#
# EXAMPLE (3 seeds on UMLS):
#   ./run_seed_sweep.sh UMLs umls 5 42 123
#
# Requires LLM_API_KEY to be exported already (only used once, for the
# shared proposer call).

set -e

DATASET=$1
PREFIX=$2
shift 2
SEEDS=("$@")

if [ -z "$LLM_API_KEY" ]; then
  echo "LLM_API_KEY is not set. export LLM_API_KEY=... first (or add it to ~/.bashrc)."
  exit 1
fi

SHARED="${PREFIX}_shared"

echo "=== Extractor + proposer (ONE-TIME, shared across all seeds) ==="
if [ ! -d "runs/${SHARED}/proposed" ]; then
  python3 lesr.py --run_name "${SHARED}" --dataset "${DATASET}" --run_extractor --rand_seed 5
  python3 lesr.py --run_name "${SHARED}" --dataset "${DATASET}" --run_proposer --llm_name gpt35 --llm_api_key "$LLM_API_KEY"
else
  echo "Shared extractor/proposer output already exists at runs/${SHARED} -- skipping (no API cost incurred)."
fi

VANILLA_DIRS=()
EMVR_DIRS=()

for SEED in "${SEEDS[@]}"; do
  VANILLA="${PREFIX}_seed${SEED}_vanilla"
  EMVR="${PREFIX}_seed${SEED}_emvr"

  # fresh copy per seed so learned_weights.pt doesn't already exist --
  # otherwise the reasoner stage would just reload a previous seed's weights
  # instead of retraining with this seed's random init.
  rm -rf "runs/${VANILLA}" "runs/${EMVR}"
  cp -r "runs/${SHARED}" "runs/${VANILLA}"
  cp -r "runs/${SHARED}" "runs/${EMVR}"

  echo "=== Seed ${SEED}: vanilla LeSR reasoner (no API call) ==="
  python3 lesr.py --run_name "${VANILLA}" --dataset "${DATASET}" --run_reasoner --rand_seed "${SEED}"

  echo "=== Seed ${SEED}: EMVR-KBC reasoner (no API call) ==="
  python3 lesr.py --run_name "${EMVR}" --dataset "${DATASET}" --run_reasoner \
    --use_emvr --use_emvr_warmstart --emvr_save_stats \
    --emvr_sample_size 500 --emvr_beta 0.7 --emvr_tau_min 0.1 --emvr_tau_max 0.5 --rand_seed "${SEED}"

  VANILLA_DIRS+=("runs/${VANILLA}")
  EMVR_DIRS+=("runs/${EMVR}")
done

VANILLA_JOINED=$(IFS=,; echo "${VANILLA_DIRS[*]}")
EMVR_JOINED=$(IFS=,; echo "${EMVR_DIRS[*]}")

echo ""
echo "=== All seeds done. Consolidated report: ==="
python3 compare_runs.py \
  --vanilla_run_dir "${VANILLA_JOINED}" \
  --emvr_run_dir "${EMVR_JOINED}" \
  --dataset "${DATASET}"