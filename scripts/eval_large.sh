#!/usr/bin/env bash
# Evaluate large-scale policies on the held-out test splits, with baselines.
#
# Usage:
#   scripts/eval_large.sh [MAX_CIRCUITS] [DEVICE]
#   MAX_CIRCUITS: limit circuits per split (default: all)
#   DEVICE:       torch device for PPO forward passes (default cpu)
#
# Set BEAM=3 to use beam-search decoding, PHASE2=1 to also evaluate the
# noise-aware model with fidelity reporting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
MAX_CIRCUITS="${1:-}"
DEVICE="${2:-cpu}"
BEAM="${BEAM:-0}"
PHASE2="${PHASE2:-0}"

DATA_DIR="$ROOT/traindata"
MODELS_DIR="$ROOT/models"
RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR"

declare -A SCALE_TOPOS=(
  [8]="line_8q.json"
  [10]="ring_10q.json"
  [12]="grid_3x4_12q.json"
  [16]="grid_4x4_16q.json"
  [20]="grid_5x4_20q.json"
)

for n in 8 10 12 16 20; do
  TOPO="$DATA_DIR/topo/${SCALE_TOPOS[$n]}"
  PREFIX="large_n${n}"
  MODEL="$MODELS_DIR/policy_${PREFIX}_phase1.pt"
  [ -f "$MODEL" ] || { echo "skip n=$n (missing $MODEL)"; continue; }
  MAX_STEPS=$((250 + n * 15))
  OUT="$RESULTS_DIR/eval_${PREFIX}.json"
  EXTRA=()
  [ -n "$MAX_CIRCUITS" ] && EXTRA+=(--max-circuits "$MAX_CIRCUITS")
  [ "$BEAM" -gt 0 ] && EXTRA+=(--beam-width "$BEAM")

  echo "==================================================================="
  echo "n=$n  topo=$TOPO  split=${PREFIX}_test"
  echo "==================================================================="

  # ---- routing mode (SWAP count comparison vs SABRE / greedy) ----
  PYTHONPATH="$ROOT/src" python3 -m routing.rl.eval_policy \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --split "${PREFIX}_test" \
    --reward-mode routing \
    --topo "$TOPO" \
    --max-episode-steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --baselines \
    --out "$OUT" \
    "${EXTRA[@]}"

  # ---- noise-aware model (fidelity comparison) ----
  if [ "$PHASE2" = "1" ]; then
    PH2_MODEL="$MODELS_DIR/policy_${PREFIX}_noiseaware.pt"
    [ -f "$PH2_MODEL" ] || { echo "skip noise-aware n=$n (missing $PH2_MODEL)"; continue; }
    PYTHONPATH="$ROOT/src" python3 -m routing.rl.eval_policy \
      --model "$PH2_MODEL" \
      --data-dir "$DATA_DIR" \
      --split "${PREFIX}_test" \
      --reward-mode noise_aware \
      --topo "$TOPO" \
      --max-episode-steps "$MAX_STEPS" \
      --device "$DEVICE" \
      --baselines \
      --out "$OUT" \
      "${EXTRA[@]}"
  fi
done

echo "Done. Results in $RESULTS_DIR"
