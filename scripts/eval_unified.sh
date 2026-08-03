#!/usr/bin/env bash
# Evaluate the unified policy on all topologies and the held-out test split.
#
# Usage:
#   scripts/eval_unified.sh [MODEL] [DEVICE]
#   MODEL:  policy checkpoint (default models/policy_unified_phase1.pt)
#   DEVICE: torch device (default cpu)
#
#   BEAM=3  to enable beam search
#   BASELINES=1  to also run SABRE and Greedy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
MODEL="${1:-$ROOT/models/policy_unified_phase1.pt}"
DEVICE="${2:-cpu}"
BEAM="${BEAM:-0}"
BASELINES="${BASELINES:-0}"

DATA_DIR="$ROOT/traindata"
RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR"

TOPO_NAMES=(
  "line_20q" "ring_20q" "grid_5x4_20q"
)

BASELINE_FLAG=""
[ "$BASELINES" = "1" ] && BASELINE_FLAG="--baselines --no-random"

BEAM_FLAG=""
[ "$BEAM" -gt 0 ] && BEAM_FLAG="--beam-width $BEAM"

echo "Model: $MODEL"
echo "Split: unified_test (60 circuits across n=8..20)"
echo "Max num qubits: 20"
echo

for topo in "${TOPO_NAMES[@]}"; do
  TOPO_PATH="$DATA_DIR/topo/${topo}.json"
  OUT="$RESULTS_DIR/eval_unified_${topo}.json"

  echo "--- $topo ---"
  PYTHONPATH="$ROOT/src" python3 -m routing.rl.eval_policy \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --split unified_test \
    --reward-mode routing \
    --topo "$TOPO_PATH" \
    --max-num-qubits 20 \
    --max-episode-steps 400 \
    --device "$DEVICE" \
    --out "$OUT" \
    $BASELINE_FLAG \
    $BEAM_FLAG
  echo
done

echo "Done. Results in $RESULTS_DIR"
