#!/usr/bin/env bash
# Train a PPO routing policy on the real Tianyan 176 topology (60q/81 edges).
#
# Unlike earlier 20q-scale policies (fixed obs dims), this model is built
# with max_num_qubits=60 / max_num_edges=81 (from tianyan176_66q.json),
# so it can also be evaluated on smaller tianyan sub-topologies.
#
# Phase 1: route-only (distance reward + SABRE features)
# Phase 2: noise-aware fine-tuning (optional, set PHASE2=1)
#
# Usage:
#   scripts/train_tianyan.sh [DEVICE] [PHASE1_STEPS] [PHASE2_STEPS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
DEVICE="${1:-cuda:0}"
PH1="${2:-300000}"
PH2="${3:-200000}"
PHASE2="${PHASE2:-1}"

DATA_DIR="$ROOT/traindata"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

TOPO="$DATA_DIR/topo/tianyan176_66q.json"

PH1_OUT="$MODELS_DIR/policy_tianyan176_phase1.pt"
PH2_OUT="$MODELS_DIR/policy_tianyan176_noiseaware.pt"
CKPT_DIR="$MODELS_DIR/ckpts_tianyan"
MAX_STEPS=800

echo "=========================================================="
echo "Tianyan 176 Policy Training (60q / 81 edges)"
echo "  device:          $DEVICE"
echo "  phase 1 steps:   $PH1"
echo "  phase 2 steps:   $PH2 (enabled=$PHASE2)"
echo "  max qubits:      60"
echo "  max edges:       81"
echo "  max episode len: $MAX_STEPS"
echo "  topology:        tianyan176_66q.json"
echo "  split:           tianyan"
echo "=========================================================="

# ---- Phase 1: route-only ----
PYTHONPATH="$ROOT/src" python3 -m routing.rl.train_agent \
  --data-dir "$DATA_DIR" \
  --split-prefix tianyan \
  --topo "$TOPO" \
  --reward-mode routing \
  --timesteps "$PH1" \
  --max-episode-steps "$MAX_STEPS" \
  --max-num-qubits 60 \
  --mapping-budget 8 \
  --device "$DEVICE" \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint-interval 20 \
  --out "$PH1_OUT"

# ---- Phase 2: noise-aware fine-tuning ----
if [ "$PHASE2" = "1" ]; then
  PYTHONPATH="$ROOT/src" python3 -m routing.rl.train_agent \
    --data-dir "$DATA_DIR" \
    --split-prefix tianyan \
    --topo "$TOPO" \
    --reward-mode noise_aware \
    --timesteps "$PH2" \
    --max-episode-steps "$MAX_STEPS" \
    --max-num-qubits 60 \
    --mapping-budget 8 \
    --load "$PH1_OUT" \
    --device "$DEVICE" \
    --checkpoint-dir "${CKPT_DIR}_ph2" \
    --out "$PH2_OUT"
fi

echo "Done. Models in $MODELS_DIR"
