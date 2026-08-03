#!/usr/bin/env bash
# Train a single unified routing policy that handles all circuit scales
# (n=8..20) and all topologies (line/ring/grid, 14 total).
#
# The observation/action spaces are padded to the largest sizes
# (max_num_qubits=20, max_num_edges=31 from grid_5x4), so one model
# serves all circuit+topology combinations.
#
# Phase 1: pure routing (distance reward + SABRE features)
# Phase 2: noise-aware fine-tuning (optional, set PHASE2=1)
#
# Usage:
#   scripts/train_unified.sh [DEVICE] [PHASE1_STEPS] [PHASE2_STEPS]
#   DEVICE:       torch device (default cuda:0)
#   PHASE1_STEPS: PPO timesteps for phase 1 (default 500000)
#   PHASE2_STEPS: PPO timesteps for phase 2 (default 300000)
#
#   PHASE2=0  to skip phase 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
DEVICE="${1:-cuda:0}"
PH1="${2:-500000}"
PH2="${3:-300000}"
PHASE2="${PHASE2:-1}"

DATA_DIR="$ROOT/traindata"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

# 3 large topologies (all have 20 physical qubits = max_num_qubits)
# line_20q: 20 edges, ring_20q: 20 edges, grid_5x4_20q: 31 edges
TOPO_LIST="line_20q.json,ring_20q.json,grid_5x4_20q.json"

TOPO_ARGS=""
for t in ${TOPO_LIST//,/ }; do
  TOPO_ARGS="$TOPO_ARGS,$DATA_DIR/topo/$t"
done
TOPO_ARGS="${TOPO_ARGS:1}"

PH1_OUT="$MODELS_DIR/policy_unified_phase1.pt"
PH2_OUT="$MODELS_DIR/policy_unified_noiseaware.pt"
CKPT_DIR="$MODELS_DIR/ckpts_unified"
MAX_STEPS=400

echo "=========================================================="
echo "Unified Policy Training"
echo "  device:          $DEVICE"
echo "  phase 1 steps:   $PH1"
echo "  phase 2 steps:   $PH2 (enabled=$PHASE2)"
echo "  max qubits:      20"
echo "  max edges:       31 (grid_5x4)"
echo "  max episode len: $MAX_STEPS"
echo "  topologies:      3 (line_20q, ring_20q, grid_5x4_20q)"
echo "  split:           unified"
echo "=========================================================="

# ---- Phase 1: route-only ----
PYTHONPATH="$ROOT/src" python3 -m routing.rl.train_agent \
  --data-dir "$DATA_DIR" \
  --split-prefix unified \
  --topo-list "$TOPO_ARGS" \
  --topo-balance episodes \
  --reward-mode routing \
  --timesteps "$PH1" \
  --max-episode-steps "$MAX_STEPS" \
  --max-num-qubits 20 \
  --device "$DEVICE" \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint-interval 20 \
  --out "$PH1_OUT"

# ---- Phase 2: noise-aware fine-tuning ----
if [ "$PHASE2" = "1" ]; then
  PYTHONPATH="$ROOT/src" python3 -m routing.rl.train_agent \
    --data-dir "$DATA_DIR" \
    --split-prefix unified \
    --topo-list "$TOPO_ARGS" \
    --topo-balance episodes \
    --reward-mode noise_aware \
    --timesteps "$PH2" \
    --max-episode-steps "$MAX_STEPS" \
    --max-num-qubits 20 \
    --load "$PH1_OUT" \
    --device "$DEVICE" \
    --checkpoint-dir "${CKPT_DIR}_ph2" \
    --out "$PH2_OUT"
fi

echo "Done. Models in $MODELS_DIR"
