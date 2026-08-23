#!/usr/bin/env bash
# Train a unified 20q routing policy (line/ring/grid, n=8..20 circuits).
#
# Observation/action spaces are padded to the largest sizes
# (max_num_qubits=20, max_num_edges=31 from grid_5x4), so one model
# serves all circuit+topology combinations.
#
# Phase 1: route-only (distance reward + SABRE features)
# Phase 2: noise-aware fine-tuning with state-vector (trajectory) simulator
#          (--fidelity-sim trajectory, O(2^n) 内存，20q 不再 OOM；
#           P0 优化后默认 16 条轨迹 + 批量向量化演化)
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
TOPO_LIST="$DATA_DIR/topo/line_20q.json,$DATA_DIR/topo/ring_20q.json,$DATA_DIR/topo/grid_5x4_20q.json"

PH1_OUT="$MODELS_DIR/policy_unified_phase1.pt"
PH2_OUT="$MODELS_DIR/policy_unified_noiseaware.pt"
CKPT_DIR="$MODELS_DIR/ckpts_unified"
MAX_STEPS=400

echo "=========================================================="
echo "Unified 20q Policy Training (line/ring/grid, n=8..20)"
echo "  device:          $DEVICE"
echo "  phase 1 steps:   $PH1"
echo "  phase 2 steps:   $PH2 (enabled=$PHASE2)"
echo "  max qubits:      20"
echo "  max edges:       31 (grid_5x4)"
echo "  max episode len: $MAX_STEPS"
echo "  topologies:      line_20q, ring_20q, grid_5x4_20q"
echo "  split:           unified"
echo "  phase 2 sim:     trajectory (state-vector, O(2^n))"
echo "=========================================================="

# ---- Phase 1: route-only ----
PYTHONPATH="$ROOT/src" python3 -u -m routing.rl.train_agent \
  --data-dir "$DATA_DIR" \
  --split-prefix unified \
  --topo-list "$TOPO_LIST" \
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
  PYTHONPATH="$ROOT/src" python3 -u -m routing.rl.train_agent \
    --data-dir "$DATA_DIR" \
    --split-prefix unified \
    --topo-list "$TOPO_LIST" \
    --topo-balance episodes \
    --reward-mode noise_aware \
    --timesteps "$PH2" \
    --max-episode-steps "$MAX_STEPS" \
    --max-num-qubits 20 \
    --load "$PH1_OUT" \
    --device "$DEVICE" \
    --fidelity-sim trajectory \
    --traj-trajectories 16 \
    --checkpoint-dir "${CKPT_DIR}_ph2" \
    --out "$PH2_OUT"
fi

echo "Done. Models in $MODELS_DIR"
