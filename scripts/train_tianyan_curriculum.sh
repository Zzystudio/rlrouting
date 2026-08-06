#!/usr/bin/env bash
# Curriculum training on the Tianyan 176 topology (60q/81 edges):
#   adaptive GAE λ + cross-scale curriculum (n10 -> n20 -> tianyan).
#
# 相比 train_tianyan.sh 的改动:
#   --curriculum-keys large_n10,large_n20,tianyan   按训练进度切换规模
#   --gae-adaptive --gae-lam-min 0.95 --gae-lam-max 0.995
#     GAE λ 随 episode 进度线性增长, 缓解 800 步长电路截断信号衰减
#
# Usage:
#   scripts/train_tianyan_curriculum.sh [DEVICE] [PHASE1_STEPS] [PHASE2_STEPS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
DEVICE="${1:-cuda:0}"
PH1="${2:-500000}"
PH2="${3:-200000}"
PHASE2="${PHASE2:-1}"

DATA_DIR="$ROOT/traindata"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

TOPO="$DATA_DIR/topo/tianyan176_66q.json"

PH1_OUT="$MODELS_DIR/policy_tianyan176_curric_phase1.pt"
PH2_OUT="$MODELS_DIR/policy_tianyan176_curric_noiseaware.pt"
CKPT_DIR="$MODELS_DIR/ckpts_tianyan_curric"
MAX_STEPS=800

echo "=========================================================="
echo "Tianyan 176 Curriculum Training (60q / 81 edges)"
echo "  device:          $DEVICE"
echo "  phase 1 steps:   $PH1"
echo "  phase 2 steps:   $PH2 (enabled=$PHASE2)"
echo "  max qubits:      60"
echo "  max edges:       81"
echo "  max episode len: $MAX_STEPS"
echo "  topology:        tianyan176_66q.json"
echo "  curriculum:      large_n10 -> large_n20 -> tianyan"
echo "  GAE lambda:      adaptive 0.95 -> 0.995"
echo "=========================================================="

# ---- Phase 1: route-only (curriculum + adaptive lambda) ----
PYTHONPATH="$ROOT/src" python3 -u -m routing.rl.train_agent \
  --data-dir "$DATA_DIR" \
  --curriculum-keys large_n10,large_n20,tianyan \
  --topo "$TOPO" \
  --reward-mode routing \
  --timesteps "$PH1" \
  --max-episode-steps "$MAX_STEPS" \
  --max-num-qubits 60 \
  --mapping-budget 8 \
  --gae-adaptive \
  --gae-lam-min 0.95 \
  --gae-lam-max 0.995 \
  --device "$DEVICE" \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint-interval 20 \
  --out "$PH1_OUT"

# ---- Phase 2: noise-aware fine-tuning ----
if [ "$PHASE2" = "1" ]; then
  PYTHONPATH="$ROOT/src" python3 -u -m routing.rl.train_agent \
    --data-dir "$DATA_DIR" \
    --curriculum-keys large_n10,large_n20,tianyan \
    --topo "$TOPO" \
    --reward-mode noise_aware \
    --timesteps "$PH2" \
    --max-episode-steps "$MAX_STEPS" \
    --max-num-qubits 60 \
    --mapping-budget 8 \
    --gae-adaptive \
    --gae-lam-min 0.95 \
    --gae-lam-max 0.995 \
    --load "$PH1_OUT" \
    --device "$DEVICE" \
    --checkpoint-dir "${CKPT_DIR}_ph2" \
    --out "$PH2_OUT"
fi

echo "Done. Models in $MODELS_DIR"
