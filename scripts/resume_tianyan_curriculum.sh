#!/usr/bin/env bash
# 断点续训: 从上一个 checkpoint 精确恢复 (权重 + optimizer + step + metrics 追加),
# 并自动选择空闲显存最多的 GPU (防止被 llama-server 占满的卡再次 OOM).
#
# Usage:
#   scripts/resume_tianyan_curriculum.sh [PHASE1_STEPS] [MIN_FREE_MIB]
# Example:
#   scripts/resume_tianyan_curriculum.sh 800000 4000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
PH1="${1:-800000}"
MIN_FREE_MIB="${2:-4000}"

DATA_DIR="$ROOT/traindata"
MODELS_DIR="$ROOT/models"
TOPO="$DATA_DIR/topo/tianyan176_66q.json"
CKPT_DIR="$MODELS_DIR/ckpts_tianyan_curric"
PH1_OUT="$MODELS_DIR/policy_tianyan176_curric_phase1.pt"
MAX_STEPS=800

# ---- 选最新 checkpoint ----
CKPT=$(ls -1 "$CKPT_DIR"/ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ]; then
  echo "No checkpoint found in $CKPT_DIR" >&2
  exit 1
fi
echo "Resuming from: $CKPT"

# ---- 选空闲显存最多的 GPU ----
DEVICE=""
BEST=0
for i in $(seq 0 7); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i" | tr -d ' ')
  if [ "$FREE" -gt "$BEST" ]; then
    BEST=$FREE
    DEVICE=$i
  fi
done
if [ "$BEST" -lt "$MIN_FREE_MIB" ]; then
  echo "WARNING: GPU$DEVICE only has ${BEST} MiB free (< ${MIN_FREE_MIB} MiB). Proceeding anyway." >&2
fi
echo "Using GPU $DEVICE (${BEST} MiB free)"

echo "=========================================================="
echo "Tianyan 176 Curriculum RESUME (60q / 81 edges)"
echo "  device:          cuda:$DEVICE"
echo "  resume step from: $CKPT"
echo "  phase 1 steps:    $PH1 (总步数，含已完成步骤)"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=$DEVICE \
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
  --device cuda:0 \
  --load "$CKPT" \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint-interval 20 \
  --out "$PH1_OUT"

echo "Done. Models in $MODELS_DIR"