#!/usr/bin/env bash
# Train policies for larger circuits (n = 8..20) at multiple scales.
#
# One policy per scale (qubit count): the env observation dimension and the
# topologies both depend on n, so models are not transferable across scales.
#
# Timesteps are scaled per scale:  base * (n / 5)^1.5
#   n=5 (reference) -> base         (100k)
#   n=8  -> ~2.0x,  n=10 -> ~2.8x,  n=12 -> ~3.7x,  n=16 -> ~5.7x,  n=20 -> 8x
# because episodes at large n are 4-8x longer (more 2q gates + larger
# topology diameter) and each scale has ~1000+ distinct circuits.
#
# Usage:
#   scripts/train_large.sh [TIMESTEPS_BASE] [DEVICE] [SCALES]
#   TIMESTEPS_BASE: PPO timesteps for n=5-equivalent workload (default 100000)
#   DEVICE:         torch device, e.g. cuda:0 / cpu (default cuda:0)
#   SCALES:         comma-separated subset to train, e.g. 8,10 (default all)
#
# Set PHASE2=0 to skip noise-aware fine-tuning (default: PHASE2=1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
TIMESTEPS_BASE="${1:-100000}"
DEVICE="${2:-cuda:0}"
SCALES="${3:-8,10,12,16,20}"
PHASE2="${PHASE2:-1}"

DATA_DIR="$ROOT/traindata"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

# scale -> topology list (all topologies of that scale are mixed per-episode)
declare -A SCALE_TOPOS=(
  [8]="line_8q.json,ring_8q.json"
  [10]="line_10q.json,ring_10q.json,grid_2x5_10q.json"
  [12]="line_12q.json,ring_12q.json,grid_3x4_12q.json"
  [16]="line_16q.json,ring_16q.json,grid_4x4_16q.json"
  [20]="line_20q.json,ring_20q.json,grid_5x4_20q.json"
)

for n in ${SCALES//,/ }; do
  TIMESTEPS=$(python3 -c "print(int(round($TIMESTEPS_BASE * ($n / 5.0) ** 1.5)))")
  TOPOS="${SCALE_TOPOS[$n]}"
  TOPO_ARGS=""
  for t in ${TOPOS//,/ }; do
    TOPO_ARGS="$TOPO_ARGS,$DATA_DIR/topo/$t"
  done
  TOPO_ARGS="${TOPO_ARGS:1}"
  PREFIX="large_n${n}"
  MAX_STEPS=$((250 + n * 15))
  PH1_OUT="$MODELS_DIR/policy_${PREFIX}_phase1.pt"
  PH2_OUT="$MODELS_DIR/policy_${PREFIX}_noiseaware.pt"

  echo "==================================================================="
  echo "Scale n=$n  timesteps=$TIMESTEPS  topologies: $TOPOS  max_episode_steps=$MAX_STEPS"
  echo "==================================================================="

  # ---- Phase 1: route-only (SABRE features + distance reward) ----
  PYTHONPATH="$ROOT/src" python3 -m routing.rl.train_agent \
    --data-dir "$DATA_DIR" \
    --split-prefix "$PREFIX" \
    --topo-list "$TOPO_ARGS" \
    --reward-mode routing \
    --timesteps "$TIMESTEPS" \
    --max-episode-steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --out "$PH1_OUT"

  # ---- Phase 2 (optional): noise-aware fine-tuning ----
  if [ "$PHASE2" = "1" ]; then
    PYTHONPATH="$ROOT/src" python3 -m routing.rl.train_agent \
      --data-dir "$DATA_DIR" \
      --split-prefix "$PREFIX" \
      --topo-list "$TOPO_ARGS" \
      --reward-mode noise_aware \
      --timesteps "$TIMESTEPS" \
      --max-episode-steps "$MAX_STEPS" \
      --load "$PH1_OUT" \
      --device "$DEVICE" \
      --out "$PH2_OUT"
  fi
done

echo "Done. Models in $MODELS_DIR"
