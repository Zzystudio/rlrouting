#!/usr/bin/env bash
# 方向2 Phase 1：单臂训练（SABRE 脚本路由 + RL 纯调度）
#   ARM ∈ {analytic, traj, hybrid}——只变终端保真信号，其余控制变量全同
# Usage: scripts/train_sched_arm.sh <ARM> [DEVICE] [TIMESTEPS]
set -uo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?ARM: analytic|traj|hybrid}"
DEVICE="${2:-cuda:0}"
STEPS="${3:-150000}"
TOPO="../traindata/topo/tianyan287_20q.json"
SREF="../traindata/routed/tianyan287_20q_sref.json"
NAMDIR="../traindata/gen_structured_v2,../traindata/gen_structured_v3"
LOG=/tmp/opencode
mkdir -p "$LOG" models

EXTRA=""
if [ "$ARM" = "traj" ] || [ "$ARM" = "hybrid" ]; then
  EXTRA="--traj-trajectories 16 --sim-device $DEVICE"
fi

echo "[arm $ARM] start $(date)"
(cd src && PYTHONPATH=. python3 -u -m routing.rl.train_agent \
    --clocked --sched-only --no-gnn \
    --topo "$TOPO" \
    --curriculum-keys tianyan20q \
    --reward-mode noise_aware \
    --sref-cache "$SREF" \
    --nam-circuits-dir "$NAMDIR" --nam-circuit-prob 0.3 \
    --max-episode-steps 800 --step-cap-factor 2.0 \
    --timesteps "$STEPS" --rollout-steps 512 \
    --seed 0 \
    --fid-arm "$ARM" $EXTRA \
    --out "../models/sched_arm_${ARM}.pt" \
    > "$LOG/train_sched_${ARM}.log" 2>&1)
echo "[arm $ARM] done $(date)"
