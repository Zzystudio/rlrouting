#!/usr/bin/env bash
# l05 的第一阶段（routing + layout-mix）训练，训练集混入 NAM 电路。
# 用法: bash scripts/train_l05_nam.sh <device> <timesteps>
set -euo pipefail

DEVICE="${1:-cuda:0}"
TIMESTEPS="${2:-100000}"

cd "$(dirname "$0")/.."

PYTHONPATH=src python3 -u -m routing.rl.train_agent \
  --data-dir traindata \
  --split-prefix tianyan20q \
  --topo-list traindata/topo/tianyan176_20q.json \
  --reward-mode routing \
  --timesteps "$TIMESTEPS" \
  --max-episode-steps 400 \
  --max-num-qubits 20 \
  --mapping-budget 8 \
  --use-scheduler --eta-xtalk-par 0.05 --swap-cost 0 \
  --layout-mix 0.3,0.3,0.4 --lambda-layout 0.5 \
  --sabre-layout-trials 5 --sabre-cache-file models/sabre_cache_tianyan20q.pkl \
  --nam-circuits-dir benchmark/nam_circs --nam-circuit-prob 0.5 --nam-max-qubits 20 \
  --load models/policy_tianyan20q_ft_phase1_sched_fix_eta05.pt \
  --rollout-steps 256 --epochs 4 --lr 3e-4 --seed 0 \
  --checkpoint-dir models/ckpts_l05_nam_phase1 \
  --checkpoint-interval 20 \
  --out models/policy_l05_nam_phase1.pt \
  --device "$DEVICE"