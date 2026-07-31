#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../src"

echo "========================================="
echo "Phase 1: AlphaZero routing (no simulator)"
echo "========================================="
python3 -m routing.rl.train_agent \
  --mode alphazero \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode routing \
  --timesteps 30000 \
  --mcts-simulations 50 \
  --self-play-episodes 8 \
  --alphazero-train-steps 100 \
  --load ../models/policy_phase1.pt \
  --out ../models/policy_alphazero_routing.pt

echo ""
echo "================================================"
echo "Phase 2: AlphaZero noise_aware fine-tune"
echo "================================================"
python3 -m routing.rl.train_agent \
  --mode alphazero \
  --topo-list ../traindata/topo/cross_5q.json,../traindata/topo/ring_5q.json,../traindata/topo/ibmq_5_line.json \
  --reward-mode noise_aware \
  --timesteps 20000 \
  --mcts-simulations 50 \
  --self-play-episodes 8 \
  --alphazero-train-steps 100 \
  --load ../models/policy_alphazero_routing.pt \
  --out ../models/policy_alphazero_noiseaware.pt

echo ""
echo "Done! Models saved to:"
echo "  ../models/policy_alphazero_routing.pt"
echo "  ../models/policy_alphazero_noiseaware.pt"
