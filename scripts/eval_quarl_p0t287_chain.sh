#!/usr/bin/env bash
# QUARL 第二评估集链（doc/20260915训练方案.md §7.0 定稿：18 条 ≤20q，held-out）
#   [1] p0t287 argmax 路由（GPU）
#   [2] p0t287 beam5 路由（GPU）
#   [3] SABRE hybrid-T 保真度（GPU 加速模拟器）
#   [4] p0t287 argmax hybrid-T 保真度
#   [5] p0t287 beam5 hybrid-T 保真度
# 混合精度 T：≤9q T=16 / 10-14q T=32 / ≥15q T=64, seed=42
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TOPO="$ROOT/traindata/topo/tianyan287_20q.json"
LABELS="$ROOT/traindata/topo/tianyan287_20q_labels.json"
QUARL="$ROOT/benchmark/quarl_opt_wo_rm"

echo "=== [1/5] p0t287 argmax routing (QUARL) ==="
cd "$ROOT/src"
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_p0_t287.pt" --model-name p0t287_quarl \
  --circuit-dir "$QUARL" --topo "$TOPO" --label-map "$LABELS" \
  --max-num-qubits 20 --out-dir "$ROOT/benchmark/routed/p0t287_quarl" \
  --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5

echo "=== [2/5] p0t287 beam5 routing (QUARL) ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_p0_t287.pt" --model-name p0t287_beam5_quarl \
  --circuit-dir "$QUARL" --topo "$TOPO" --label-map "$LABELS" \
  --max-num-qubits 20 --out-dir "$ROOT/benchmark/routed/p0t287_beam5_quarl" \
  --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 --beam-width 5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5

cd "$ROOT"
echo "=== [3/5] SABRE hybrid-T fidelity (QUARL) ==="
python3 -u "$ROOT/scripts/eval_nam_hybrid_T.py" SABRE \
  "$ROOT/benchmark/routed/hybridT_sabre_quarl.json" tianyan287_20q \
  --sabre --circ-dir "$QUARL"

echo "=== [4/5] p0t287 argmax hybrid-T fidelity (QUARL) ==="
python3 -u "$ROOT/scripts/eval_nam_hybrid_T.py" p0t287_quarl \
  "$ROOT/benchmark/routed/hybridT_p0t287_quarl.json" tianyan287_20q \
  --circ-dir "$QUARL"

echo "=== [5/5] p0t287 beam5 hybrid-T fidelity (QUARL) ==="
python3 -u "$ROOT/scripts/eval_nam_hybrid_T.py" p0t287_beam5_quarl \
  "$ROOT/benchmark/routed/hybridT_p0t287_beam5_quarl.json" tianyan287_20q \
  --circ-dir "$QUARL"

echo "=== ALL DONE ==="
