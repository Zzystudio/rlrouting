#!/usr/bin/env bash
# P0-t287 V5 评估链（精简版 v2）：仅新模型 vs SABRE，NAM held-out × tianyan287_20q
#   [1] SABRE hybrid-T 保真度（现场 sabre_route seed=0, decay 20 trials）
#   [2] p0t287 argmax hybrid-T 保真度（路由已完成: benchmark/routed/p0t287/）
#   [3] p0t287 beam5 路由（beam 打分与训练奖励同构：potential + w_* + Φ shaping）
#   [4] p0t287 beam5 hybrid-T 保真度
# 混合精度：≤9q T=16 / 10-14q T=32 / ≥15q T=64, seed=42
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"   # eval_nam_hybrid_T.py 使用仓库相对路径（traindata/ benchmark/）
TOPO="$ROOT/traindata/topo/tianyan287_20q.json"
LABELS="$ROOT/traindata/topo/tianyan287_20q_labels.json"
NAM="$ROOT/benchmark/nam_circs"

echo "=== [1/4] SABRE hybrid-T fidelity ==="
python3 -u "$ROOT/scripts/eval_nam_hybrid_T.py" SABRE \
  "$ROOT/benchmark/routed/hybridT_sabre_t287.json" tianyan287_20q --sabre

echo "=== [2/4] p0t287 argmax hybrid-T fidelity ==="
python3 -u "$ROOT/scripts/eval_nam_hybrid_T.py" p0t287 \
  "$ROOT/benchmark/routed/hybridT_p0t287_t287.json" tianyan287_20q

echo "=== [3/4] p0t287 beam5 routing ==="
cd "$ROOT/src"
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_p0_t287.pt" --model-name p0t287_beam5 \
  --circuit-dir "$NAM" --topo "$TOPO" --label-map "$LABELS" \
  --max-num-qubits 20 --out-dir "$ROOT/benchmark/routed/p0t287_beam5" \
  --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 --beam-width 5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5

echo "=== [4/4] p0t287 beam5 hybrid-T fidelity ==="
cd "$ROOT"
python3 -u "$ROOT/scripts/eval_nam_hybrid_T.py" p0t287_beam5 \
  "$ROOT/benchmark/routed/hybridT_p0t287_beam5_t287.json" tianyan287_20q

echo "=== ALL DONE ==="
