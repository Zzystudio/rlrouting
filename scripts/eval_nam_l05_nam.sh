#!/usr/bin/env bash
# 新模型（l05+NAM）NAM benchmark 评估：路由 → 保真度 → 对比。
# 用法: bash scripts/eval_nam_l05_nam.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/policy_l05_nam_phase1.pt}
NAME=${NAME:-l05_nam}

# 1) 路由生成（argmax，无保真度，与 l05 同口径）
PYTHONPATH=src python3 -u -m routing.rl.generate_routing \
  --model "$MODEL" --model-name "$NAME" \
  --circuit-dir benchmark/nam_circs \
  --topo traindata/topo/tianyan176_20q.json \
  --label-map traindata/topo/tianyan176_20q_labels.json \
  --max-num-qubits 20 \
  --out-dir "benchmark/routed/$NAME" \
  --no-fidelity

# 2) 保真度计算（trajectory_sched ×16, seed=0, scheduled）
PYTHONPATH=src python3 -u scripts/compute_fidelity_model.py "$NAME"

# 3) 对比
PYTHONPATH=src python3 scripts/compare_nam.py "$NAME" l05