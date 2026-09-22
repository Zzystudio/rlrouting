#!/usr/bin/env bash
# 监控 E1/E12 训练完成 → 自动触发对应评估链（双 GPU 并行）
# 用法: bash scripts/monitor_eval_launch.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

wait_done() {
  local log="$1" name="$2"
  echo "[monitor] 等待 $name 训练完成 ($log)"
  while ! grep -q "TRAIN_DONE" "$log" 2>/dev/null; do
    sleep 30
  done
  echo "[monitor] $name TRAIN_DONE @ $(date +%H:%M:%S)"
}

wait_done /tmp/opencode/la287e.log "E1(la287e)"
sleep 20
echo "[monitor] 启动 E1 评估链 (GPU0)"
tmux new-session -d -s eval_e "bash $ROOT/scripts/eval_newmodel_chain.sh policy_LA287e.pt la287e cuda:0 2>&1 | tee /tmp/opencode/la287e_eval.log"

wait_done /tmp/opencode/la287m.log "E12(la287m)"
sleep 20
echo "[monitor] 启动 E12 评估链 (GPU1)"
tmux new-session -d -s eval_m "bash $ROOT/scripts/eval_newmodel_chain.sh policy_LA287m.pt la287m cuda:1 2>&1 | tee /tmp/opencode/la287m_eval.log"

echo "[monitor] 两条评估链均已启动 @ $(date +%H:%M:%S)"
