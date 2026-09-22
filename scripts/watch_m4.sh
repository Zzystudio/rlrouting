#!/usr/bin/env bash
# 看守：等待 la287/ftc287 训练完成（TRAIN_DONE 标记），自动触发 M4 评估链。
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_LA=/tmp/opencode/la287.log
LOG_FTC=/tmp/opencode/ftc287.log
while true; do
  LA=$(grep -c TRAIN_DONE "$LOG_LA" 2>/dev/null || echo 0)
  FTC=$(grep -c TRAIN_DONE "$LOG_FTC" 2>/dev/null || echo 0)
  if [ "$LA" -ge 1 ] && [ "$FTC" -ge 1 ]; then
    echo "$(date) both TRAIN_DONE → start M4 eval chain" >> /tmp/opencode/m4_watcher.log
    bash "$ROOT/scripts/eval_la287_chain.sh" > /tmp/opencode/m4_eval.log 2>&1
    echo "$(date) M4 chain finished" >> /tmp/opencode/m4_watcher.log
    break
  fi
  sleep 120
done
