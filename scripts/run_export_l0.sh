#!/usr/bin/env bash
# Layer 0：全数据集 SABRE 路由缓存 + per-circuit sref（方向2 Phase 1 前置）
# 路由 pass：3 分片并行（CPU）；sref pass：GPU 串行（ASAP fid，解析+traj_v3）
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.

E="python3 scripts/export_sabre_routed.py --topo traindata/topo/tianyan287_20q.json"
SPL="--splits tianyan20q_phase1,tianyan20q_phase2,tianyan20q_phase3,tianyan20q_mixed,tianyan20q_test"
NAMS="traindata/gen_structured_v2 traindata/gen_structured_v3 benchmark/indist_val benchmark/nam_circs"
SREF="traindata/routed/tianyan287_20q_sref.json"
LOG=/tmp/opencode
mkdir -p "$LOG"

echo "[run_export_l0] routing pass start $(date)"
for i in 0 1 2; do
  ( $E $SPL --shard $i/3 > "$LOG/exp_r$i.log" 2>&1 ) &
done
for d in $NAMS; do
  b=$(basename "$d")
  for i in 0 1 2; do
    ( $E --nam-dir "$d" --shard $i/3 > "$LOG/exp_${b}_r$i.log" 2>&1 ) &
  done
done
wait
echo "[run_export_l0] routing pass done $(date)"

echo "[run_export_l0] sref pass start $(date)"
$E $SPL --with-sref --sref-out "$SREF" > "$LOG/exp_sref_splits.log" 2>&1
for d in traindata/gen_structured_v2 traindata/gen_structured_v3; do
  $E --nam-dir "$d" --with-sref --sref-out "$SREF" > "$LOG/exp_sref_$(basename "$d").log" 2>&1
done
echo "[run_export_l0] sref pass done $(date)"
