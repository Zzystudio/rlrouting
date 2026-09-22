#!/usr/bin/env bash
# Phase 1 诊断：in-distribution held-out 集基线评估
#   [1] SABRE 路由+保真度
#   [2] LA287 argmax
#   [3] FTC287 argmax（对照）
#   [4] LA287 beam5la+bud（预算约束口径）
#   [5] 分层判别分析
# 口径：v2 事件级模拟器混合精度 T=64/32/16，seed=42，tianyan287_20q
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"
VAL="$ROOT/benchmark/indist_val"

route_one() {  # model outdir extra...
  local model="$1" outdir="$2"; shift 2
  python3 -u -m routing.rl.generate_routing \
    --model "$ROOT/models/$model" --model-name "$outdir" \
    --circuit-dir "$VAL" --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
    --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
    --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
    --edge-noise-features --beta-noise 0.5 \
    --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
    --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
    --out-dir "$ROOT/benchmark/routed/$outdir" "$@"
}

echo "=== [1/5] SABRE ==="
cd "$ROOT"
python3 -u scripts/eval_nam_hybrid_T.py SABRE \
  "$ROOT/benchmark/routed/hybridT_sabre_indist.json" tianyan287_20q \
  --sabre --circ-dir "$VAL"

cd "$ROOT/src"
echo "=== [2/5] LA287 argmax ==="
route_one policy_LA287.pt la287_argmax_indist
echo "=== [3/5] FTC287 argmax ==="
route_one policy_FTC287.pt ftc287_argmax_indist
echo "=== [4/5] LA287 beam5la+bud ==="
route_one policy_LA287.pt la287_beam5la_bud_indist \
  --beam-width 5 --beam-vhead la --lambda-budget 0.5 --budget-delta 1.05

cd "$ROOT"
echo "=== [5/5] hybrid-T fidelity (agents) ==="
python3 -u scripts/eval_nam_hybrid_T.py la287_argmax_indist \
  "$ROOT/benchmark/routed/hybridT_la287_argmax_indist.json" tianyan287_20q --circ-dir "$VAL"
python3 -u scripts/eval_nam_hybrid_T.py ftc287_argmax_indist \
  "$ROOT/benchmark/routed/hybridT_ftc287_argmax_indist.json" tianyan287_20q --circ-dir "$VAL"
python3 -u scripts/eval_nam_hybrid_T.py la287_beam5la_bud_indist \
  "$ROOT/benchmark/routed/hybridT_la287_beam5la_bud_indist.json" tianyan287_20q --circ-dir "$VAL"

echo "=== Phase1 诊断汇总 ==="
python3 - << 'PYEOF'
import json, math, statistics, os
os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
def logmean(v):
    return math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v)) if v else 0
files = {
    'SABRE':          'benchmark/routed/hybridT_sabre_indist.json',
    'LA287_argmax':   'benchmark/routed/hybridT_la287_argmax_indist.json',
    'FTC287_argmax':  'benchmark/routed/hybridT_ftc287_argmax_indist.json',
    'LA287_b5la+bud': 'benchmark/routed/hybridT_la287_beam5la_bud_indist.json',
}
data = {}
print(f"{'method':<16s} {'fid_mean':>9s} {'logmean':>9s} {'swaps':>7s} {'TRUNC':>6s}")
for name, f in files.items():
    if not os.path.exists(f):
        print(f'{name:<16s} (missing)'); continue
    rows = json.load(open(f))
    data[name] = {r['circuit']: r for r in rows}
    fids = [r['fid'] for r in rows]
    sws = [r['swaps'] for r in rows]
    ntr = sum(1 for r in rows if r.get('truncated'))
    print(f'{name:<16s} {sum(fids)/len(fids):>9.4f} {logmean(fids):>9.4f} '
          f'{sum(sws)/len(sws):>7.1f} {ntr:>3d}/{len(rows)}')

if 'SABRE' in data and 'LA287_argmax' in data:
    sab, la = data['SABRE'], data['LA287_argmax']
    wins = sum(1 for c in sab if la[c]['fid'] > sab[c]['fid'] + 1e-9)
    loss = sum(1 for c in sab if la[c]['fid'] < sab[c]['fid'] - 1e-9)
    d = sum(la[c]['fid'] for c in sab)/len(sab) - sum(sab[c]['fid'] for c in sab)/len(sab)
    print(f'\n[判别] in-dist Δ(LA287_argmax − SABRE) = {d:+.4f}  '
          f'(win {wins} / lose {loss})')
    print('  Δ ≥ 0 → 覆盖缺口主导 → Phase 2（v3 补覆盖族 + 续训）')
    print('  Δ < 0 → 能力缺口主导 → Phase 3（规划层改进）优先')
PYEOF
echo "=== PHASE1 ALL DONE ==="
