#!/usr/bin/env bash
# 拓扑泛化测试：in-dist 50 条 × tianyan287_20q_stronghetero（训练未见拓扑）
# 所有模型在 stronghetero 上重新路由（布局+交换决策全部在新图上重做）+ 混合精度模拟
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

SH_TOPO="$ROOT/traindata/topo/tianyan287_20q_stronghetero.json"
SH_LABELS="$ROOT/traindata/topo/tianyan287_20q_stronghetero_labels.json"
VAL="$ROOT/benchmark/indist_val"

BASE=(--circuit-dir "$VAL" \
  --topo "$SH_TOPO" \
  --label-map "$SH_LABELS" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05)

route_one() {
  local model="$1" outdir="$2"; shift 2
  python3 -u -m routing.rl.generate_routing \
    --model "$ROOT/models/$model" --model-name "$outdir" \
    "${BASE[@]}" --out-dir "$ROOT/benchmark/routed/$outdir" "$@"
}

echo "=== [1/6] LA287d argmax routing (stronghetero) ==="
route_one policy_LA287d.pt la287d_sh_argmax
echo "=== [2/6] LA287d beam5la+bud routing (stronghetero) ==="
route_one policy_LA287d.pt la287d_beam5la_bud_sh --beam-width 5 --beam-vhead la
echo "=== [3/6] LA287ctl argmax routing (stronghetero) ==="
route_one policy_LA287ctl.pt la287ctl_argmax_sh
echo "=== [4/6] LA287c argmax routing (stronghetero) ==="
route_one policy_LA287c.pt la287c_argmax_sh

cd "$ROOT"
echo "=== [5/6] SABRE + hybrid-T fidelity (stronghetero) ==="
python3 -u scripts/eval_nam_hybrid_T.py SABRE \
  "$ROOT/benchmark/routed/hybridT_sabre_shetero_indist.json" tianyan287_20q_stronghetero \
  --sabre --circ-dir "$VAL"
for m in la287d_argmax la287d_beam5la_bud la287ctl_argmax la287c_argmax; do
  python3 -u scripts/eval_nam_hybrid_T.py "$m" \
    "$ROOT/benchmark/routed/hybridT_${m}_shetero_indist.json" tianyan287_20q_stronghetero \
    --circ-dir "$VAL"
done

echo "=== [6/6] 汇总 ==="
python3 - << 'PYEOF'
import json, math, statistics, os
os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
def logmean(v):
    return math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v)) if v else 0
files = {
    'SABRE':        'benchmark/routed/hybridT_sabre_shetero_indist.json',
    'LA287':        'benchmark/routed/hybridT_la287_argmax_shetero_indist.json',
    'LA287c':       'benchmark/routed/hybridT_la287c_argmax_shetero_indist.json',
    'LA287d':       'benchmark/routed/hybridT_la287d_argmax_shetero_indist.json',
    'LA287d_b5la':  'benchmark/routed/hybridT_la287d_beam5la_bud_shetero_indist.json',
    'LA287ctl':     'benchmark/routed/hybridT_la287ctl_argmax_shetero_indist.json',
}
print(f"{'method':<14s} {'fid_mean':>9s} {'logmean':>9s} {'swaps':>7s} {'TRUNC':>7s} {'胜SABRE':>7s}")
sab_f = json.load(open(files['SABRE']))
sab_map = {r['circuit']: r['fid'] for r in sab_f}
for name, f in files.items():
    if not os.path.exists(f):
        print(f'{name:<14s} (missing)'); continue
    rows = json.load(open(f))
    fids = [r['fid'] for r in rows]; sws = [r['swaps'] for r in rows]
    ntr = sum(1 for r in rows if r.get('truncated'))
    w = sum(1 for r in rows if r['method'] != 'SABRE' and r['fid'] > sab_map.get(r['circuit'], 9) + 1e-9)
    nn = len(fids)
    wt = f'{w}/{nn}' if name != 'SABRE' else '-'
    print(f'{name:<14s} {sum(fids)/nn:>9.4f} {logmean(fids):>9.4f} {sum(sws)/nn:>7.1f} {ntr:>3d}/{nn} {wt:>7s}')
PYEOF
echo "=== SH-INDIST ALL DONE ==="
