#!/usr/bin/env bash
# 强异构变体（tianyan287_20q_bighetero：训练子拓扑只改噪声数值，耦合图不变）
# in-dist 50 条：LA287c 之后的模型（LA287ctl/LA287d）argmax + beam5la+bud vs SABRE
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

BH_TOPO="$ROOT/traindata/topo/tianyan287_20q_bighetero.json"
BH_LABELS="$ROOT/traindata/topo/tianyan287_20q_bighetero_labels.json"
VAL="$ROOT/benchmark/indist_val"

BASE=(--circuit-dir "$VAL" \
  --topo "$BH_TOPO" \
  --label-map "$BH_LABELS" \
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

echo "=== [1/8] LA287d argmax routing (bighetero) ==="
route_one policy_LA287d.pt la287d_bh_argmax
echo "=== [2/8] LA287d beam5la+bud routing (bighetero) ==="
route_one policy_LA287d.pt la287d_bh_beam5la_bud --beam-width 5 --beam-vhead la
echo "=== [3/8] LA287ctl argmax routing (bighetero) ==="
route_one policy_LA287ctl.pt la287ctl_bh_argmax
echo "=== [4/8] LA287ctl beam5la+bud routing (bighetero) ==="
route_one policy_LA287ctl.pt la287ctl_bh_beam5la_bud --beam-width 5 --beam-vhead la
echo "=== [5/8] LA287c argmax routing (bighetero, 参照) ==="
route_one policy_LA287c.pt la287c_bh_argmax

cd "$ROOT"
echo "=== [6/8] SABRE + hybrid-T fidelity (bighetero) ==="
python3 -u scripts/eval_nam_hybrid_T.py SABRE \
  "$ROOT/benchmark/routed/hybridT_sabre_bh_indist.json" tianyan287_20q_bighetero \
  --sabre --circ-dir "$VAL"
for m in la287d_bh_argmax la287d_bh_beam5la_bud la287ctl_bh_argmax la287ctl_bh_beam5la_bud la287c_bh_argmax; do
  python3 -u scripts/eval_nam_hybrid_T.py "$m" \
    "$ROOT/benchmark/routed/hybridT_${m}_indist.json" tianyan287_20q_bighetero \
    --circ-dir "$VAL"
done

echo "=== [7/8] 汇总 ==="
python3 - << 'PYEOF'
import json, math, statistics, os
os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
def logmean(v):
    return math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v)) if v else 0
files = {
    'SABRE':            'benchmark/routed/hybridT_sabre_bh_indist.json',
    'LA287c argmax':    'benchmark/routed/hybridT_la287c_bh_argmax_indist.json',
    'LA287ctl argmax':  'benchmark/routed/hybridT_la287ctl_bh_argmax_indist.json',
    'LA287ctl b5la+bud':'benchmark/routed/hybridT_la287ctl_bh_beam5la_bud_indist.json',
    'LA287d argmax':    'benchmark/routed/hybridT_la287d_bh_argmax_indist.json',
    'LA287d b5la+bud':  'benchmark/routed/hybridT_la287d_bh_beam5la_bud_indist.json',
}
print('--- tianyan287_20q_bighetero × in-dist 50 (hybrid T, seed=42) ---')
base_fid = None
for tag, f in files.items():
    if not os.path.exists(f):
        print(f'{tag:<20s} (missing)'); continue
    rows = json.load(open(f))
    fids = [r['fid'] for r in rows]; sws = [r['swaps'] for r in rows]
    ntr = sum(1 for r in rows if r.get('truncated'))
    m = sum(fids)/len(fids)
    rel = '' if base_fid is None else f'  ({(m-base_fid)/base_fid*100:+.1f}% vs SABRE)'
    if base_fid is None: base_fid = m
    print(f'{tag:<20s} fid={m:.4f} logmean={logmean(fids):.4f} '
          f'swaps={sum(sws)/len(sws):.1f} TRUNC={ntr}/{len(rows)}{rel}')
PYEOF
echo "=== EVAL ALL DONE ==="
