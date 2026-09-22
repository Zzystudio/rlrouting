#!/usr/bin/env bash
# la287d/la287ctl 校准+蒸馏训练完成后的评估链
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

NAM_FLAGS=(--circuit-dir "$ROOT/benchmark/nam_circs" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05)

IND_FLAGS=(--circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05)

echo "=== [1/8] LA287d argmax routing (NAM) ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287d.pt" --model-name la287d_argmax \
  "${NAM_FLAGS[@]}" --out-dir "$ROOT/benchmark/routed/la287d_argmax"

echo "=== [2/8] LA287d beam5la+bud routing (NAM) ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287d.pt" --model-name la287d_beam5la_bud \
  "${NAM_FLAGS[@]}" --beam-width 5 --beam-vhead la \
  --out-dir "$ROOT/benchmark/routed/la287d_beam5la_bud"

cd "$ROOT"
echo "=== [3/8] NAM hybrid-T fidelity ==="
python3 -u scripts/eval_nam_hybrid_T.py la287d_argmax \
  "$ROOT/benchmark/routed/hybridT_la287d_argmax_t287.json" tianyan287_20q
python3 -u scripts/eval_nam_hybrid_T.py la287d_beam5la_bud \
  "$ROOT/benchmark/routed/hybridT_la287d_beam5la_bud_t287.json" tianyan287_20q

cd "$ROOT/src"
echo "=== [4/8] LA287d argmax routing (in-dist) ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287d.pt" --model-name la287d_argmax_indist \
  --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05 \
  --out-dir "$ROOT/benchmark/routed/la287d_argmax_indist"

echo "=== [5/8] LA287d beam5la+bud routing (in-dist) ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287d.pt" --model-name la287d_beam5la_bud_indist \
  --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 --beam-width 5 --beam-vhead la \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05 \
  --out-dir "$ROOT/benchmark/routed/la287d_beam5la_bud_indist"

echo "=== [6/8] LA287ctl argmax routing (in-dist) ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287ctl.pt" --model-name la287ctl_argmax_indist \
  --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05 \
  --out-dir "$ROOT/benchmark/routed/la287ctl_argmax_indist"

cd "$ROOT"
echo "=== [7/8] in-dist hybrid-T fidelity ==="
python3 -u scripts/eval_nam_hybrid_T.py la287d_argmax_indist \
  "$ROOT/benchmark/routed/hybridT_la287d_argmax_indist.json" tianyan287_20q \
  --circ-dir "$ROOT/benchmark/indist_val"
python3 -u scripts/eval_nam_hybrid_T.py la287d_beam5la_bud_indist \
  "$ROOT/benchmark/routed/hybridT_la287d_beam5la_bud_indist.json" tianyan287_20q \
  --circ-dir "$ROOT/benchmark/indist_val"
python3 -u scripts/eval_nam_hybrid_T.py la287ctl_argmax_indist \
  "$ROOT/benchmark/routed/hybridT_la287ctl_argmax_indist.json" tianyan287_20q \
  --circ-dir "$ROOT/benchmark/indist_val"

echo "=== [8/8] 汇总 ==="
python3 - << 'PYEOF'
import json, math, statistics, os
os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
def logmean(v):
    return math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v)) if v else 0
def show(tag, f):
    if not os.path.exists(f):
        print(f'{tag:<24s} (missing)'); return
    rows = json.load(open(f))
    fids = [r['fid'] for r in rows]; sws = [r['swaps'] for r in rows]
    ntr = sum(1 for r in rows if r.get('truncated'))
    gps = [r['gates']/max(1,r['swaps']) for r in rows] if 'gates' in rows[0] else None
    print(f'{tag:<24s} fid={sum(fids)/len(fids):.4f} logmean={logmean(fids):.4f} '
          f'swaps={sum(sws)/len(sws):.1f} TRUNC={ntr}/{len(rows)}')

print('--- NAM (tianyan287_20q) ---')
show('SABRE', 'benchmark/routed/hybridT_sabre_t287.json')
show('LA287 argmax', 'benchmark/routed/hybridT_la287_argmax_t287.json')
show('LA287c argmax', 'benchmark/routed/hybridT_la287c_argmax_t287.json')
show('LA287d argmax', 'benchmark/routed/hybridT_la287d_argmax_t287.json')
show('LA287d b5la+bud', 'benchmark/routed/hybridT_la287d_beam5la_bud_t287.json')
print('--- in-dist (50 条) ---')
show('SABRE', 'benchmark/routed/hybridT_sabre_indist.json')
show('LA287 argmax', 'benchmark/routed/hybridT_la287_argmax_indist.json')
show('LA287c argmax', 'benchmark/routed/hybridT_la287c_argmax_indist.json')
show('LA287d argmax', 'benchmark/routed/hybridT_la287d_argmax_indist.json')
show('LA287d b5la+bud', 'benchmark/routed/hybridT_la287d_beam5la_bud_indist.json')
show('LA287ctl argmax', 'benchmark/routed/hybridT_la287ctl_argmax_indist.json')
PYEOF
echo "=== EVAL ALL DONE ==="
