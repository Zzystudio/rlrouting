#!/usr/bin/env bash
# A1-reval：推理侧预算约束（排序有效版）下的 LA287 beam 重评
#   [1] la287_beam5la_budget 路由（beam5 + V_LA + 预算惩罚）
#   [2] la287_beam5r_budget  路由（beam5 + route 打分 + 预算惩罚，G9 复测）
#   [3][4] 两口径 hybrid-T 保真度（GPU）
# 口径：NAM 19 × tianyan287_20q，T=64/32/16，seed=42；lambda_budget=0.5（与训练一致）
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

FLAGS=(--circuit-dir "$ROOT/benchmark/nam_circs" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 --beam-width 5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --lambda-budget 0.5 --budget-delta 1.05)

echo "=== [1/4] la287_beam5la_budget routing ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287.pt" --model-name la287_beam5la_budget \
  "${FLAGS[@]}" --beam-vhead la --out-dir "$ROOT/benchmark/routed/la287_beam5la_budget"

echo "=== [2/4] la287_beam5r_budget routing ==="
python3 -u -m routing.rl.generate_routing \
  --model "$ROOT/models/policy_LA287.pt" --model-name la287_beam5r_budget \
  "${FLAGS[@]}" --beam-vhead route --out-dir "$ROOT/benchmark/routed/la287_beam5r_budget"

cd "$ROOT"
echo "=== [3/4] hybrid-T fidelity ==="
python3 -u scripts/eval_nam_hybrid_T.py la287_beam5la_budget \
  "$ROOT/benchmark/routed/hybridT_la287_beam5la_budget_t287.json" tianyan287_20q
echo "=== [4/4] hybrid-T fidelity (route head) ==="
python3 -u scripts/eval_nam_hybrid_T.py la287_beam5r_budget \
  "$ROOT/benchmark/routed/hybridT_la287_beam5r_budget_t287.json" tianyan287_20q

echo "=== A1 汇总 ==="
python3 - << 'PYEOF'
import json, math, statistics, os
os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
def logmean(v):
    return math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v)) if v else 0
files = {
    'SABRE':              'benchmark/routed/hybridT_sabre_t287.json',
    'LA287_argmax':       'benchmark/routed/hybridT_la287_argmax_t287.json',
    'LA287_beam5la':      'benchmark/routed/hybridT_la287_beam5la_t287.json',
    'LA287_beam5la+bud':  'benchmark/routed/hybridT_la287_beam5la_budget_t287.json',
    'LA287_beam5r+bud':   'benchmark/routed/hybridT_la287_beam5r_budget_t287.json',
}
print(f"{'method':<20s} {'fid_mean':>9s} {'logmean':>9s} {'swaps':>7s} {'TRUNC':>6s}")
for name, f in files.items():
    if not os.path.exists(f):
        print(f'{name:<20s} (missing)')
        continue
    rows = json.load(open(f))
    fids = [r['fid'] for r in rows]
    sws = [r['swaps'] for r in rows]
    ntr = sum(1 for r in rows if r.get('truncated'))
    print(f'{name:<20s} {sum(fids)/len(fids):>9.4f} {logmean(fids):>9.4f} '
          f'{sum(sws)/len(sws):>7.1f} {ntr:>3d}/19')
PYEOF
echo "=== A1 ALL DONE ==="
