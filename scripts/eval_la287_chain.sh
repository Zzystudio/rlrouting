#!/usr/bin/env bash
# M4 评估链：LA287 / FTC287 × {argmax, beam5(route), beam5(la)} → NAM hybrid-T 保真度
# 训练（la287/ftc287 tmux 会话）完成后自动运行（由 watch_m4.sh 触发）。
# 口径与 p0t287 评估完全一致：tianyan287_20q、v2 混合精度 T=64/32/16、seed=42。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

ROUTE_FLAGS=(--circuit-dir "$ROOT/benchmark/nam_circs" \
  --topo "$ROOT/traindata/topo/tianyan287_20q.json" \
  --label-map "$ROOT/traindata/topo/tianyan287_20q_labels.json" \
  --max-num-qubits 20 --reward-mode routing --no-fidelity --device cuda:0 --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5)

route_one() {  # model outdir extra_flags...
  local model="$1" outdir="$2"; shift 2
  python3 -u -m routing.rl.generate_routing \
    --model "$ROOT/models/$model" --model-name "$outdir" \
    "${ROUTE_FLAGS[@]}" --out-dir "$ROOT/benchmark/routed/$outdir" "$@"
}

echo "=== [1/6] LA287 argmax routing ==="
route_one policy_LA287.pt la287_argmax
echo "=== [2/6] LA287 beam5(route) routing ==="
route_one policy_LA287.pt la287_beam5r --beam-width 5 --beam-vhead route
echo "=== [3/6] LA287 beam5(la) routing ==="
route_one policy_LA287.pt la287_beam5la --beam-width 5 --beam-vhead la
echo "=== [4/6] FTC287 argmax routing ==="
route_one policy_FTC287.pt ftc287_argmax
echo "=== [5/6] FTC287 beam5(route) routing ==="
route_one policy_FTC287.pt ftc287_beam5r --beam-width 5 --beam-vhead route

cd "$ROOT"
fid_one() {  # model_dir out_json
  python3 -u scripts/eval_nam_hybrid_T.py "$1" "$ROOT/benchmark/routed/$2" \
    tianyan287_20q
}

echo "=== [6/6] hybrid-T fidelity (GPU) ==="
fid_one la287_argmax     hybridT_la287_argmax_t287.json
fid_one la287_beam5r     hybridT_la287_beam5r_t287.json
fid_one la287_beam5la    hybridT_la287_beam5la_t287.json
fid_one ftc287_argmax    hybridT_ftc287_argmax_t287.json
fid_one ftc287_beam5r    hybridT_ftc287_beam5r_t287.json

echo "=== 汇总（mean / logmean / SWAPs / TRUNC）==="
python3 - << 'PYEOF'
import json, math, statistics, os
os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
def logmean(v):
    return math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in v)) if v else 0
files = {
    'SABRE':            'benchmark/routed/hybridT_sabre_t287.json',
    'p0t287_argmax':    'benchmark/routed/hybridT_p0t287_t287.json',
    'p0t287_beam5':     'benchmark/routed/hybridT_p0t287_beam5_t287.json',
    'LA287_argmax':     'benchmark/routed/hybridT_la287_argmax_t287.json',
    'LA287_beam5r':     'benchmark/routed/hybridT_la287_beam5r_t287.json',
    'LA287_beam5la':    'benchmark/routed/hybridT_la287_beam5la_t287.json',
    'FTC287_argmax':    'benchmark/routed/hybridT_ftc287_argmax_t287.json',
    'FTC287_beam5r':    'benchmark/routed/hybridT_ftc287_beam5r_t287.json',
}
print(f"{'method':<16s} {'fid_mean':>9s} {'logmean':>9s} {'swaps':>7s} {'TRUNC':>6s}")
for name, f in files.items():
    if not os.path.exists(f):
        print(f'{name:<16s} (missing)')
        continue
    rows = json.load(open(f))
    fids = [r['fid'] for r in rows]
    sws = [r['swaps'] for r in rows]
    ntr = sum(1 for r in rows if r.get('truncated'))
    print(f'{name:<16s} {sum(fids)/len(fids):>9.4f} {logmean(fids):>9.4f} '
          f'{sum(sws)/len(sws):>7.1f} {ntr:>3d}/19')
PYEOF
echo "=== M4 ALL DONE ==="
