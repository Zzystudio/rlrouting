#!/usr/bin/env bash
# 新训练模型全协议评估链（E1/E12 通用）：
#   1. NAM 19 argmax + beam5la+bud（基础拓扑）→ hybrid T=64
#   2. in-dist 50 argmax + beam5la+bud（基础拓扑）→ hybrid T
#   3. bighetero s42 argmax + beam5la+bud（held-out 强异构）→ hybrid T
#   4. NAM T=128 复核（近胜局判定）
# 用法: bash scripts/eval_newmodel_chain.sh <policy路径> <TAG> [device] [跳过T128=1]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"; TAG="$2"; DEV="${3:-cuda:0}"; SKIP_T128="${4:-0}"; EH="${5:-64}"
cd "$ROOT/src"

T287_TOPO="$ROOT/traindata/topo/tianyan287_20q.json"
T287_LABELS="$ROOT/traindata/topo/tianyan287_20q_labels.json"
BH_TOPO="$ROOT/traindata/topo/tianyan287_20q_bighetero.json"
BH_LABELS="$ROOT/traindata/topo/tianyan287_20q_bighetero_labels.json"

BASE=(--max-num-qubits 20 --reward-mode routing --no-fidelity --device "$DEV" --seed 0 \
  --edge-noise-features --beta-noise 0.5 \
  --reward-potential --w-err 0.02 --w-xt 0.01 --w-xt-swap 0.02 \
  --pot-progress-b 0.20 --shaping-gamma 0.99 --eta-shape 0.3 --alpha-ext 0.5 \
  --swap-price-scale 4.6 --lambda-budget 1.8 --budget-delta 1.05)

route_one() {
  local model="$1" outdir="$2"; shift 2
  python3 -u -m routing.rl.generate_routing \
    --model "$ROOT/models/$model" --model-name "$outdir" --edge-hidden "$EH" \
    "${BASE[@]}" --out-dir "$ROOT/benchmark/routed/$outdir" "$@"
}

echo "=== [1/8] $TAG NAM argmax routing ==="
route_one "$MODEL" "${TAG}_argmax" --circuit-dir "$ROOT/benchmark/nam_circs" \
  --topo "$T287_TOPO" --label-map "$T287_LABELS"
echo "=== [2/8] $TAG NAM beam5la+bud routing ==="
route_one "$MODEL" "${TAG}_beam5la_bud" --circuit-dir "$ROOT/benchmark/nam_circs" \
  --topo "$T287_TOPO" --label-map "$T287_LABELS" --beam-width 5 --beam-vhead la
echo "=== [3/8] $TAG in-dist argmax routing ==="
route_one "$MODEL" "${TAG}_argmax_indist" --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$T287_TOPO" --label-map "$T287_LABELS"
echo "=== [4/8] $TAG in-dist beam5la+bud routing ==="
route_one "$MODEL" "${TAG}_beam5la_bud_indist" --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$T287_TOPO" --label-map "$T287_LABELS" --beam-width 5 --beam-vhead la
echo "=== [5/8] $TAG bighetero argmax routing ==="
route_one "$MODEL" "${TAG}_bh_argmax" --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$BH_TOPO" --label-map "$BH_LABELS"
echo "=== [6/8] $TAG bighetero beam5la+bud routing ==="
route_one "$MODEL" "${TAG}_bh_beam5la_bud" --circuit-dir "$ROOT/benchmark/indist_val" \
  --topo "$BH_TOPO" --label-map "$BH_LABELS" --beam-width 5 --beam-vhead la

cd "$ROOT"
echo "=== [7/8] hybrid-T fidelity ==="
python3 -u scripts/eval_nam_hybrid_T.py "${TAG}_argmax" \
  "$ROOT/benchmark/routed/hybridT_${TAG}_argmax_t287.json" tianyan287_20q
python3 -u scripts/eval_nam_hybrid_T.py "${TAG}_beam5la_bud" \
  "$ROOT/benchmark/routed/hybridT_${TAG}_beam5la_bud_t287.json" tianyan287_20q
python3 -u scripts/eval_nam_hybrid_T.py "${TAG}_argmax_indist" \
  "$ROOT/benchmark/routed/hybridT_${TAG}_argmax_indist.json" tianyan287_20q \
  --circ-dir "$ROOT/benchmark/indist_val"
python3 -u scripts/eval_nam_hybrid_T.py "${TAG}_beam5la_bud_indist" \
  "$ROOT/benchmark/routed/hybridT_${TAG}_beam5la_bud_indist.json" tianyan287_20q \
  --circ-dir "$ROOT/benchmark/indist_val"
python3 -u scripts/eval_nam_hybrid_T.py "${TAG}_bh_argmax" \
  "$ROOT/benchmark/routed/hybridT_${TAG}_bh_argmax_indist.json" tianyan287_20q_bighetero \
  --circ-dir "$ROOT/benchmark/indist_val"
python3 -u scripts/eval_nam_hybrid_T.py "${TAG}_bh_beam5la_bud" \
  "$ROOT/benchmark/routed/hybridT_${TAG}_bh_beam5la_bud_indist.json" tianyan287_20q_bighetero \
  --circ-dir "$ROOT/benchmark/indist_val"

echo "=== [8/8] 汇总（含批效率）==="
python3 - << PYEOF
import json, math, statistics as st, os, sys
sys.path.insert(0, 'src')
os.chdir('$ROOT')
TAG = '$TAG'
from qiskit.qasm2 import loads as _qload
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo
_SABRE_GPS = {}
def _sabre_gps(topo_name, circ_dir):
    key = (topo_name, circ_dir)
    if key not in _SABRE_GPS:
        _cfg, _hw, _cm = load_topo(f'traindata/topo/{topo_name}.json')
        _SABRE_GPS[key] = {}
        for _f in sorted(os.listdir(circ_dir)):
            if not _f.endswith('.qasm'): continue
            _qc = _qload(open(f'{circ_dir}/{_f}').read())
            _ph, _info = sabre_route(_qc, _cfg, seed=0)
            _ops = _ph.count_ops()
            _nns = sum(v for k, v in _ops.items() if k != 'swap')
            _SABRE_GPS[key][_f] = _nns / max(1, _info['num_swaps'])
    return _SABRE_GPS[key]
def logmean(v):
    return math.exp(st.mean(math.log(max(x,1e-9)) for x in v)) if v else 0
import re as _re
def gps_from(dir, circ):
    # 全门口径（非 swap 门 / SWAP）：与 doc 基线（SABRE ≈6.0 / 模型 3.8）一致
    p = f'benchmark/routed/{dir}/{circ.removesuffix(".qasm")}.json'
    if not os.path.exists(p): return None
    d = json.load(open(p))
    ops = _re.findall(r'^(\w+) q', d['routed_qasm'], _re.M)
    n_non_swap = sum(1 for o in ops if o != 'swap')
    return n_non_swap / max(1, d['num_swaps'])
def show(tag, f, rdir=None, sabre_dir=None, sabre_topo=None):
    if not os.path.exists(f): print(f'{tag:<22s} (missing)'); return
    rows = json.load(open(f))
    fids=[r['fid'] for r in rows]; sws=[r['swaps'] for r in rows]
    ntr=sum(1 for r in rows if r.get('truncated'))
    gps=[]
    if rdir:
        gps=[gps_from(rdir, r['circuit']) for r in rows]
        gps=[g for g in gps if g is not None]
    elif sabre_dir:
        _g = _sabre_gps(sabre_topo or 'tianyan287_20q', sabre_dir)
        gps=[_g.get(r['circuit']) for r in rows]
        gps=[g for g in gps if g is not None]
    gps_med = f'{st.median(gps):.2f}' if gps else '--'
    print(f'{tag:<22s} fid={st.mean(fids):.4f} logmean={logmean(fids):.4f} '
          f'swaps={st.mean(sws):.1f} TRUNC={ntr}/{len(rows)} '
          f'gps_med={gps_med}' + (f' gps_n={len(gps)}' if len(gps)!=len(rows) else ''))
print(f'--- {TAG} 全协议汇总 ---')
show('SABRE(NAM)', 'benchmark/routed/hybridT_sabre_t287.json', sabre_dir='benchmark/nam_circs')
show(f'{TAG} argmax(NAM)', f'benchmark/routed/hybridT_{TAG}_argmax_t287.json', f'{TAG}_argmax')
show(f'{TAG} beam(NAM)', f'benchmark/routed/hybridT_{TAG}_beam5la_bud_t287.json', f'{TAG}_beam5la_bud')
show('SABRE(indist)', 'benchmark/routed/hybridT_sabre_indist.json', sabre_dir='benchmark/indist_val')
show(f'{TAG} argmax(indist)', f'benchmark/routed/hybridT_{TAG}_argmax_indist.json', f'{TAG}_argmax_indist')
show(f'{TAG} beam(indist)', f'benchmark/routed/hybridT_{TAG}_beam5la_bud_indist.json', f'{TAG}_beam5la_bud_indist')
show('SABRE(bighetero)', 'benchmark/routed/hybridT_sabre_bh_indist.json', sabre_dir='benchmark/indist_val', sabre_topo='tianyan287_20q_bighetero')
show(f'{TAG} argmax(bh)', f'benchmark/routed/hybridT_{TAG}_bh_argmax_indist.json', f'{TAG}_bh_argmax')
show(f'{TAG} beam(bh)', f'benchmark/routed/hybridT_{TAG}_bh_beam5la_bud_indist.json', f'{TAG}_bh_beam5la_bud')
PYEOF

if [ "$SKIP_T128" != "1" ]; then
  echo "=== T=128 NAM 复核 ==="
  python3 -u scripts/eval_nam_T128_model.py --tag "$TAG" \
    --out "$ROOT/benchmark/routed/audit_T128_${TAG}_nam.json" \
    --topo tianyan287_20q \
    --model-dirs "${TAG}_argmax:${TAG},${TAG}_beam5la_bud:${TAG}_beam"
fi
echo "=== EVAL DONE ($TAG) ==="
