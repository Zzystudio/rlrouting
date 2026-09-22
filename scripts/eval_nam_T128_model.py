"""NAM 19 条 T=128 复核（通用）：SABRE 现场路由 + 模型路由目录读取。

用法:
  python3 scripts/eval_nam_T128_model.py \
    --tag LA287e --out benchmark/routed/audit_T128_la287e_nam.json \
    --topo tianyan287_20q \
    --model-dirs la287e_argmax:LA287e,la287e_beam5la_bud:LA287e_beam
断点续传：已有 OUT 的方法+电路组合直接复用。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events as _fid_v2
from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3 as _fid_v3
from routing.routing import sabre_route
from routing.rl.eval_policy import load_topo

SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
SIM_VERSION = 'v2'
if '--sim' in sys.argv:
    SIM_VERSION = sys.argv[sys.argv.index('--sim') + 1]
_FID = _fid_v3 if SIM_VERSION == 'v3' else _fid_v2
T_DEEP = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="模型标签（输出/打印用）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topo", default="tianyan287_20q")
    ap.add_argument("--model-dirs", required=True,
                    help="逗号分隔 dir:label（相对 benchmark/routed/）")
    ap.add_argument("--circ-dir", default="benchmark/nam_circs")
    args = ap.parse_args()

    config, hw, coupling_map = load_topo(f'traindata/topo/{args.topo}.json')
    circuits = sorted(f for f in os.listdir(args.circ_dir) if f.endswith('.qasm'))

    sabre_phys = {}
    for f in circuits:
        qc = qasm2_loads(open(f'{args.circ_dir}/{f}').read())
        phys, info = sabre_route(qc, config, swap_trials=20, seed=0)
        sabre_phys[f] = (phys, info['num_swaps'])

    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    _done = {(r['method'], r['circuit']) for r in results}

    def run_one(method, fname, phys, swaps, truncated):
        key = (method, fname)
        if key in _done:
            return
        if truncated:
            results.append({'method': method, 'circuit': fname, 'swaps': swaps,
                            'T': 0, 'fid': 0.0, 'truncated': True})
            print(f'{fname} [{method}] TRUNC 跳过', flush=True)
        else:
            t0 = time.perf_counter()
            fid = _FID(phys, config,
                                                     num_trajectories=T_DEEP,
                                                     seed=42, backend='auto')
            dt = round(time.perf_counter() - t0, 1)
            results.append({'method': method, 'circuit': fname, 'swaps': swaps,
                            'T': T_DEEP, 'fid': fid, 'sec': dt})
            print(f'{fname} [{method}] sw={swaps} fid={fid:.4f} ({dt:.0f}s)',
                  flush=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)

    for fname in circuits:
        run_one('SABRE', fname, sabre_phys[fname][0], sabre_phys[fname][1], False)
        name = fname.removesuffix('.qasm')
        for d, label in [x.split(":") for x in args.model_dirs.split(",")]:
            p = f'benchmark/routed/{d}/{name}.json'
            if not os.path.exists(p):
                print(f'{fname} [{label}] 缺 {p}，跳过', flush=True)
                continue
            dd = json.load(open(p))
            phys = qasm2_loads(dd['routed_qasm'].replace(
                'include "qelib1.inc";',
                'include "qelib1.inc";\n' + SWAP_DEF))
            run_one(label, fname, phys, dd['num_swaps'],
                    not dd.get('completed', True))

    by = {}
    for r in results:
        by.setdefault(r['circuit'], {})[r['method']] = r['fid']
    print(f'\n=== T=128 NAM 全 {len(circuits)} 条：SABRE vs {args.tag} ===')
    for c in sorted(by):
        s = by[c].get('SABRE')
        if s is None:
            continue
        marks = []
        for label in [x.split(":")[1] for x in args.model_dirs.split(",")]:
            m = by[c].get(label)
            marks.append(f'{label}={m:.4f}' if m is not None else f'{label}=--')
        print(f'{c:<24s} SABRE={s:.4f}  ' + '  '.join(marks))
    sm = sum(by[c]['SABRE'] for c in by) / len(by)
    print(f'\nfid_mean SABRE={sm:.4f}')
    for label in [x.split(":")[1] for x in args.model_dirs.split(",")]:
        mm = [by[c][label] for c in by if label in by[c]]
        if mm:
            print(f'fid_mean {label}={sum(mm)/len(mm):.4f} '
                  f'({(sum(mm)/len(mm)-sm)/sm*100:+.1f}%)')


if __name__ == "__main__":
    main()
