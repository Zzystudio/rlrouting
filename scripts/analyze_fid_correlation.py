"""analytic vs trajectory_sched 保真度排序相关性分析（同电路多路由候选）。

对每个 NAM 电路，收集 8 种路由（l05/l05_beam3/l05_beam5/ph2v4/ph2v4_beam3/
ph2v4_beam5/nam_l05_v2/nam_l05_v2_beam3）的 routed_qasm：
  - trajectory_fid：从 per-circuit JSON 直接读取（trajectory_sched×16, seed=0）
  - analytic_fid：用 make_analytic_fidelity_fn 现算
然后算 per-circuit Spearman 秩相关，以及合并相关/散点。
"""
import json, os, sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim import make_analytic_fidelity_fn
from routing.rl.eval_policy import load_topo

ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIRS = ['l05', 'l05_beam3', 'l05_beam5', 'ph2v4', 'ph2v4_beam3',
              'ph2v4_beam5', 'nam_l05_v2', 'nam_l05_v2_beam3']
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'


class _FakeEnv:
    """make_analytic_fidelity_fn 期望 env._phys_circuit。"""
    def __init__(self, qc):
        self._phys_circuit = qc


def load_per_circuit(thermal, crosstalk):
    config, hw, _ = load_topo(os.path.join(ROOT, 'traindata/topo/tianyan176_20q.json'))
    ana_fn = make_analytic_fidelity_fn(config, include_thermal=thermal,
                                       include_crosstalk=crosstalk)
    per_circuit = defaultdict(list)
    for d in MODEL_DIRS:
        dpath = os.path.join(ROOT, 'benchmark/routed', d)
        for f in sorted(os.listdir(dpath)):
            if not f.endswith('.json'):
                continue
            rec = json.load(open(os.path.join(dpath, f)))
            traj_fid = rec.get('fidelity')
            if traj_fid is None:
                continue
            qasm_with_swap = rec['routed_qasm'].replace(
                'include "qelib1.inc";',
                'include "qelib1.inc";\n' + SWAP_DEF)
            qc = qasm2_loads(qasm_with_swap)
            try:
                ana_fid = ana_fn(_FakeEnv(qc))
            except Exception:
                ana_fid = None
            circuit = f.replace('.json', '')
            per_circuit[circuit].append({
                'model': d, 'traj': traj_fid, 'ana': ana_fid,
                'q': rec['num_logical_qubits'], 'swaps': rec['num_swaps'],
            })
    return per_circuit


def dedup(cands):
    seen = {}
    for c in cands:
        if c['ana'] is None or c['traj'] is None:
            continue
        key = round(c['ana'], 8)
        seen[key] = c
    return list(seen.values())


def analyze(per_circuit):
    rows = []
    for circuit in sorted(per_circuit, key=lambda c: per_circuit[c][0]['q']):
        cands = dedup(per_circuit[circuit])
        if len(cands) < 3:
            continue
        traj = np.array([c['traj'] for c in cands])
        ana = np.array([c['ana'] for c in cands])
        sp, p = spearmanr(traj, ana)
        rows.append({'circuit': circuit, 'q': cands[0]['q'], 'n': len(cands),
                     'sp': sp, 'p': p, 'ana_span': ana.max() - ana.min()})
    return rows


def main():
    for tag, thermal, crosstalk in [
            ('thermal on, xtalk off (v2 training)', True, False),
            ('thermal off, xtalk off', False, False),
            ('thermal on, xtalk on', True, True),
    ]:
        per_circuit = load_per_circuit(thermal, crosstalk)
        rows = analyze(per_circuit)
        sps = [r['sp'] for r in rows]
        print(f"\n===== analytic[{tag}] =====")
        print(f"per-circuit Spearman: mean={np.mean(sps):.3f} median={np.median(sps):.3f} "
              f"pos={(np.array(sps) > 0).sum()}/{len(sps)} (n={len(sps)} circuits)")
        for r in rows:
            print(f"  {r['circuit']:<15} q={r['q']:>2} n={r['n']} "
                  f"sp={r['sp']:+.3f} ana_span={r['ana_span']:.5f}")


if __name__ == '__main__':
    main()
