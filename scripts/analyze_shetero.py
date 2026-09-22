"""stronghetero 拓扑泛化分析（in-dist 50 条）。"""
import json, math, statistics, os, sys, traceback

os.chdir(os.path.expanduser('~/opencode-server/opencode-docker/projects/rlrouting'))
sys.path.insert(0, 'src')
from qiskit.qasm2 import loads as qasm2_loads
from sim.trajectory_sim_v2 import trajectory_circuit_fidelity_events
from routing.rl.eval_policy import load_topo

config, hw, cm = load_topo('traindata/topo/tianyan287_20q_stronghetero.json')
mf_raw = json.load(open('benchmark/indist_val/manifest.json'))
mf = {m['name'].removesuffix('.qasm'): m for m in mf_raw}
SWAP_DEF = 'gate swap a,b { cx a,b; cx b,a; cx a,b; }\n'
T_MAP = lambda q: 16 if q <= 9 else (32 if q <= 14 else 64)

dirs = {
    'LA287d':      'benchmark/routed/la287d_sh_argmax',
    'LA287d_b5la': 'benchmark/routed/la287d_beam5la_bud_sh',
    'LA287ctl':    'benchmark/routed/la287ctl_argmax_sh',
    'LA287c':      'benchmark/routed/la287c_argmax_sh',
}

sabre_data = {}
sab_f = 'benchmark/routed/hybridT_sabre_shetero_indist.json'
if os.path.exists(sab_f):
    for r in json.load(open(sab_f)):
        sabre_data[r['circuit'].removesuffix('.qasm')] = r['fid']
print(f'SABRE 数据: {len(sabre_data)} 条', flush=True)

all_data = {}
errors = []
for name in sorted(mf):
    q = mf[name]['qubits']
    T = T_MAP(q)
    row = {}
    if name in sabre_data:
        row['SABRE'] = sabre_data[name]
    for tag, dd in [('LA287d', dirs['LA287d']), ('LA287d_b5la', dirs['LA287d_b5la']),
                    ('LA287ctl', dirs['LA287ctl']), ('LA287c', dirs['LA287c'])]:
        jf = os.path.join(dd, name + '.json')
        if not os.path.exists(jf):
            continue
        try:
            d = json.load(open(jf))
            if not d.get('completed', True):
                row[tag] = None; continue
            phys = qasm2_loads(d['routed_qasm'].replace(
                'include "qelib1.inc";', 'include "qelib1.inc";\n' + SWAP_DEF))
            fid = trajectory_circuit_fidelity_events(phys, config,
                                                     num_trajectories=T, seed=42,
                                                     backend='auto')
            row[tag] = fid
        except Exception as e:
            print(f'  [ERR] {name}/{tag}: {e}', flush=True)
            row[tag] = None
    all_data[name] = row
    print(f'{name}: done', flush=True)

# 汇总
print('\n=== stronghetero 拓扑泛化汇总 ===')
methods = ['SABRE', 'LA287d', 'LA287d_b5la', 'LA287ctl', 'LA287c']
print(f"{'method':<14s} {'fid_mean':>9s} {'logmean':>9s} {'swaps':>7s} {'胜SABRE':>8s} {'n':>3s}")
for m in methods:
    fids, sws, wins, total = [], [], 0, 0
    for name in all_data:
        v = all_data[name].get(m)
        if v is None: continue
        fids.append(v)
        if m == 'SABRE':
            sws.append(sabre_data.get(name, {}).get('num_swaps', 0))
        else:
            jf = os.path.join(dirs[m], name + '.json') if m != 'SABRE' else None
            if jf and os.path.exists(jf):
                sws.append(json.load(open(jf))['num_swaps'])
        if m != 'SABRE' and 'SABRE' in all_data[name]:
            sv = all_data[name]['SABRE']
            if sv is not None:
                total += 1
                if v > sv + 1e-9: wins += 1
    if not fids: continue
    lm = math.exp(statistics.mean(math.log(max(x, 1e-9)) for x in fids))
    sw_mean = sum(sws)/len(sws) if sws else 0
    wt = f'{wins}/{total}' if m != 'SABRE' else '-'
    print(f'{m:<14s} {sum(fids)/len(fids):>9.4f} {lm:>9.4f} {sw_mean:>7.1f} {wt:>8s} {len(fids):>3d}')

print('\nDONE')
