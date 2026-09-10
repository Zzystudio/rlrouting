"""对比 NAM benchmark：新模型 vs l05 vs SABRE。

用法: python3 scripts/compare_nam.py <new_model_name> [base_model_name=l05]
输出统计表与逐电路图，打印到 stdout。
"""
import json, os, sys, statistics, math

new = sys.argv[1] if len(sys.argv) > 1 else 'l05_nam'
base = sys.argv[2] if len(sys.argv) > 2 else 'l05'

def load_summary(name):
    p = f'benchmark/routed/{name}_summary.json'
    d = json.load(open(p))
    return {r['circuit'].removesuffix('.qasm'): r for r in d['results']}

new_d = load_summary(new)
base_d = load_summary(base)
sabre = json.load(open('benchmark/routed/sabre_summary.json'))
sabre_d = {r['circuit'].removesuffix('.qasm'): r for r in sabre}

def logmean(vals):
    return math.exp(statistics.mean(math.log(max(v, 1e-9)) for v in vals))

print(f'\n=== NAM benchmark 对比: {new} vs {base} vs SABRE ===\n')
print(f'{"circuit":<16} {"q":>2} {"base_sw":>7} {"new_sw":>7} {"sab_sw":>7} '
      f'{"base_fid":>9} {"new_fid":>9} {"sab_fid":>9} {"winner":>8}')
rows = []
for c in sorted(new_d):
    n, b, s = new_d[c], base_d.get(c, {}), sabre_d.get(c, {})
    bf, nf, sf = b.get('fidelity'), n.get('fidelity'), s.get('fidelity')
    bw, nw, sw = b.get('num_swaps'), n.get('num_swaps'), s.get('swaps')
    # 胜者按保真度（数值越大越好），缺失按 -1
    cand = [('base', bf), ('new', nf), ('sab', sf)]
    winner = max(cand, key=lambda t: (t[1] if t[1] is not None else -1))[0]
    print(f'{c:<16} {n["num_logical_qubits"]:>2} '
          f'{bw if bw is not None else "--":>7} {nw if nw is not None else "--":>7} '
          f'{sw if sw is not None else "--":>7} '
          f'{bf if bf is not None else "--":>9} {nf if nf is not None else "--":>9} '
          f'{sf if sf is not None else "--":>9} {winner:>8}')
    rows.append((c, n, b, s))

new_sw = [n.get('num_swaps') for c, n, b, s in rows if n.get('num_swaps') is not None]
base_sw = [b.get('num_swaps') for c, n, b, s in rows if b.get('num_swaps') is not None]
sab_sw = [s.get('swaps') for c, n, b, s in rows if s.get('swaps') is not None]
new_f = [n.get('fidelity') for c, n, b, s in rows if n.get('fidelity') is not None]
base_f = [b.get('fidelity') for c, n, b, s in rows if b.get('fidelity') is not None]
sab_f = [s.get('fidelity') for c, n, b, s in rows if s.get('fidelity') is not None]

print(f'\n=== 汇总 ===')
print(f'{"指标":<24} {base:>12} {new:>12} {"SABRE":>12}')
print(f'{"SWAPs mean":<24} {statistics.mean(base_sw):>12.1f} {statistics.mean(new_sw):>12.1f} {statistics.mean(sab_sw):>12.1f}')
print(f'{"Fidelity mean":<24} {statistics.mean(base_f):>12.4f} {statistics.mean(new_f):>12.4f} {statistics.mean(sab_f):>12.4f}')
print(f'{"Fidelity log-mean":<24} {logmean(base_f):>12.4f} {logmean(new_f):>12.4f} {logmean(sab_f):>12.4f}')
print(f'{"vs SABRE (mean)":<24} {statistics.mean(base_f)/statistics.mean(sab_f)-1:>11.1%} {statistics.mean(new_f)/statistics.mean(sab_f)-1:>11.1%}')

# 逐电路胜率（保真度）
new_beats_base = sum(1 for c, n, b, s in rows
                     if n.get('fidelity') is not None and b.get('fidelity') is not None
                     and n['fidelity'] > b['fidelity'])
new_beats_sab = sum(1 for c, n, b, s in rows
                    if n.get('fidelity') is not None and s.get('fidelity') is not None
                    and n['fidelity'] > s['fidelity'])
print(f'\n{new} 胜 {base}: {new_beats_base}/{len(rows)}')
print(f'{new} 胜 SABRE: {new_beats_sab}/{len(rows)}')