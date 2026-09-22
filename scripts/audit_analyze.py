"""审计分析：合并重放奖励轴 + T=128 保真度轴 → 象限判定 + 分量归因 + 单价对照。"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

REP = 'benchmark/routed/audit_replay_indist.json'
FID = 'benchmark/routed/audit_T128_indist.json'

rep = json.load(open(REP))['circuits']
fid_rows = json.load(open(FID))
fid = {}
for r in fid_rows:
    fid.setdefault(r['method'], {})[r['circuit'].removesuffix('.qasm')] = r

methods = ['SABRE', 'argmax', 'beam5la+bud']


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    return float((dx * dy).sum() / denom) if denom > 0 else float('nan')


print('=== 主判别：ΔR(奖励) vs ΔF(保真度, T=128) ===\n')
for pair in [('argmax', 'SABRE'), ('beam5la+bud', 'SABRE'), ('beam5la+bud', 'argmax')]:
    m1, m2 = pair
    rows = []
    for c in rep:
        if m1 not in rep[c] or m2 not in rep[c]:
            continue
        f1 = fid.get(m1, {}).get(c)
        f2 = fid.get(m2, {}).get(c)
        if not f1 or not f2:
            continue
        dR = rep[c][m1]['r_total'] - rep[c][m2]['r_total']
        dF = f1['fid'] - f2['fid']
        rows.append((c, dR, dF, rep[c][m1]['swaps'] - rep[c][m2]['swaps']))
    qA = sum(1 for _, dR, dF, _ in rows if dR > 0 and dF < 0)
    qB = sum(1 for _, dR, dF, _ in rows if dR < 0 and dF > 0)
    qC = sum(1 for _, dR, dF, _ in rows if dR < 0 and dF < 0)
    qD = sum(1 for _, dR, dF, _ in rows if dR > 0 and dF > 0)
    rho = spearman([r[1] for r in rows], [r[2] for r in rows])
    print(f'--- {m1} vs {m2} (n={len(rows)}) ---')
    print(f'  H1 错配象限 (ΔR>0, ΔF<0): {qA}')
    print(f'  反向错配   (ΔR<0, ΔF>0): {qB}')
    print(f'  同向劣势   (ΔR<0, ΔF<0): {qC}')
    print(f'  一致优势   (ΔR>0, ΔF>0): {qD}')
    print(f'  Spearman ρ(ΔR, ΔF) = {rho:+.3f}')
    print()

print('=== 奖励分量分解（均值/电路）===')
comps = ['r_gate', 'r_swap_price', 'r_xt_swap', 'r_sched_txi', 'r_prop',
         'r_budget', 'r_shape', 'r_terminal']
print(f"{'分量':<14s} {'SABRE':>9s} {'argmax':>9s} {'b5la+bud':>9s}")
for comp in comps:
    line = f'{comp:<14s}'
    for m in methods:
        vals = [rep[c][m].get(comp, 0.0) for c in rep if m in rep[c]]
        line += f' {sum(vals)/len(vals):>9.3f}'
    print(line)
for m in methods:
    vals = [rep[c][m]['r_total'] for c in rep if m in rep[c]]
    print(f'R_total[{m:<10s}] = {sum(vals)/len(vals):+.3f}')

print('\n=== 单价对照：奖励计价 vs 模拟器实测（ΔF 回归）===')
# ΔF ~ Δswaps, Δedge_err(用 r_gate 反推不可行，改用模拟器侧物理量), Δmakespan
# 这里做：ΔF 对 (Δswaps, ΔR_gate, ΔR_swap+ΔR_budget, ΔR_sched_txi) 的回归系数
import numpy as np
pair = ('argmax', 'SABRE')
X, Y = [], []
for c in rep:
    if 'argmax' not in rep[c] or 'SABRE' not in rep[c]:
        continue
    f1 = fid.get('argmax', {}).get(c)
    f2 = fid.get('SABRE', {}).get(c)
    if not f1 or not f2:
        continue
    dsw = rep[c]['argmax']['swaps'] - rep[c]['SABRE']['swaps']
    drgate = rep[c]['argmax']['r_gate'] - rep[c]['SABRE']['r_gate']
    drsw = (rep[c]['argmax']['r_swap_price'] + rep[c]['argmax']['r_budget']
            - rep[c]['SABRE']['r_swap_price'] - rep[c]['SABRE']['r_budget'])
    drsch = rep[c]['argmax']['r_sched_txi'] - rep[c]['SABRE']['r_sched_txi']
    X.append([1, drgate, drsw, drsch])
    Y.append(f1['fid'] - f2['fid'])
Xa, Ya = np.array(X, dtype=float), np.array(Y, dtype=float)
coef, *_ = np.linalg.lstsq(Xa, Ya, rcond=None)
pred = Xa @ coef
r2 = 1 - ((Ya - pred) ** 2).sum() / ((Ya - Ya.mean()) ** 2).sum()
names = ['截距', 'Δr_gate(边质量节省)', 'Δr_swap+budget(SWAP代价)', 'Δr_sched_txi(时间空闲罚)']
print(f'  n={len(Ya)}  R²={r2:.3f}')
for nm, cf in zip(names, coef):
    print(f'  {nm:<30s} 系数 = {cf:+.4f}')
print('  解读：系数 = 该奖励分量 +1 单位对应的真实保真度变化；')
print('  若为负/近零 → 奖励在该项上与保真度脱钩（错配点）。')

print('\n=== 单价对照 2：物理量回归（ΔF ~ Δ物理量）===')
X2, Y2 = [], []
for c in rep:
    if 'argmax' not in rep[c] or 'SABRE' not in rep[c]:
        continue
    f1 = fid.get('argmax', {}).get(c)
    f2 = fid.get('SABRE', {}).get(c)
    if not f1 or not f2:
        continue
    dsw = rep[c]['argmax']['swaps'] - rep[c]['SABRE']['swaps']
    dmk = (rep[c]['argmax']['makespan'] - rep[c]['SABRE']['makespan'])
    did = (rep[c]['argmax']['idle'] - rep[c]['SABRE']['idle']) / 20.0
    X2.append([1, dsw, dmk, did])
    Y2.append(f1['fid'] - f2['fid'])
X2a, Y2a = np.array(X2, dtype=float), np.array(Y2, dtype=float)
coef2, *_ = np.linalg.lstsq(X2a, Y2a, rcond=None)
pred2 = X2a @ coef2
r22 = 1 - ((Y2a - pred2) ** 2).sum() / ((Y2a - Y2a.mean()) ** 2).sum()
print(f'  n={len(Y2a)}  R²={r22:.3f}')
for nm, cf in zip(['截距', 'ΔSWAP数', 'Δmakespan(µs)', 'Δidle(µs/qubit)'], coef2):
    print(f'  {nm:<18s} 系数 = {cf:+.5f}')
print('  → ΔSWAP 系数即"每颗额外 SWAP 的真实保真度代价"（与奖励侧隐含价格对照）')
