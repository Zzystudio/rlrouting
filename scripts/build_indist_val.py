"""构建 in-distribution held-out 验证集（~50 条）。

来源：/tmp/opencode/gen_ho_v1（variant_offset=10）与 gen_ho_v2（seed_offset=5000）
的新种子电路；内容哈希去重 vs 训练集（gen_structured/gen_structured_v2）与 NAM；
分层抽取：v1 算术族（NAM 结构同源，诊断主力）~26 + v2 宽并行族 ~24。
输出: benchmark/indist_val/ + manifest.json
"""
import hashlib
import json
import os
import shutil
import sys

OUT = 'benchmark/indist_val'
TARGET = 50
SRC = [('/tmp/opencode/gen_ho_v1', 'v1'),
       ('/tmp/opencode/gen_ho_v2', 'v2')]
TRAIN_DIRS = ['traindata/gen_structured', 'traindata/gen_structured_v2',
              'benchmark/nam_circs']


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# 训练/NAM 内容指纹
train_hashes = set()
for d in TRAIN_DIRS:
    for f in os.listdir(d):
        if f.endswith('.qasm'):
            train_hashes.add(sha(os.path.join(d, f)))
print(f'训练/NAM 指纹: {len(train_hashes)} 条')

# 候选（去重）
cands = []
seen = set()
for d, tag in SRC:
    mf = json.load(open(os.path.join(d, 'manifest.json')))
    for e in mf:
        p = os.path.join(d, e['name'])
        h = sha(p)
        if h in train_hashes or h in seen:
            continue
        seen.add(h)
        cands.append({**e, 'src': tag, 'path': p, 'hash': h})
print(f'候选（去重后）: {len(cands)}')

# 分层抽取
picked = []
by_fam = {}
for c in cands:
    by_fam.setdefault(c['family'], []).append(c)


def take(fam_prefix, n, pool):
    got = []
    for c in pool:
        if len(got) >= n:
            break
        if c['family'].startswith(fam_prefix) and c not in picked:
            got.append(c)
    return got


# v1 算术族（NAM 结构同源，诊断主力）
for fam, n in [('tof_tower', 12), ('gf2_mult', 6), ('perm', 4)]:
    picked.extend(take(fam, n, sorted(by_fam.get(fam, []),
                                       key=lambda x: x['qubits'])))
# v1 其他算术族的新鲜电路（若有）
for fam, n in [('cdkm', 2), ('vbe', 2), ('draper', 2), ('adder_stack', 2),
               ('rgqft', 1), ('wadd', 1), ('grover', 1)]:
    picked.extend(take(fam, n, sorted(by_fam.get(fam, []),
                                       key=lambda x: x['qubits'])))
# v2 宽并行族：按族轮转抽取（每 family×size 至多 1），凑满 50
v2_pool = sorted([c for c in cands if c['src'] == 'v2'],
                 key=lambda x: (x['family'], x['qubits']))
v2_byf = {}
for c in v2_pool:
    v2_byf.setdefault(c['family'], []).append(c)
fams_order = sorted(v2_byf)
round_i = 0
while len(picked) < TARGET:
    added = False
    for f in fams_order:
        if len(picked) >= TARGET:
            break
        if round_i < len(v2_byf[f]):
            c = v2_byf[f][round_i]
            if c not in picked:
                picked.append(c)
                added = True
    if not added:
        break
    round_i += 1

# 截断到 TARGET（保持族均衡：按 family 轮转裁剪）
if len(picked) > TARGET:
    byf = {}
    for c in picked:
        byf.setdefault(c['family'], []).append(c)
    picked = []
    while len(picked) < TARGET:
        for f in sorted(byf):
            if byf[f] and len(picked) < TARGET:
                picked.append(byf[f].pop(0))

os.makedirs(OUT, exist_ok=True)
manifest = []
for c in picked:
    shutil.copy(c['path'], os.path.join(OUT, c['name']))
    manifest.append({k: v for k, v in c.items() if k not in ('path', 'hash')})

with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)

fams = {}
for c in manifest:
    fams[c['family']] = fams.get(c['family'], 0) + 1
qs = sorted(m['qubits'] for m in manifest)
print(f'held-out 集完成: {len(manifest)} 条 -> {OUT}')
print(f'族分布: {dict(sorted(fams.items()))}')
print(f'规模分布: {qs}')
