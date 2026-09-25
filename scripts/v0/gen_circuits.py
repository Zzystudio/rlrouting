"""v0 电路生成器：小规模（4-8q）路由测试电路。

两类：
  - uniform random：随机 2Q 门（cx）序列，qiskit 电路；结构化约束由 DAG 依赖隐含
  - structured 族：链式/阶梯式/置换混合等局部结构化模式（对应 clocked 框架已知的
    结构化电路弱点），生成小门数版本供精确求解。

输出: traindata/v0/<family>_<n>q_<seed>.pkl （qiskit QuantumCircuit pickle，与
现有 traindata 格式一致）；同时打印每实例 (nq, n_2q) 清单便于选择精确可行域。
"""

import argparse
import os
import pickle
import random

from qiskit import QuantumCircuit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT_DIR = os.path.join(ROOT, "traindata", "v0")


def gen_random(nq: int, n_2q: int, rng: random.Random) -> QuantumCircuit:
    qc = QuantumCircuit(nq)
    pairs = [(i, j) for i in range(nq) for j in range(i + 1, nq)]
    for _ in range(n_2q):
        q0, q1 = rng.choice(pairs)
        qc.cx(q0, q1)
    return qc


def gen_chain(nq: int, n_2q: int, rng: random.Random) -> QuantumCircuit:
    """串行链：门按 (i, i+1) 轮转，深度大、frontier 窄（最易路由）。"""
    qc = QuantumCircuit(nq)
    for k in range(n_2q):
        i = rng.randrange(nq - 1)
        qc.cx(i, i + 1)
    return qc


def gen_staircase(nq: int, n_2q: int, rng: random.Random) -> QuantumCircuit:
    """阶梯：固定跨度 2 的对 (i, i+2)，需 SWAP 搬运距离 2 的逻辑对。"""
    qc = QuantumCircuit(nq)
    for k in range(n_2q):
        i = rng.randrange(max(1, nq - 2))
        qc.cx(i, min(nq - 1, i + 2))
    return qc


def gen_perm_mix(nq: int, n_2q: int, rng: random.Random) -> QuantumCircuit:
    """置换混合：若干轮"两两交换远距对"，模拟通信密集结构。"""
    qc = QuantumCircuit(nq)
    count = 0
    while count < n_2q:
        perm = list(range(nq))
        rng.shuffle(perm)
        for k in range(0, nq - 1, 2):
            if count >= n_2q:
                break
            qc.cx(perm[k], perm[k + 1])
            count += 1
    return qc


FAMILIES = {
    "random": gen_random,
    "chain": gen_chain,
    "staircase": gen_staircase,
    "perm_mix": gen_perm_mix,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--n-2q", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--family", default="all", choices=list(FAMILIES) + ["all"])
    ap.add_argument("--seeds", type=int, default=8, help="每个 (族,nq,n2q) 的 seed 数")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    families = list(FAMILIES) if args.family == "all" else [args.family]
    manifest_path = os.path.join(args.out, "manifest.json")
    seen = set()
    if os.path.exists(manifest_path):
        for m in json.load(open(manifest_path)):
            seen.add(m["file"])
    manifest = []
    total = 0
    for fam in families:
        for nq in args.nq:
            for n2q in args.n_2q:
                for s in range(args.seeds):
                    rng = random.Random(1000 * nq + 100 * n2q + s)
                    qc = FAMILIES[fam](nq, n2q, rng)
                    fn = f"{fam}_{nq}q_{n2q}g_s{s}.pkl"
                    if fn in seen:
                        continue
                    with open(os.path.join(args.out, fn), "wb") as f:
                        pickle.dump(qc, f)
                    manifest.append({"family": fam, "nq": nq, "n2q": n2q,
                                     "seed": s, "file": fn})
                    total += 1
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path)) + manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"generated {total} new circuits -> {args.out} (manifest 累计 {len(manifest)})")


if __name__ == "__main__":
    import json
    main()
