"""结构化电路生成器 v2：S5 三轴扩展（交互模式 + 并行结构 + 规模谱）。

新增 8 族（对齐 doc/20260915训练方案.md §7.2）：
  交互模式族（长程/可控交互）:
    qft_butterfly   : QFT 蝶形（全对全长程，现有族最缺）
    qpe_block       : 受控相位梯（相位估计块）
    pauli_evolution : 随机图 Trotter 层，局部/长程项比例可控
    mcx_ladder      : 多控制 X 分解链
    clifford_layer  : 分层随机 CZ 图案 + 随机 1q
  并行结构族（宽 front-layer 高并发 → 串扰信号方差来源）:
    parallel_blocks : L 层 × 每层 k 个不相交比特组独立块
    layered_matching: 每层随机匹配 2q 门，宽度 w 可扫描
    mixed_serial_parallel : 串行链与并行块交替

全部转译到基础门（与 v1 BASIS / GATE_DURATION_TABLE / v2 模拟器兼容），
过滤 5 ≤ n_qubits ≤ 20、门数 ≥ 30、≤ --max-gates，输出 QASM + manifest
（含 front-layer 宽度/并发对/交互距离结构指标，服务 D1 诊断）。

用法: python3 scripts/gen_structured_circuits_v2.py [--out-dir DIR]
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import qasm2

BASIS = ["h", "rz", "x", "cx", "y", "z", "s", "t", "sdg", "tdg"]
SCALES = [5, 8, 10, 12, 14, 16, 20]


# ---------------------------------------------------------------------------
# 各族构造（逻辑电路，未转译）
# ---------------------------------------------------------------------------
def gen_qft_butterfly(n: int, inverse: bool, rng) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    for k in range(1, n):
        for i in range(n - k):
            qc.cp(np.pi / (2 ** k), i, i + k)
    if inverse:
        qc = qc.inverse()
    return qc


def gen_qpe_block(n: int, reps: int, rng) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    perm = list(range(n))
    rng.shuffle(perm)
    for _ in range(reps):
        for k in range(1, n):
            for i in range(n - k):
                a, b = perm[i], perm[i + k]
                qc.cp(np.pi / (2 ** k), a, b)
        rng.shuffle(perm)
    return qc


def gen_pauli_evolution(n: int, layers: int, lr_frac: float, rng) -> QuantumCircuit:
    """随机图哈密顿量 Trotter：每层一个匹配，长程项占比 lr_frac。"""
    qc = QuantumCircuit(n)
    for _ in range(layers):
        pairs = []
        nodes = list(range(n))
        rng.shuffle(nodes)
        used = set()
        for idx in range(0, n - 1, 2):
            a, b = nodes[idx], nodes[idx + 1]
            if rng.random() < lr_frac:  # 长程：随机远距对
                a, b = rng.randrange(n), rng.randrange(n)
                if a == b or (a, b) in used:
                    a, b = nodes[idx], nodes[idx + 1]
            used.add((min(a, b), max(a, b)))
            pairs.append((a, b))
        for (a, b) in pairs:
            theta = rng.uniform(0, np.pi)
            qc.cx(a, b)
            qc.rz(theta, b)
            qc.cx(a, b)
        for q in range(n):
            if rng.random() < 0.5:
                qc.rz(rng.uniform(0, np.pi), q)
    return qc


def gen_mcx_ladder(n: int, rng) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for c in range(1, n):
        controls = list(range(c))
        qc.mcx(controls, c)
    return qc


def gen_clifford_layer(n: int, layers: int, rng) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for _ in range(layers):
        nodes = list(range(n))
        rng.shuffle(nodes)
        for idx in range(0, n - 1, 2):
            qc.cz(nodes[idx], nodes[idx + 1])
        for q in range(n):
            qc.h(q) if rng.random() < 0.5 else qc.s(q)
    return qc


def gen_parallel_blocks(n: int, layers: int, k: int, rng) -> QuantumCircuit:
    """L 层 × 每层 k 个不相交比特组独立块（宽 front layer 高并发）。"""
    qc = QuantumCircuit(n)
    k = max(1, min(k, n // 2))
    for _ in range(layers):
        nodes = list(range(n))
        rng.shuffle(nodes)
        groups = [nodes[i::k] for i in range(k)]
        for g in groups:
            for idx in range(len(g) - 1):
                qc.cx(g[idx], g[idx + 1])
                qc.rz(rng.uniform(0, np.pi), g[idx + 1])
            if len(g) >= 2:
                qc.cx(g[-1], g[0])
    return qc


def gen_layered_matching(n: int, layers: int, w: int, rng) -> QuantumCircuit:
    """每层 w 个不相交 2q 门（宽度 w=1 纯串行 → n/2 满并发）。"""
    qc = QuantumCircuit(n)
    w = max(1, min(w, n // 2))
    for _ in range(layers):
        nodes = list(range(n))
        rng.shuffle(nodes)
        for idx in range(w):
            a, b = nodes[2 * idx], nodes[2 * idx + 1]
            qc.cx(a, b)
            qc.rz(rng.uniform(0, np.pi), b)
        for q in range(n):
            if rng.random() < 0.3:
                qc.h(q)
    return qc


def gen_mixed_serial_parallel(n: int, layers: int, rng) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for li in range(layers):
        if li % 2 == 0:  # 串行链
            order = list(range(n))
            rng.shuffle(order)
            for idx in range(n - 1):
                qc.cx(order[idx], order[idx + 1])
        else:  # 并行匹配
            nodes = list(range(n))
            rng.shuffle(nodes)
            for idx in range(0, n - 1, 2):
                qc.cx(nodes[idx], nodes[idx + 1])
                qc.rz(rng.uniform(0, np.pi), nodes[idx + 1])
    return qc


# ---------------------------------------------------------------------------
# 结构指标（ASAP 分层）
# ---------------------------------------------------------------------------
def structure_metrics(qc: QuantumCircuit) -> dict:
    gates = []
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        gates.append((inst.operation.name, qs))
    # ASAP 分层：每门层号 = 其所有 qubit 当前层高的 max + 1
    q_level = [0] * qc.num_qubits
    waves = {}
    cx_dists = []
    for name, qs in gates:
        lv = max(q_level[q] for q in qs) + 1
        for q in qs:
            q_level[q] = lv
        waves.setdefault(lv, []).append((name, qs))
        if name == "cx" and len(qs) == 2:
            cx_dists.append(abs(qs[0] - qs[1]))
    widths = [len(v) for _, v in sorted(waves.items())]
    two_q_per_wave = [sum(1 for nm, _ in v if nm in ("cx", "cz", "ecr"))
                      for _, v in sorted(waves.items())]
    conc_fracs = []
    for _, v in waves.items():
        g2 = [(nm, qs) for nm, qs in v if nm in ("cx", "cz", "ecr")]
        if len(g2) >= 2:
            tot = 0
            disjoint = 0
            for i in range(len(g2)):
                for j in range(i + 1, len(g2)):
                    tot += 1
                    if not set(g2[i][1]) & set(g2[j][1]):
                        disjoint += 1
            conc_fracs.append(disjoint / tot)
    return {
        "front_width_mean": round(float(np.mean(widths)), 2) if widths else 0.0,
        "front_width_max": int(max(widths)) if widths else 0,
        "two_q_wave_mean": round(float(np.mean(two_q_per_wave)), 2) if two_q_per_wave else 0.0,
        "concurrency_mean": (round(float(np.mean(conc_fracs)), 3)
                             if conc_fracs else 0.0),
        "cx_dist_mean": round(float(np.mean(cx_dists)), 2) if cx_dists else 0.0,
        "cx_dist_max": int(max(cx_dists)) if cx_dists else 0,
    }


# ---------------------------------------------------------------------------
# 批量计划
# ---------------------------------------------------------------------------
def build_all():
    out = []

    def add(name, qc):
        out.append((name, qc))

    for n in SCALES:
        rng = random.Random(1000 + n)
        # 交互模式族
        for inv in (False, True):
            add(f"qft_butterfly_{n}{'_inv' if inv else ''}_0",
                gen_qft_butterfly(n, inv, rng))
        add(f"qpe_block_{n}_0", gen_qpe_block(n, 2, rng))
        for lr in (0.0, 0.5):
            add(f"pauli_evolution_{n}_lr{int(lr*100)}_0",
                gen_pauli_evolution(n, max(2, 24 // max(2, n // 4)), lr, rng))
        if n <= 12:
            add(f"mcx_ladder_{n}_0", gen_mcx_ladder(n, rng))
        add(f"clifford_layer_{n}_0", gen_clifford_layer(n, max(2, 40 // n), rng))
        # 并行结构族
        add(f"parallel_blocks_{n}_0",
            gen_parallel_blocks(n, max(2, 30 // max(2, n // 4)), max(2, n // 4), rng))
        for w_tag, w in (("w1", 1), ("wq", max(1, n // 4)), ("wh", n // 2)):
            add(f"layered_matching_{n}_{w_tag}_0",
                gen_layered_matching(n, max(3, 60 // max(1, w * 2)), w, rng))
        add(f"mixed_serial_parallel_{n}_0",
            gen_mixed_serial_parallel(n, max(3, 40 // max(2, n // 4)), rng))
        # 种子变体（×5）
        for vi in (1, 2, 3, 4, 5):
            rng2 = random.Random(2000 + n * 10 + vi)
            add(f"parallel_blocks_{n}_{vi}",
                gen_parallel_blocks(n, max(2, 30 // max(2, n // 4)),
                                    max(2, n // 4), rng2))
            add(f"layered_matching_{n}_wq_{vi}",
                gen_layered_matching(n, max(3, 60 // max(1, n // 2)),
                                     max(1, n // 4), rng2))
            add(f"layered_matching_{n}_w1_{vi}",
                gen_layered_matching(n, max(3, 60 // 2), 1, rng2))
            add(f"mixed_serial_parallel_{n}_{vi}",
                gen_mixed_serial_parallel(n, max(3, 40 // max(2, n // 4)), rng2))
            add(f"pauli_evolution_{n}_lr50_{vi}",
                gen_pauli_evolution(n, max(2, 24 // max(2, n // 4)), 0.5, rng2))
            add(f"clifford_layer_{n}_{vi}",
                gen_clifford_layer(n, max(2, 40 // n), rng2))
            if vi <= 2:
                add(f"qpe_block_{n}_{vi}", gen_qpe_block(n, 3, rng2))
                add(f"pauli_evolution_{n}_lr25_{vi}",
                    gen_pauli_evolution(n, max(2, 24 // max(2, n // 4)), 0.25, rng2))
                add(f"layered_matching_{n}_wh_{vi}",
                    gen_layered_matching(n, max(3, 60 // max(1, n)),
                                         max(1, n // 2), rng2))
                if n <= 12:
                    add(f"mcx_ladder_{n}_{vi}", gen_mcx_ladder(n, rng2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="traindata/gen_structured_v2")
    ap.add_argument("--max-gates", type=int, default=1500)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    kept = skipped_small = skipped_gates = 0
    for name, qc in build_all():
        try:
            tqc = transpile(qc, basis_gates=BASIS, optimization_level=0,
                            seed_transpiler=0)
        except Exception as e:
            print(f"[skip] {name}: transpile 失败 {e}", flush=True)
            continue
        if not (5 <= tqc.num_qubits <= 20):
            skipped_small += 1
            continue
        n_nl = sum(1 for inst in tqc.data if len(inst.qubits) == 2)
        if tqc.size() < 30 or n_nl == 0:
            skipped_gates += 1
            continue
        if tqc.size() > args.max_gates:
            skipped_gates += 1
            continue
        fname = f"{name}.qasm"
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write(qasm2.dumps(tqc))
        m = structure_metrics(tqc)
        manifest.append({
            "name": fname,
            "family": name.rsplit("_", 1)[0].rsplit("_", 1)[0]
            if name.count("_") >= 1 else name,
            "qubits": tqc.num_qubits, "gates": tqc.size(),
            "cx": n_nl, "depth": tqc.depth(), **m,
        })
        kept += 1
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    by_q = {}
    for m in manifest:
        by_q.setdefault(m["qubits"], 0)
        by_q[m["qubits"]] += 1
    print(f"kept={kept} skipped_small={skipped_small} skipped_gates={skipped_gates}")
    print("by qubits:", dict(sorted(by_q.items())))
    fams = {}
    for m in manifest:
        fam = m["name"].split("_")[0] + "_" + m["name"].split("_")[1] \
            if not m["name"].split("_")[1].startswith(("w", "lr", "inv")) \
            else m["name"].split("_")[0]
        fams.setdefault(m["name"].split("_")[0], 0)
        fams[m["name"].split("_")[0]] += 1
    print("by family:", dict(sorted(fams.items())))


if __name__ == "__main__":
    main()
