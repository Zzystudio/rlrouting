"""结构化算术电路批量生成器：扩充 NAM 类训练分布。

生成 6 大族 5-20q 电路（对齐 NAM benchmark 的电路类型构成）：
  tof_tower  : Toffoli 塔链（模仿 tof_N / barenco_tof_N）
  cdkm_adder : CDKMRippleCarryAdder（模仿 rc_adder_N）
  vbe_adder  : VBERippleCarryAdder（模仿 vbe_adder_N）
  draper_add : DraperQFTAdder（QFT 加法器）
  rgqft_mult : RGQFTMultiplier（QFT 乘法器）
  gf2_mult   : GF(2^n) 移位乘法（模仿 gf2^k_mult：CCX 梯 + XOR 约化）
  grover     : MCZ oracle + diffusion 迭代（模仿 grover_5）
  wadd       : WeightedAdder
  perm       : PermutationGate 随机置换网络
  adder_stack: 加法器级联（模仿 mod_mult_55 的深度）

全部转译到 1-2qubit 基础门（h/rz/x/sx/cx/y/z/s/t/sdg/tdg/swap，与
GATE_DURATION_TABLE / v2 模拟器 / CircuitDAG 兼容），过滤 5 ≤ n_qubits ≤ 20
且门数 ≥ 30 的电路，输出 .qasm 到 --out-dir 并写 manifest JSON。

用法: python3 scripts/gen_structured_circuits.py [--out-dir DIR] [--per-variant N]
"""
import argparse
import itertools
import json
import os
import sys

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    VBERippleCarryAdder,
    RGQFTMultiplier,
    WeightedAdder,
    GroverOperator,
    PermutationGate,
)

BASIS = ["h", "rz", "x", "cx", "y", "z", "s", "t", "sdg", "tdg"]


# ---------------------------------------------------------------------------
# 各族电路构造（逻辑电路，未转译）
# ---------------------------------------------------------------------------
def gen_tof_tower(n_q: int, depth: int, rng) -> QuantumCircuit:
    """CCX 链塔：随机控制/目标选取，层间共享比特（模仿 tof_N 连乘）。"""
    qc = QuantumCircuit(n_q)
    for k in range(depth):
        cands = list(itertools.combinations(range(n_q), 3))
        c1, c2, t = cands[rng.randrange(len(cands))]
        qc.ccx(c1, c2, t)
    return qc


def gen_gf2_mult(n: int, rng) -> QuantumCircuit:
    """GF(2^n) 移位乘法：a(n)+b(n)+r(2n) 比特，对每个 (i,j) 用 CCX 累积
    部分积，再做 XOR 约化梯（模仿 gf2^k_mult 的 CNOT/CCX 结构）。"""
    qc = QuantumCircuit(4 * n)
    a, b, r = list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 4 * n))
    for i in range(n):
        for j in range(n):
            qc.ccx(a[i], b[j], r[i + j])
    # GF(2) 约化：高位乘积项 XOR 回低位（模 x^n+x+1 的近似梯）
    for i in range(n, 2 * n):
        tgt = r[(i * 3) % n]
        for k in range(1, rng.randint(2, 4)):
            src = r[(i + k) % (2 * n)]
            if src is tgt:
                continue  # 控制与目标不能是同一比特
            qc.cx(src, tgt)
    return qc


def gen_grover(m: int, iters: int, rng) -> QuantumCircuit:
    """Grover：MCZ oracle + diffusion 迭代（模仿 grover_5）。

    qiskit 1.4.6 的 GroverOperator 不含 diffusion，手动补
    diffusion = H⊗m · MCZ · H⊗m。
    """
    qc = QuantumCircuit(m)
    qc.h(range(m))
    oracle = QuantumCircuit(m)
    oracle.h(m - 1)
    oracle.mcx(list(range(m - 1)), m - 1)
    oracle.h(m - 1)
    gop = GroverOperator(oracle)
    diff = QuantumCircuit(m)
    diff.h(range(m))
    diff.h(m - 1)
    diff.mcx(list(range(m - 1)), m - 1)
    diff.h(m - 1)
    diff.h(range(m))
    for _ in range(iters):
        qc.compose(gop, inplace=True)
        qc.compose(diff, inplace=True)
    return qc


def gen_adder_stack(n: int, count: int, cls, rng) -> QuantumCircuit:
    """加法器级联：count 个 n 位加法器串在移位寄存器上（模仿 mod_mult 深度）。"""
    total = cls(num_state_qubits=n).num_qubits + (count - 1) * n
    total = min(total, 20)
    qc = QuantumCircuit(total)
    offset = 0
    for k in range(count):
        if offset + cls(num_state_qubits=n).num_qubits > total:
            break
        qc.compose(cls(num_state_qubits=n), qubits=range(offset, offset + cls(num_state_qubits=n).num_qubits),
                   inplace=True)
        offset += max(1, n // 2)
    return qc


def build_all(per_variant: int, variant_offset: int = 0):
    """返回 [(name, logical_qc), ...]，name 已含族名/规模/种子。

    variant_offset：变体索引整体平移（held-out 用非零 offset 产出训练未见过的
    新种子电路；默认 0 = 训练集行为）。"""
    import random

    out = []

    def add(name, qc):
        out.append((name, qc))

    # --- Toffoli 塔链 ---
    for n_q in range(5, 20):
        for vi in range(variant_offset, variant_offset + max(2, per_variant // 2)):
            rng = random.Random(hash((n_q, vi)) & 0xFFFF)
            depth = rng.randint(4, max(5, 45 // max(1, n_q // 6)))
            add(f"tof_tower_{n_q}_{vi}", gen_tof_tower(n_q, depth, rng))

    # --- 三类加法器 ---
    for n in range(2, 10):
        if 5 <= CDKMRippleCarryAdder(num_state_qubits=n).num_qubits <= 20:
            for vi in range(variant_offset, variant_offset + per_variant):
                add(f"cdkm_adder_{2*n+2}_{vi}", CDKMRippleCarryAdder(num_state_qubits=n))
    for n in range(2, 7):
        qn = VBERippleCarryAdder(num_state_qubits=n).num_qubits
        if 5 <= qn <= 20:
            for vi in range(variant_offset, variant_offset + per_variant):
                add(f"vbe_adder_{qn}_{vi}", VBERippleCarryAdder(num_state_qubits=n))
    for n in range(3, 11):
        qn = DraperQFTAdder(num_state_qubits=n).num_qubits
        if 5 <= qn <= 20:
            for vi in range(variant_offset, variant_offset + max(1, per_variant // 2)):
                add(f"draper_add_{qn}_{vi}", DraperQFTAdder(num_state_qubits=n))

    # --- QFT 乘法器 ---
    for n in range(2, 6):
        qn = RGQFTMultiplier(num_state_qubits=n).num_qubits
        if 5 <= qn <= 20:
            for vi in range(variant_offset, variant_offset + per_variant):
                add(f"rgqft_mult_{qn}_{vi}", RGQFTMultiplier(num_state_qubits=n))

    # --- GF(2) 乘法 ---
    for n in range(2, 6):
        for vi in range(variant_offset, variant_offset + per_variant):
            add(f"gf2_mult_{4*n}_{vi}", gen_gf2_mult(n, __import__("random").Random(hash((n, vi)) & 0xFFFF)))

    # --- Grover ---
    for m in range(3, 9):
        for it in range(1, 4):
            for vi in range(variant_offset, variant_offset + max(1, per_variant // 3)):
                add(f"grover_{m}_{it}it_{vi}", gen_grover(m, it, __import__("random").Random(m * 7 + it)))

    # --- WeightedAdder ---
    for k in range(4, 15):
        qn = k + 1
        if not (5 <= qn <= 20):
            continue
        for vi in range(variant_offset, variant_offset + max(1, per_variant // 2)):
            add(f"wadd_{qn}_{vi}", WeightedAdder(num_state_qubits=k))

    # --- 随机置换网络 ---
    for n_q in range(6, 21, 2):
        for vi in range(variant_offset, variant_offset + per_variant):
            rng = random.Random(hash((n_q, vi, "perm")) & 0xFFFF)
            perm = list(range(n_q))
            rng.shuffle(perm)
            add(f"perm_{n_q}_{vi}", QuantumCircuit(n_q).compose(PermutationGate(perm)))

    # --- 加法器级联 ---
    for n in range(3, 8):
        for vi in range(variant_offset, variant_offset + max(1, per_variant // 3)):
            add(f"adder_stack_{n}_{vi}", gen_adder_stack(n, 3, CDKMRippleCarryAdder,
                                                         __import__("random").Random(n)))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="traindata/gen_structured")
    ap.add_argument("--per-variant", type=int, default=2,
                    help="每个(族,规模)组合的种子变体数；总电路数 ≈ 150×per_variant")
    ap.add_argument("--variant-offset", type=int, default=0,
                    help="变体索引平移（held-out 用非零值产出训练未见过的种子电路）")
    ap.add_argument("--max-gates", type=int, default=1500,
                    help="转译后门数上限（对齐 NAM max=831 与 episode 步数上限）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    kept = skipped_small = skipped_gates = 0
    for name, qc in build_all(args.per_variant, args.variant_offset):
        try:
            tqc = transpile(qc, basis_gates=BASIS, optimization_level=0, seed_transpiler=0)
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
        from qiskit import qasm2
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write(qasm2.dumps(tqc))
        manifest.append({
            "name": fname, "family": name.rsplit("_", 2)[0] if name.count("_") >= 2 else name,
            "qubits": tqc.num_qubits, "gates": tqc.size(),
            "cx": n_nl, "depth": tqc.depth(),
        })
        kept += 1
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    by_q = {}
    for m in manifest:
        by_q.setdefault(m["qubits"], 0)
        by_q[m["qubits"]] += 1
    print(f"生成 {kept} 条（跳过: 规模 {skipped_small} / 过浅 {skipped_gates}）")
    print("按比特数分布:", dict(sorted(by_q.items())))
    print(f"输出目录: {args.out_dir}  manifest.json 已写")


if __name__ == "__main__":
    main()
