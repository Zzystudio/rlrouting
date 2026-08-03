"""Generate training / test datasets for larger qubit counts.

One dataset per scale (qubit count), because the env observation dimension
and the required physical topology both depend on the number of qubits.
Each scale has its own subdirectories and split manifests so a policy
trained at scale n only ever sees n-qubit circuits.

Scales: n in [8, 10, 12, 16, 20]
Per scale:
  - random circuits, 3 difficulty phases (2q-gate counts scale with n)
  - QAOA ansatz circuits
  - VQE (hardware-efficient) ansatz circuits
  - splits:  large_{n}_phase{1,2,3}.txt   (curriculum, train)
             large_{n}_mixed.txt           (mixed, noise-aware phase 2)
             large_{n}_alg.txt             (QAOA+VQE, phase 3)
             large_{n}_test.txt            (held-out test set)
"""

import argparse
import os
import pickle
import random as pyrand

from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp

ROOT = os.path.dirname(os.path.abspath(__file__))
RANDOM_DIR = os.path.join(ROOT, "random")
QAOA_DIR = os.path.join(ROOT, "qaoa")
VQE_DIR = os.path.join(ROOT, "vqe")
SPLITS_DIR = os.path.join(ROOT, "splits")


def _two_q_count(qc) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def make_random_circuits(n: int, target_count: int, two_q_min: int, two_q_max: int,
                         depth_range: range, seed_offset: int) -> list:
    paths = []
    seed = seed_offset
    subdir = os.path.join(RANDOM_DIR, f"n{n}")
    os.makedirs(subdir, exist_ok=True)
    while len(paths) < target_count:
        depth = pyrand.choice(depth_range)
        qc = random_circuit(n, depth, seed=seed,
                            max_operands=2, conditional=False)
        qc2 = qc.decompose(reps=3) if hasattr(qc, "decompose") else qc
        n2 = _two_q_count(qc2)
        if two_q_min <= n2 <= two_q_max:
            name = f"random_n{n}d{depth}_s{seed}"
            path = os.path.join(subdir, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(qc2, f)
            paths.append(path)
        seed += 1
    return paths


def make_qaoa_circuits(n: int, count: int, seed_offset: int) -> list:
    paths = []
    subdir = os.path.join(QAOA_DIR, f"n{n}")
    os.makedirs(subdir, exist_ok=True)
    rng = pyrand.Random(seed_offset)
    for i in range(count):
        reps = rng.randint(1, 3)
        ham_terms = [("Z" * n, 1.0 / n)]
        for j in range(n - 1):
            op = ["I"] * n
            op[j] = "Z"
            op[j + 1] = "Z"
            ham_terms.append(("".join(op), rng.uniform(0.2, 1.0)))
        ham = SparsePauliOp.from_list(ham_terms)
        qc = QAOAAnsatz(ham, reps=reps)
        qc2 = qc.decompose(reps=3) if hasattr(qc, "decompose") else qc
        name = f"qaoa_n{n}r{reps}_{i:03d}"
        path = os.path.join(subdir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(qc2, f)
        paths.append(path)
    return paths


def make_vqe_circuits(n: int, count: int, seed_offset: int) -> list:
    paths = []
    subdir = os.path.join(VQE_DIR, f"n{n}")
    os.makedirs(subdir, exist_ok=True)
    rng = pyrand.Random(seed_offset)
    for i in range(count):
        reps = rng.randint(1, 3)
        ent = rng.choice(["linear", "circular", "sca"])
        qc = EfficientSU2(n, reps=reps, entanglement=ent)
        qc2 = qc.decompose(reps=3) if hasattr(qc, "decompose") else qc
        name = f"vqe_n{n}r{reps}_{ent}_{i:03d}"
        path = os.path.join(subdir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(qc2, f)
        paths.append(path)
    return paths


def phase_ranges(n: int) -> list:
    """2q-gate count ranges per difficulty phase, scaled by n.

    n=5  -> phase1: 3-6,   phase2: 6-15,  phase3: 12-25 (matches existing)
    n=20 -> phase1: 12-24, phase2: 24-60, phase3: 48-100
    """
    return [
        (max(3, n * 3 // 5), max(6, n * 6 // 5)),
        (max(6, n * 6 // 5), n * 3),
        (n * 12 // 5, n * 5),
    ]


def depth_ranges(n: int) -> list:
    return [range(2, 5), range(4, 10), range(6, 14)]


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale datasets")
    parser.add_argument("--scales", type=str, default="8,10,12,16,20",
                        help="comma-separated qubit counts")
    parser.add_argument("--counts", type=str, default="150,300,300,80,80",
                        help="per-phase circuit counts: phase1,phase2,phase3,qaoa,vqe")
    parser.add_argument("--test-count", type=int, default=60,
                        help="held-out test circuits per scale (random circuits)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="random seed (ensures reproducible regeneration)")
    args = parser.parse_args()

    pyrand.seed(args.seed)

    scales = [int(s) for s in args.scales.split(",")]
    c1, c2, c3, cq, cv = (int(c) for c in args.counts.split(","))
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for n in scales:
        print(f"=== n = {n} qubits ===")
        (p1_lo, p1_hi), (p2_lo, p2_hi), (p3_lo, p3_hi) = phase_ranges(n)
        dr = depth_ranges(n)

        phase1 = make_random_circuits(n, c1, p1_lo, p1_hi, dr[0], seed_offset=100_000 + n * 10_000)
        phase2 = make_random_circuits(n, c2, p2_lo, p2_hi, dr[1], seed_offset=200_000 + n * 10_000)
        phase3 = make_random_circuits(n, c3, p3_lo, p3_hi, dr[2], seed_offset=300_000 + n * 10_000)
        qaoa = make_qaoa_circuits(n, cq, seed_offset=400_000 + n * 10_000)
        vqe = make_vqe_circuits(n, cv, seed_offset=500_000 + n * 10_000)

        test = make_random_circuits(n, args.test_count, p2_lo, p3_hi, dr[1],
                                    seed_offset=900_000 + n * 10_000)

        def rel(path):
            return os.path.relpath(path, ROOT)

        def write_split(name, paths):
            path = os.path.join(SPLITS_DIR, name)
            with open(path, "w") as f:
                for p in sorted(paths):
                    f.write(rel(p) + "\n")
            print(f"  {name}: {len(paths)}")

        write_split(f"large_n{n}_phase1.txt", phase1)
        write_split(f"large_n{n}_phase2.txt", phase2)
        write_split(f"large_n{n}_phase3.txt", phase3)
        write_split(f"large_n{n}_mixed.txt", phase2[::3] + phase3[::3] + qaoa[::2] + vqe[::2])
        write_split(f"large_n{n}_alg.txt", qaoa + vqe)
        write_split(f"large_n{n}_test.txt", test)

    print("\nDone.")


if __name__ == "__main__":
    main()
