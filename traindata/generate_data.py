"""Generate training dataset for rlrouting.

All circuits use a fixed number of logical qubits so the env observation
dimension stays consistent across episodes.
"""

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
for d in [RANDOM_DIR, QAOA_DIR, VQE_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

N_QUBITS = 5  # fixed (matches `--num-qubits 5` in train_agent.py)


def _two_q_count(qc) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


# ---------------------------------------------------------------------------
#  Random circuits  (Stage 1 curriculum: 3 phases)
# ---------------------------------------------------------------------------
print(f"Generating random circuits ({N_QUBITS} qubits) ...")


def make_random_circuits(target_count: int, two_q_min: int, two_q_max: int,
                         depths: range, seed_offset: int):
    paths = []
    seed = seed_offset
    while len(paths) < target_count:
        depth = pyrand.choice(depths)
        qc = random_circuit(N_QUBITS, depth, seed=seed,
                            max_operands=2, conditional=False)
        qc2 = qc.decompose(reps=3) if hasattr(qc, "decompose") else qc
        n2 = _two_q_count(qc2)
        if two_q_min <= n2 <= two_q_max:
            name = f"random_n{N_QUBITS}d{depth}_s{seed}"
            path = os.path.join(RANDOM_DIR, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(qc2, f)
            paths.append(path)
        seed += 1
    return paths


phase1_paths = make_random_circuits(150, 3, 6, range(2, 5), seed_offset=0)
print(f"  Phase 1 (3-6 2q gates): {len(phase1_paths)}")

phase2_paths = make_random_circuits(300, 6, 15, range(4, 10), seed_offset=2000)
print(f"  Phase 2 (6-15 2q gates): {len(phase2_paths)}")

phase3_paths = make_random_circuits(300, 12, 25, range(6, 14), seed_offset=5000)
print(f"  Phase 3 (12-25 2q gates): {len(phase3_paths)}")

# ---------------------------------------------------------------------------
#  QAOA circuits  (Stage 2/3)
# ---------------------------------------------------------------------------
print(f"Generating QAOA circuits ({N_QUBITS} qubits) ...")
qaoa_paths = []
for i in range(80):
    reps = pyrand.randint(1, 3)
    ham_terms = [("Z" * N_QUBITS, 1.0 / N_QUBITS)]
    for j in range(N_QUBITS - 1):
        op = ["I"] * N_QUBITS
        op[j] = "Z"
        op[j + 1] = "Z"
        ham_terms.append(("".join(op), pyrand.uniform(0.2, 1.0)))
    ham = SparsePauliOp.from_list(ham_terms)
    qc = QAOAAnsatz(ham, reps=reps)
    qc2 = qc.decompose(reps=3) if hasattr(qc, "decompose") else qc
    name = f"qaoa_n{N_QUBITS}r{reps}_{i:03d}"
    path = os.path.join(QAOA_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(qc2, f)
    qaoa_paths.append(path)
print(f"  {len(qaoa_paths)}")

# ---------------------------------------------------------------------------
#  VQE (hardware-efficient ansatz)  (Stage 2/3)
# ---------------------------------------------------------------------------
print(f"Generating VQE ansatz circuits ({N_QUBITS} qubits) ...")
vqe_paths = []
for i in range(80):
    reps = pyrand.randint(1, 3)
    ent = pyrand.choice(["linear", "circular", "sca"])
    qc = EfficientSU2(N_QUBITS, reps=reps, entanglement=ent)
    qc2 = qc.decompose(reps=3) if hasattr(qc, "decompose") else qc
    name = f"vqe_n{N_QUBITS}r{reps}_{ent}_{i:03d}"
    path = os.path.join(VQE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(qc2, f)
    vqe_paths.append(path)
print(f"  {len(vqe_paths)}")

# ---------------------------------------------------------------------------
#  Write split manifests
# ---------------------------------------------------------------------------


def rel(path):
    return os.path.relpath(path, ROOT)


with open(os.path.join(SPLITS_DIR, "stage1_phase1.txt"), "w") as f:
    for p in sorted(phase1_paths):
        f.write(rel(p) + "\n")
with open(os.path.join(SPLITS_DIR, "stage1_phase2.txt"), "w") as f:
    for p in sorted(phase2_paths):
        f.write(rel(p) + "\n")
with open(os.path.join(SPLITS_DIR, "stage1_phase3.txt"), "w") as f:
    for p in sorted(phase3_paths):
        f.write(rel(p) + "\n")

stage2_paths = phase2_paths[::3] + phase3_paths[::3] + qaoa_paths[::2] + vqe_paths[::2]
with open(os.path.join(SPLITS_DIR, "stage2_mixed.txt"), "w") as f:
    for p in sorted(stage2_paths):
        f.write(rel(p) + "\n")

stage3_paths = qaoa_paths + vqe_paths
with open(os.path.join(SPLITS_DIR, "stage3_alg.txt"), "w") as f:
    for p in sorted(stage3_paths):
        f.write(rel(p) + "\n")

print("\nDone. Splits:")
for split in sorted(os.listdir(SPLITS_DIR)):
    path = os.path.join(SPLITS_DIR, split)
    n = sum(1 for _ in open(path))
    print(f"  {split}: {n}")
