"""Generate hardware topologies for larger qubit counts.

Creates line / ring / grid topologies for n in [8, 10, 12, 16, 20] with
heterogeneous (per-qubit / per-edge) noise parameters, matching the JSON
format of the existing 5-qubit topologies. Uses a fixed seed for
reproducibility.
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TOPO_DIR = os.path.join(ROOT, "topo")

N_SCALES = [8, 10, 12, 16, 20]


def line_coupling(n: int):
    return [[i, i + 1] for i in range(n - 1)]


def ring_coupling(n: int):
    return line_coupling(n) + [[n - 1, 0]]


def grid_coupling(rows: int, cols: int):
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append([q, q + 1])
            if r + 1 < rows:
                edges.append([q, q + cols])
    return edges


def gen_device_params(n: int, coupling: list, rng: np.random.Generator) -> dict:
    t1 = rng.uniform(35.0, 90.0, size=n)
    t2 = rng.uniform(30.0, 70.0, size=n)
    t2 = np.minimum(t2, t1)
    two_q_errs = []
    for q1, q2 in coupling:
        e = rng.uniform(0.005, 0.05)
        two_q_errs.append([q1, q2, round(float(e), 6)])
        two_q_errs.append([q2, q1, round(float(e), 6)])
    xtalk = []
    for q1, q2 in coupling:
        s = rng.uniform(0.0, 0.01)
        if s > 0:
            xtalk.append([q1, q2, round(float(s), 6)])
    return {
        "t1_times": [round(float(v), 3) for v in t1],
        "t2_times": [round(float(v), 3) for v in t2],
        "freq_ghz": [round(float(v), 4) for v in rng.uniform(4.8, 5.4, size=n)],
        "readout_error": [round(float(v), 5) for v in rng.uniform(0.01, 0.05, size=n)],
        "single_q_gate_error": round(float(rng.uniform(0.0005, 0.002)), 6),
        "two_q_gate_error": two_q_errs,
        "shots": 1024,
    }


def write_topo(name: str, desc: str, coupling: list, rng: np.random.Generator):
    n = max(max(e) for e in coupling) + 1
    topo = {
        "name": name,
        "description": desc,
        "num_qubits": n,
        "coupling_map": coupling,
        "device_params": gen_device_params(n, coupling, rng),
        "crosstalk_strength": None,
    }
    path = os.path.join(TOPO_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(topo, f, indent=2)
    print(f"  {path} ({n}q, {len(coupling)} edges)")


def main():
    os.makedirs(TOPO_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    for n in N_SCALES:
        write_topo(f"line_{n}q", f"{n}-qubit linear chain topology",
                   line_coupling(n), rng)
        write_topo(f"ring_{n}q", f"{n}-qubit ring topology",
                   ring_coupling(n), rng)

    grids = [
        ("grid_2x5_10q", 2, 5),
        ("grid_3x4_12q", 3, 4),
        ("grid_4x4_16q", 4, 4),
        ("grid_5x4_20q", 5, 4),
    ]
    for name, rows, cols in grids:
        write_topo(name, f"{rows}x{cols} grid topology ({rows * cols} qubits)",
                   grid_coupling(rows, cols), rng)

    print("\nDone.")


if __name__ == "__main__":
    main()
