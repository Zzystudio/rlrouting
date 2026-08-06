#!/usr/bin/env python3
"""Generate random circuits for Tianyan 176 (60q) topology training.

Sizes n = 30/40/50/60, depths cover the curriculum (shallow for phase1,
deeper for phase3), following the existing random_n{n}d{d}_s{seed}.pkl
naming convention so CircuitDAG/pickle loading stays uniform.

Usage:
  python3 scripts/gen_tianyan_data.py [out_dir]
"""
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.data_gen import random_circuit

def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "traindata/random"
    scales = [30, 40, 50, 60]
    depths = [2, 4, 8, 12]
    per = 60
    rng = random.Random(0)

    paths = []
    for n in scales:
        out_dir = os.path.join(out_root, f"n{n}")
        os.makedirs(out_dir, exist_ok=True)
        for d in depths:
            for i in range(per):
                seed = 100000 + i * 100 + d
                pkl = os.path.join(out_dir, f"random_n{n}d{d}_s{seed}.pkl")
                if os.path.exists(pkl):
                    continue
                qc = random_circuit(n, d, seed=seed)
                with open(pkl, "wb") as f:
                    pickle.dump(qc, f)
                paths.append(os.path.relpath(pkl, out_root))
    print(f"generated {len(paths)} circuits")

if __name__ == "__main__":
    main()