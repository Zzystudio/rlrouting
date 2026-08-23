#!/usr/bin/env python3
"""Build train/eval split manifests for the Tianyan-176 20q sub-topology.

Circuits are reused from traindata/random/n20/ (topology-agnostic pkls,
depths d2..d13, 810 circuits total). Phase manifests follow the stage1
curriculum: phase1 = shallow (d2+d3), phase2 = medium (d4+d5+d6),
phase3 = deep (d7..d13), mixed = all shuffled (used by noise-aware
phase 2), test = held-out sample from the d2/d4/d8 pool.

Usage:
  python3 scripts/build_tianyan20q_splits.py
"""
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..", "traindata")
SPLITS_DIR = os.path.join(ROOT, "splits")
PREFIX = "tianyan20q"
N = 20
DEPTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def rel_paths(pattern):
    import glob
    return sorted(glob.glob(os.path.join(ROOT, pattern)))

def write(name, paths):
    with open(os.path.join(SPLITS_DIR, f"{PREFIX}_{name}.txt"), "w") as f:
        f.write("\n".join(os.path.relpath(p, ROOT) for p in paths))
        f.write("\n")
    print(f"{PREFIX}_{name}: {len(paths)}")

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    rng = random.Random(42)

    by_depth = {d: rel_paths(f"random/n{N}/random_n{N}d{d}_s*.pkl") for d in DEPTHS}
    for d, paths in by_depth.items():
        print(f"  d{d}: {len(paths)}")

    phase1 = by_depth[2] + by_depth[3]
    phase2 = by_depth[4] + by_depth[5] + by_depth[6]
    phase3 = [p for d in [7, 8, 9, 10, 11, 12, 13] for p in by_depth[d]]
    all_p = phase1 + phase2 + phase3
    assert len(all_p) == sum(len(p) for p in by_depth.values()), "circuit count mismatch"

    rng.shuffle(phase3)
    rng.shuffle(all_p)
    pool = by_depth[2] + by_depth[4] + by_depth[8]
    rng.shuffle(pool)
    test = pool[:30]

    write("phase1", phase1)
    write("phase2", phase2)
    write("phase3", phase3)
    write("mixed", all_p)
    write("test", test)

if __name__ == "__main__":
    main()