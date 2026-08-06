#!/usr/bin/env python3
"""Build train/eval split manifests for the Tianyan 176 (60q) dataset.

Circuits live in traindata/random/n{30,40,50,60}/ with depths 2/4/8/12.
Phase manifests follow the stage1 curriculum: phase1 = shallowest (d2),
phase2 = medium (d4), phase3 = deepest (d8+d12), mixed = uniform sample
(used by noise-aware phase 2), test = held-out per-scale sample.

Usage:
  python3 scripts/build_tianyan_splits.py
"""
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..", "traindata")
SPLITS_DIR = os.path.join(ROOT, "splits")
PREFIX = "tianyan"
SCALES = [30, 40, 50, 60]

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

    by_scale = {}
    for n in SCALES:
        by_scale[n] = {d: rel_paths(f"random/n{n}/random_n{n}d{d}_s*.pkl") for d in [2, 4, 8, 12]}

    phase1 = [p for n in SCALES for p in by_scale[n][2]]
    phase2 = [p for n in SCALES for p in by_scale[n][4]]
    phase3 = [p for n in SCALES for p in by_scale[n][8] + by_scale[n][12]]
    all_p = phase1 + phase2 + phase3

    rng.shuffle(phase3)
    rng.shuffle(all_p)
    test = []
    for n in SCALES:
        pool = by_scale[n][2] + by_scale[n][4] + by_scale[n][8]
        rng.shuffle(pool)
        test += pool[:15]

    write("phase1", phase1)
    write("phase2", phase2)
    write("phase3", phase3)
    write("mixed", all_p)
    write("test", test)

if __name__ == "__main__":
    main()