"""Generate unified split manifests that mix circuits from all scales (n=8..20).

Reads the existing per-scale split files and writes unified versions with
equal representation per scale in each phase, so training sees balanced
episode counts across scales.
"""

import os
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(ROOT, "splits")
SCALES = [8, 10, 12, 16, 20]


def read_split(name: str) -> list:
    path = os.path.join(SPLITS_DIR, name + ".txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing split: {path}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def sample_from_scales(phase_suffix: str, per_scale: int, rng: random.Random) -> list:
    entries = []
    for n in SCALES:
        lines = read_split(f"large_n{n}_{phase_suffix}")
        entries.extend(rng.sample(lines, min(per_scale, len(lines))))
    return entries


def mixed_from_scales(rng: random.Random) -> list:
    entries = []
    min_p2 = min(len(read_split(f"large_n{n}_phase2")) for n in SCALES)
    min_p3 = min(len(read_split(f"large_n{n}_phase3")) for n in SCALES)
    min_q = min(len(read_split(f"large_n{n}_alg")) for n in SCALES)
    sub2 = min_p2 // 6
    sub3 = min_p3 // 6
    suba = min_q // 4
    for n in SCALES:
        entries.extend(rng.sample(read_split(f"large_n{n}_phase2"), sub2))
        entries.extend(rng.sample(read_split(f"large_n{n}_phase3"), sub3))
        entries.extend(rng.sample(read_split(f"large_n{n}_alg"), suba))
    return entries


def main():
    rng = random.Random(2026)

    def write(name, lines):
        path = os.path.join(SPLITS_DIR, name + ".txt")
        with open(path, "w") as f:
            for l in sorted(lines):
                f.write(l + "\n")
        print(f"  {name}: {len(lines)}")

    print("Unified splits:")
    write("unified_phase1", sample_from_scales("phase1", 30, rng))
    write("unified_phase2", sample_from_scales("phase2", 60, rng))
    write("unified_phase3", sample_from_scales("phase3", 60, rng))
    write("unified_mixed", mixed_from_scales(rng))
    write("unified_alg", sample_from_scales("alg", 32, rng))
    write("unified_test", sample_from_scales("test", 12, rng))

    print("Done.")


if __name__ == "__main__":
    main()
