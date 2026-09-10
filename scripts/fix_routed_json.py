#!/usr/bin/env python3
"""Fix existing routed JSON files:
1. Swap initial_layout / final_layout (they were named wrong)
2. Convert Q labels → physical indices (0-19)
3. Add from_0_19 reference field
"""
import json, os, sys, glob

LABEL_FILE = "traindata/topo/tianyan176_20q_labels.json"
ROUTED_DIR = "benchmark/routed"

with open(LABEL_FILE) as f:
    label_data = json.load(f)
to_0_19 = {k: int(v) for k, v in label_data["to_0_19"].items()}
from_0_19 = label_data["from_0_19"]

files = sorted(glob.glob(os.path.join(ROUTED_DIR, "**", "*.json"), recursive=True))
files = [f for f in files if not f.endswith("_summary.json")]

fixed = 0
skipped = 0
for fpath in files:
    with open(fpath) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        skipped += 1
        continue

    il = data.get("initial_layout")
    fl = data.get("final_layout")
    if il is None or fl is None:
        skipped += 1
        continue

    # Check if already fixed (values are ints, not Q-label strings)
    sample_val = next(iter(il.values()), None)
    if isinstance(sample_val, int):
        # Already new format, just ensure from_0_19 exists
        if "from_0_19" not in data:
            data["from_0_19"] = from_0_19
            with open(fpath, "w") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            fixed += 1
        continue

    # Old format: Q-label strings, swapped names
    # old initial_layout = actual FINAL layout (Q labels)
    # old final_layout = actual INITIAL layout (Q labels)
    new_initial = {k: to_0_19[v] for k, v in fl.items()}
    new_final = {k: to_0_19[v] for k, v in il.items()}

    data["initial_layout"] = new_initial
    data["final_layout"] = new_final
    data["from_0_19"] = from_0_19

    with open(fpath, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    fixed += 1

print(f"Fixed: {fixed}, Skipped: {skipped}, Total: {len(files)}")
