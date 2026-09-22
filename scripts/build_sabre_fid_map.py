#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 sabre_fid_map：每个逻辑比特数 n 的 SABRE+D2@v3 平均保真度。

供 C3 终端奖励 log-ratio 归一化（λ_fid·(log F − log sref)），使保真度信号
与路由稠密奖励同量级（当前 sref=None → 原始 F ~0.001-0.15，被淹没）。
按课程 n 采样电路（large_n8/10/12/16/20/unified），SABRE 路由 + v3 fidelity
（T=64，多 seed 平均降误差）。输出 JSON {n: fid}。
用法（项目根）：PYTHONPATH=src python3 scripts/build_sabre_fid_map.py \
    [--trajectories 64] [--n-per 5] [--device cuda:0]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import QuantumCircuit
from routing.rl.eval_policy import load_topo, load_qc
from routing.routing import sabre_route
from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3

TOPO = "traindata/topo/tianyan287_20q.json"
SPLITS = {
    8: "traindata/splits/large_n8_phase1.txt",
    10: "traindata/splits/large_n10_phase1.txt",
    12: "traindata/splits/large_n12_phase1.txt",
    16: "traindata/splits/large_n16_phase1.txt",
    20: "traindata/splits/large_n20_phase1.txt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=int, default=64)
    ap.add_argument("--n-per", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/tmp/opencode/sabre_fid_map.json")
    args = ap.parse_args()

    config, hw, cm = load_topo(TOPO)
    result = {}
    for n, split in sorted(SPLITS.items()):
        paths = [l.strip() for l in open(split) if l.strip()]
        fids = []
        for rel in paths[: args.n_per]:
            qc = load_qc("traindata", rel, seed=0)
            phys, info = sabre_route(qc, config, swap_trials=5, seed=0)
            f = trajectory_circuit_fidelity_events_v3(
                phys, config, num_trajectories=args.trajectories,
                backend=args.device)
            fids.append(float(f))
        result[str(n)] = round(float(np.mean(fids)), 6)
        print(f"n={n}: mean SABRE@v3 fid = {result[str(n)]} "
              f"(n_circ={len(fids)}, T={args.trajectories})")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"写入 {args.out}: {result}")


if __name__ == "__main__":
    main()
