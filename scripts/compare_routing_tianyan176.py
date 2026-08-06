import json
import time
import sys
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, "src")
from utils.data_gen import random_circuit
from sim.sim import NoiseConfig, NoiseSimulator
from routing.routing import greedy_route, sabre_route

TOPO = "traindata/topo/tianyan176_66q.json"


def load_config(topo_path=TOPO):
    topo = json.load(open(topo_path))
    dp = topo["device_params"]
    return NoiseConfig(
        t1_times=dp["t1_times"], t2_times=dp["t2_times"], freq_ghz=dp["freq_ghz"],
        single_q_gate_error=dp["single_q_gate_error"],
        two_q_gate_error={tuple(eval(k)): v for k, v in dp["two_q_gate_error"].items()},
        coupling_map=[tuple(e) for e in topo["coupling_map"]],
        readout_error=dp["readout_error"],
        shots=2048,
    )


@dataclass
class Row:
    n: int
    depth: int
    seed: int
    method: str
    swaps: int
    phys_gates: int
    depth_after: int
    wall_ms: float
    fidelity: float = None


def phys_depth(qc):
    try:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap([tuple(e) for e in json.load(open(TOPO))["coupling_map"]])
        t = transpile(qc, coupling_map=cm, basis_gates=["rz", "sx", "x", "cx", "swap"],
                      optimization_level=0)
        return t.depth()
    except Exception:
        return None


def compute_fidelity(phys, config):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from sim.trajectory_sim import TrajectorySimulator
    # 提取路由后真正用到的物理比特（其余 60q 拓扑比特空闲，降维后再仿真）
    active = sorted({q._index for inst in phys.data for q in (inst.qubits or ())})
    if len(active) > 24:
        return None  # 超出仿真能力，跳过
    idx = {p: i for i, p in enumerate(active)}
    sub = QuantumCircuit(len(active))
    for inst in phys.data:
        if inst.operation.name == "measure":
            continue
        qs = [idx[q._index] for q in inst.qubits]
        sub.append(inst.operation, qs)
    sub.measure_all()

    # 子配置：只保留活跃比特的参数与活跃边
    dp = json.load(open(TOPO))["device_params"]
    edge_err = {tuple(eval(k)): v for k, v in dp["two_q_gate_error"].items()}
    cm_full = [tuple(e) for e in json.load(open(TOPO))["coupling_map"]]
    emap = {p: i for i, p in enumerate(active)}
    cm_sub = [(emap[p], emap[q]) for (p, q) in cm_full
              if p in emap and q in emap]
    t2e = {}
    for (p, q) in cm_sub:
        for (fp, fq), v in edge_err.items():
            if {fp, fq} == {active[p], active[q]}:
                t2e[(p, q)] = t2e[(q, p)] = v
                break
    sub_cfg = NoiseConfig(
        t1_times=[config.t1_times[p] for p in active],
        t2_times=[config.t2_times[p] for p in active],
        freq_ghz=[config.freq_ghz[p] for p in active],
        single_q_gate_error=[config.single_q_gate_error[p] for p in active],
        two_q_gate_error=t2e,
        coupling_map=cm_sub,
        readout_error=[config.readout_error[p] for p in active],
        shots=config.shots,
    )

    shots = config.shots
    if len(active) <= 16:
        sim = NoiseSimulator(sub_cfg)
        meas_t = sim._transpile(sub)
        ideal = AerSimulator().run(meas_t, shots=shots).result().get_counts()
        noisy = sim.run(meas_t, shots=shots, skip_transpile=True)
    else:
        sim = TrajectorySimulator(sub_cfg, num_trajectories=512, seed=42)
        meas_t = sim._transpile(sub)
        ideal = AerSimulator().run(meas_t, shots=shots).result().get_counts()
        noisy = sim.run(meas_t, shots=min(shots, 512), skip_transpile=True)
    all_k = set(ideal) | set(noisy)
    return sum(min(ideal.get(k, 0), noisy.get(k, 0)) for k in all_k) / shots


def main():
    config = load_config()
    rows = []
    for n, depths in [(10, [10, 20]), (30, [10, 20]), (60, [10])]:
        for depth in depths:
            for seed in range(5):
                qc = random_circuit(n, depth, seed=100 + seed)
                for method in ("greedy", "sabre"):
                    t0 = time.perf_counter()
                    if method == "greedy":
                        phys, info = greedy_route(qc, config)
                    else:
                        phys, info = sabre_route(qc, config, heuristic="decay",
                                                 swap_trials=20, seed=seed)
                    wall_ms = (time.perf_counter() - t0) * 1000
                    fid = None
                    if n == 10:
                        fid = compute_fidelity(phys, config)
                    rows.append(Row(n, depth, seed, method,
                                    info["num_swaps"],
                                    phys.num_nonlocal_gates,
                                    phys.depth(),
                                    wall_ms, fid))
                    print(f"n={n:2d} d={depth:2d} seed={seed} {method:6s} "
                          f"swaps={info['num_swaps']:3d} depth={phys.depth():4d} "
                          f"t={wall_ms:7.1f}ms fid={fid if fid is None else round(fid,4)}",
                          flush=True)

    out = "results/tianyan176_routing_cmp.json"
    with open(out, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=1)

    print("\n" + "=" * 88)
    print(f"Tianyan176 effective topo (60q/81e) routing comparison — {TOPO}")
    print("=" * 88)
    hdr = f"{'n':>3} {'d':>3} {'method':>7} {'swaps':>7} {'phys':>6} {'depth':>6} {'time_ms':>9}"
    print(hdr)
    print("-" * 88)
    for n, depth in sorted({(r.n, r.depth) for r in rows}):
        for method in ("greedy", "sabre"):
            rs = [r for r in rows if r.n == n and r.depth == depth and r.method == method]
            print(f"{n:3d} {depth:3d} {method:>7} "
                  f"{np.mean([r.swaps for r in rs]):7.1f} "
                  f"{np.mean([r.phys_gates for r in rs]):6.1f} "
                  f"{np.mean([r.depth_after for r in rs]):6.1f} "
                  f"{np.mean([r.wall_ms for r in rs]):9.1f}")
    print("\nFidelity (n=10, DM simulator, 2048 shots):")
    for d in sorted({r.depth for r in rows if r.n == 10}):
        for method in ("greedy", "sabre"):
            rs = [r for r in rows if r.n == 10 and r.depth == d and r.method == method]
            fids = [r.fidelity for r in rs]
            print(f"  d={d:2d} {method:6s}: mean fid = {np.mean(fids):.4f} +/- {np.std(fids):.4f}")


if __name__ == "__main__":
    main()
