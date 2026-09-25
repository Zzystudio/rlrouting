import sys, os, pickle
import numpy as np
sys.path.insert(0, 'src')

def _worker(task):
    (circuit_file, topo_path, variant, n_perturb, seed, n_traj) = task
    try:
        import signal
        def _alarm(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 300.0)
        from routing.graph.circuit_dag import CircuitDAG
        from routing.v0.baselines import load_topo_full, make_env
        from routing.v0.noise_rollout import (noise_aware_rollout,
                                              phys_circuit_from_ops)
        from sim.trajectory_sim_v3 import trajectory_circuit_fidelity_events_v3
        topo, cm, hw, config = load_topo_full(topo_path)
        with open(circuit_file, "rb") as f:
            qc = pickle.load(f)
        dag = CircuitDAG.from_circuit(qc)
        env = make_env(dag, cm)
        rng = np.random.default_rng(seed)
        for _ in range(n_perturb):
            if env.is_terminal():
                break
            legal = env.legal_actions()
            env.step(int(legal[rng.integers(0, len(legal))]))
        dist_noise = hw.dist_noise(1.0) * hw.num_qubits
        dist = dist_noise if variant == "noise" else None
        cost, ok, tl = noise_aware_rollout(env.clone(), config,
                                           dist_noise=dist)
        if not ok:
            return {"ok": False, "reason": "rollout_fail"}
        n_phys = max(max(e) for e in cm) + 1
        phys = phys_circuit_from_ops(tl.ops, n_phys)
        fid = trajectory_circuit_fidelity_events_v3(
            phys, config, num_trajectories=n_traj, seed=0, backend="cpu")
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return {"ok": True, "cost": float(cost), "fid": float(fid),
                "circuit": os.path.basename(circuit_file),
                "variant": variant, "n_perturb": n_perturb,
                "mech": {"depol2": tl.acc.depol2, "depol1": tl.acc.depol1,
                         "zz_static": tl.acc.zz_static,
                         "zz_dyn": tl.acc.zz_dyn,
                         "thermal": tl.acc.thermal},
                "swaps": sum(1 for op in tl.ops if op[0] == "swap")}
    except Exception as e:  # noqa: BLE001
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        except Exception:
            pass
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


