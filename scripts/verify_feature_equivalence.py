"""T1/T2 优化后与原实现的特征等价性验证。

将优化前的 _sabre_edge_features / build_routing_graph 逻辑内嵌为参考实现，
在多条电路的真实 episode 轨迹上逐步对比输出（要求逐元素相等）。
"""
import sys
import numpy as np
import torch
from qiskit import qasm2

sys.path.insert(0, "src")
from routing.graph.circuit_dag import CircuitDAG, build_routing_graph
from routing.rl.env import RoutingEnv
from routing.rl.eval_policy import load_topo
from routing.gnn.encoder import SubGNN


def sabre_reference(env):
    """优化前的 _sabre_edge_features 原始实现（逐字保留）。"""
    n_ready = max(len(env._ready_2q_gates()), 1)
    dist_before = env._front_layer_dist()
    feats = np.zeros((env.num_edges, 5), dtype=np.float32)

    for i, (p, q) in enumerate(env.coupling_map):
        tmp_map = env.mapping.copy()
        inv = {phys: log for log, phys in enumerate(tmp_map)}
        lp, lq = inv.get(p), inv.get(q)
        if lp is not None and lq is not None:
            tmp_map[lp], tmp_map[lq] = tmp_map[lq], tmp_map[lp]
        elif lp is not None:
            tmp_map[lp] = q
        elif lq is not None:
            tmp_map[lq] = p

        dist_after = env._front_layer_dist(tmp_map)
        feats[i, 0] = dist_before / max(env.num_qubits, 1)
        feats[i, 1] = dist_after / max(env.num_qubits, 1)
        feats[i, 2] = (dist_before - dist_after) / max(dist_before, 1e-8)

        improved = 0
        worsened = 0
        for g in env._ready_2q_gates():
            qa, qb = g.qubits
            d_b = env.hw.dist[env.mapping[qa], env.mapping[qb]]
            d_a = env.hw.dist[tmp_map[qa], tmp_map[qb]]
            if d_a < d_b - 1e-8:
                improved += 1
            elif d_a > d_b + 1e-8:
                worsened += 1
        feats[i, 3] = improved / n_ready
        feats[i, 4] = worsened / n_ready
    return feats


def compare_circuit(circuit_path, env_kwargs, max_steps=500, seed=0):
    qc = qasm2.load(circuit_path)
    dag = CircuitDAG.from_circuit(qc)
    env = RoutingEnv(dag, env_kwargs["hw"], env_kwargs["coupling_map"],
                     reward_mode="routing", max_episode_steps=1000, seed=seed,
                     gnn=None, use_gnn=False, noise_config=None,
                     max_num_qubits=env_kwargs["max_num_qubits"],
                     max_num_edges=len(env_kwargs["coupling_map"]),
                     mapping_phase=True, fidelity_fn=None, use_scheduler=False)
    obs, _ = env.reset()
    rng = np.random.default_rng(seed)
    n = 0
    n_ready_steps = 0
    while n < max_steps:
        new_f = env._sabre_edge_features()
        ref_f = sabre_reference(env)
        if not np.array_equal(new_f, ref_f):
            raise AssertionError(
                f"{circuit_path} step {n}: sabre feats differ\n"
                f"new={new_f}\nref={ref_f}")
        if len(env._ready_2q_gates()) > 0:
            n_ready_steps += 1
        # 随机合法动作推进，覆盖 commit / swap / 各种映射状态
        mask = np.zeros(len(env.coupling_map) + 1, dtype=bool)
        mask[:len(env.coupling_map)] = True
        if env.mapping_phase:
            dm = env.get_deadlock_mask()
            um = env.get_unmapped_mask()
            for i in range(min(len(dm), len(env.coupling_map))):
                if dm[i] or um[i]:
                    mask[i] = False
            mask[len(env.coupling_map)] = env.mapping_phase
        valid = np.where(mask)[0]
        action = int(rng.choice(valid))
        obs, r, done, trunc, info = env.step(action)
        n += 1
        if done or trunc:
            break
    return n, n_ready_steps


def main():
    config, hw, coupling_map = load_topo('traindata/topo/tianyan176_20q.json')
    kw = {"hw": hw, "coupling_map": coupling_map, "max_num_qubits": 20}
    circuits = [
        "benchmark/nam_circs/mod5_4.qasm",
        "benchmark/nam_circs/tof_3.qasm",
        "benchmark/nam_circs/barenco_tof_5.qasm",
        "benchmark/nam_circs/hwb6.qasm",
        "benchmark/nam_circs/grover_5.qasm",
    ]
    total = 0
    for c in circuits:
        n, nr = compare_circuit(c, kw, max_steps=300, seed=hash(c) % 1000)
        total += n
        print(f"  OK {c}: {n} steps ({nr} with non-empty front layer)")
    print(f"ALL EQUIVALENT: {total} steps compared")


if __name__ == "__main__":
    main()


# ============================ T2: build_routing_graph ============================

def build_routing_graph_reference(dag, mapping, hw, coupling_map,
                                  single_gate_time=0.1, two_gate_time=0.3,
                                  executed_mask=None, executable_2q=None):
    """优化前的 build_routing_graph 原始实现（逐字保留）。"""
    from routing.graph.circuit_dag import RoutingGraphData, _nearest_occupied_distance
    from routing.graph.features import EDGE_FEATURE_DIM as _EFD
    G = dag.num_gates
    M = dag.num_logical_qubits
    P = hw.num_qubits

    if executed_mask is None:
        executed_mask = np.zeros(G, dtype=bool)

    depths = dag.dag_depths()
    remaining = dag.remaining_depths()
    succ = dag.successors()
    max_depth = dag.max_depth()
    max_dist = int(max(1, hw.dist.max() * P))

    max_in_deg = max((len(g.predecessors) for g in dag.gates), default=1)
    max_out_deg = max((len(succ[g.index]) for g in dag.gates), default=1)

    occupied_mask = np.zeros(P, dtype=bool)
    for lq, pq in enumerate(mapping):
        occupied_mask[pq] = True

    pending_count = np.zeros(P, dtype=float)
    total_pending = 0
    for g in dag.gates:
        if not executed_mask[g.index]:
            for q in g.qubits:
                pq = mapping[q]
                pending_count[pq] += 1.0
                total_pending += 1

    executable_on_qubit = {}
    for g in dag.gates:
        if g.is_two_qubit and executable_2q and g.index in executable_2q:
            for q in g.qubits:
                pq = mapping[q]
                executable_on_qubit[pq] = executable_on_qubit.get(pq, 0) + 1
    max_exec = max(executable_on_qubit.values()) if executable_on_qubit else 1

    nearest_dist = _nearest_occupied_distance(hw.adj, occupied_mask)
    diameter = int(hw.dist.max() * P)
    diameter = max(1, diameter)

    gate_template = dag.build_gate_template()
    gate_feat = gate_template.copy()
    for g in dag.gates:
        phys = [mapping[q] for q in g.qubits]
        if g.is_measure:
            err = hw.readout[phys[0]] if phys else 0.0
        elif not g.is_two_qubit:
            err = hw.single_q_err[phys[0]] if phys else 0.0
        else:
            if len(phys) >= 2:
                err = max(hw.two_q_err[phys[0], phys[1]], 0.0)
            else:
                err = hw.two_q_err.max()

        is_executed = bool(executed_mask[g.index])
        if is_executed:
            exec_status = 2.0
        elif g.is_two_qubit and executable_2q and g.index in executable_2q:
            exec_status = 1.0
        elif not g.is_two_qubit and all(executed_mask[p] for p in g.predecessors):
            exec_status = 1.0
        else:
            exec_status = 0.0

        n_rem_pred = sum(1 for p in g.predecessors if not executed_mask[p])
        rem_pred_norm = n_rem_pred / max(1, len(g.predecessors))

        if g.is_two_qubit and len(phys) >= 2:
            map_dist = hw.dist[phys[0], phys[1]] * P
            map_dist_norm = map_dist / max(1, max_dist)
            is_adj = 1.0 if hw.adj[phys[0], phys[1]] > 0 else 0.0
        else:
            map_dist_norm = 0.0
            is_adj = 0.0

        gate_feat[g.index, 15] = err
        gate_feat[g.index, 22] = exec_status
        gate_feat[g.index, 23] = rem_pred_norm
        gate_feat[g.index, 25] = map_dist_norm
        gate_feat[g.index, 26] = is_adj

    qubit_feat = hw.qubit_template.copy()
    for pq in range(P):
        occupied = 1.0 if occupied_mask[pq] else 0.0
        occupancy = pending_count[pq] / max(1, total_pending)
        exec_dep = executable_on_qubit.get(pq, 0) / max(1, max_exec)
        near_dist = nearest_dist[pq] / max(1, diameter)
        neighbors = [n for n in range(P) if hw.adj[pq, n] > 0]
        degree = len(neighbors)
        occ_neighbors = sum(1 for n in neighbors if occupied_mask[n])
        occ_neighbor_norm = occ_neighbors / max(1, degree) if degree > 0 else 0.0

        qubit_feat[pq, 13] = occupied
        qubit_feat[pq, 14] = occupancy
        qubit_feat[pq, 16] = exec_dep
        qubit_feat[pq, 17] = near_dist
        qubit_feat[pq, 18] = occ_neighbor_norm

    dep_index, dep_template = dag.build_dep_template()
    dep_attr_arr = dep_template.copy()
    if dep_attr_arr.shape[0] > 0:
        edge_idx = 0
        for g in dag.gates:
            for p in g.predecessors:
                src_exec = 1.0 if executed_mask[p] else 0.0
                n_rem = sum(1 for pp in g.predecessors if not executed_mask[pp])
                tgt_rem = n_rem / max(1, len(g.predecessors))
                dep_attr_arr[edge_idx, 8] = src_exec
                dep_attr_arr[edge_idx, 9] = tgt_rem
                edge_idx += 1

    coup_index = hw.coupling_index
    coup_attr_arr = hw.coupling_template.copy()
    if coup_attr_arr.shape[0] > 0:
        for i in range(0, len(coupling_map) * 2, 2):
            q1, q2 = coupling_map[i // 2]
            both_occ = 1.0 if (occupied_mask[q1] and occupied_mask[q2]) else 0.0
            coup_attr_arr[i, 10] = both_occ
            coup_attr_arr[i + 1, 10] = both_occ

    map_src, map_tgt = [], []
    map_attr = []
    for g in dag.gates:
        for role, q in enumerate(g.qubits):
            pq = mapping[q]
            map_src.append(g.index)
            map_tgt.append(pq)
            neighbors = [n for n in range(P) if hw.adj[pq, n] > 0]
            avg_two = np.mean([hw.two_q_err[pq, n] for n in neighbors]) if neighbors else 0.0
            dist_to_other = 0.0
            if g.is_two_qubit and len(g.qubits) == 2:
                other_role = 1 - role
                other_pq = mapping[g.qubits[other_role]]
                dist_to_other = hw.dist[pq, other_pq] * P
                dist_to_other /= max(1, diameter)
            from routing.graph.features import maps_to_edge_feature
            map_attr.append(maps_to_edge_feature(
                phys_index_norm=pq / max(1, P - 1),
                role=float(role),
                t1_norm=hw.t1[pq],
                t2_norm=hw.t2[pq],
                freq_norm=hw.freq[pq],
                readout_err_norm=hw.readout[pq],
                single_q_err_norm=hw.single_q_err[pq],
                avg_two_q_err_norm=avg_two,
                occupied=1.0 if occupied_mask[pq] else 0.0,
                distance_to_other_norm=dist_to_other,
            ))

    map_index = np.asarray([map_src, map_tgt], dtype=int) if map_src else np.empty((2, 0), dtype=int)
    map_attr_arr = np.asarray(map_attr, dtype=float) if map_attr else np.empty((0, _EFD), dtype=float)

    return RoutingGraphData(
        num_gates=G, num_physical=P,
        gate_feat=gate_feat,
        dep_edge_index=dep_index, dep_edge_attr=dep_attr_arr,
        qubit_feat=qubit_feat,
        coupling_edge_index=coup_index, coupling_edge_attr=coup_attr_arr,
        map_edge_index=map_index, map_edge_attr=map_attr_arr,
        coupling_edges=list(coupling_map),
    )


def compare_graph(circuit_path, env_kwargs, max_steps=200, seed=0):
    from routing.graph.circuit_dag import build_routing_graph
    qc = qasm2.load(circuit_path)
    dag = CircuitDAG.from_circuit(qc)
    env = RoutingEnv(dag, env_kwargs["hw"], env_kwargs["coupling_map"],
                     reward_mode="routing", max_episode_steps=1000, seed=seed,
                     gnn=None, use_gnn=False, noise_config=None,
                     max_num_qubits=env_kwargs["max_num_qubits"],
                     max_num_edges=len(env_kwargs["coupling_map"]),
                     mapping_phase=True, fidelity_fn=None, use_scheduler=False)
    env.reset()
    rng = np.random.default_rng(seed)
    n = 0
    while n < max_steps:
        executed_mask = np.zeros(dag.num_gates, dtype=bool)
        for gi in env.executed:
            executed_mask[gi] = True
        ex2 = set(env.executable_2q)
        new = build_routing_graph(dag, env.mapping, env_kwargs["hw"],
                                  env_kwargs["coupling_map"],
                                  executed_mask=executed_mask, executable_2q=ex2)
        ref = build_routing_graph_reference(dag, env.mapping, env_kwargs["hw"],
                                            env_kwargs["coupling_map"],
                                            executed_mask=executed_mask, executable_2q=ex2)
        for name in ["gate_feat", "dep_edge_attr", "qubit_feat",
                     "coupling_edge_attr", "map_edge_attr"]:
            a = getattr(new, name); b = getattr(ref, name)
            if a.shape != b.shape:
                raise AssertionError(f"{circuit_path} step {n} {name}: shape {a.shape} vs {b.shape}")
            if not np.array_equal(a, b):
                d = np.argwhere(a != b)
                raise AssertionError(
                    f"{circuit_path} step {n} {name}: {len(d)} diffs, "
                    f"first at {d[0]}: new={a[tuple(d[0])]} ref={b[tuple(d[0])]}")
        if not np.array_equal(new.map_edge_index, ref.map_edge_index):
            raise AssertionError(f"{circuit_path} step {n}: map_edge_index differ")
        mask = np.zeros(len(env.coupling_map) + 1, dtype=bool)
        mask[:len(env.coupling_map)] = True
        if env.mapping_phase:
            dm = env.get_deadlock_mask(); um = env.get_unmapped_mask()
            for i in range(min(len(dm), len(env.coupling_map))):
                if dm[i] or um[i]:
                    mask[i] = False
            mask[len(env.coupling_map)] = env.mapping_phase
        valid = np.where(mask)[0]
        action = int(rng.choice(valid))
        env.step(action)
        n += 1
        if done_flag(env):
            break
    return n


def done_flag(env):
    return False  # 固定步数，覆盖尽量多状态
