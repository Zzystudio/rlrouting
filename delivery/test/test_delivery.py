"""交付包冒烟测试：验证模型加载、路由有效性、耦合图合法性与指标输出。

运行（在 delivery/ 目录下）：
    python -m pytest test/ -q
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_route  # noqa: E402
from routing.rl.eval_policy import load_topo  # noqa: E402

TOPO = os.path.join(ROOT, "topologies", "tianyan176_20q.json")
MODEL = os.path.join(ROOT, "models", "policy_r3b.pt")


@pytest.fixture(scope="module")
def topo():
    config, hw, coupling_map = load_topo(TOPO)
    return config, hw, coupling_map


def _edges(coupling_map):
    return {frozenset(e) for e in coupling_map}


def _check_coupling_validity(routed_qasm_path, coupling_map):
    """路由结果中所有 2Q 门必须落在耦合图的边上。"""
    from qiskit.qasm2 import load as qasm2_load
    qc = qasm2_load(routed_qasm_path)
    edges = _edges(coupling_map)
    for inst in qc.data:
        if len(inst.qubits) == 2:
            pair = frozenset(qc.find_bit(q).index for q in inst.qubits)
            assert pair in edges, f"2Q 门 {inst.operation.name} 作用在非耦合边 {pair}"


def test_profile_detection():
    prof = run_route.detect_profile(MODEL)
    assert prof["profile"] == "r3b"
    assert prof["lookahead_features"] is False


def test_route_ghz_argmax(topo, tmp_path):
    """GHZ 5q：argmax 完成、耦合图合法、SWAP 数合理。"""
    config, hw, coupling_map = topo
    phys, metrics = run_route.route_with_model(
        _load_qc("examples/toy/ghz_5.qasm"), config, hw, coupling_map,
        MODEL, beam_width=0)
    assert metrics["completed"]
    assert metrics["num_swaps"] <= 8
    out = tmp_path / "ghz5"
    _dump(phys, str(out))
    _check_coupling_validity(str(out) + ".qasm", coupling_map)


def test_route_tof3_beam(topo):
    """tof_3：beam3 完成、SWAP 数与 SABRE 同量级（±4 颗内）。"""
    config, hw, coupling_map = topo
    phys, metrics = run_route.route_with_model(
        _load_qc("examples/nam_circs/tof_3.qasm"), config, hw, coupling_map,
        MODEL, beam_width=3)
    assert metrics["completed"]
    _, sab = run_route.route_with_sabre(
        _load_qc("examples/nam_circs/tof_3.qasm"), config)
    assert abs(metrics["num_swaps"] - sab["num_swaps"]) <= 4


def test_sabre_baseline(topo):
    config, hw, coupling_map = topo
    _, metrics = run_route.route_with_sabre(
        _load_qc("examples/nam_circs/tof_3.qasm"), config)
    assert metrics["completed"] and metrics["num_swaps"] >= 0


def test_fidelity_estimate(topo):
    """v1 调度感知模拟器保真度估计落在 [0,1]。"""
    config, hw, coupling_map = topo
    phys, metrics = run_route.route_with_model(
        _load_qc("examples/toy/ghz_5.qasm"), config, hw, coupling_map,
        MODEL, beam_width=0)
    assert metrics["completed"]
    from sim.trajectory_sim import trajectory_circuit_fidelity
    fid = trajectory_circuit_fidelity(
        phys, config, num_trajectories=8, seed=0, scheduled=True)
    assert 0.0 <= fid <= 1.0


def test_cli_end_to_end(tmp_path):
    """CLI 端到端：生成 .qasm + .json 双输出。"""
    import subprocess
    out = tmp_path / "cli"
    r = subprocess.run(
        [sys.executable, "run_route.py",
         "--circuit", os.path.join("examples", "toy", "ghz_5.qasm"),
         "--topo", TOPO, "--out", str(out), "--fidelity", "8"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    assert os.path.exists(str(out) + ".qasm")
    m = json.load(open(str(out) + ".json"))
    assert m["completed"] and 0.0 <= m["fidelity_v1"] <= 1.0


# ---------------------------------------------------------------------------
def _load_qc(rel: str):
    from qiskit.qasm2 import load as qasm2_load
    return qasm2_load(os.path.join(ROOT, rel))


def _dump(qc, path_prefix: str):
    from qiskit.qasm2 import dumps as qasm2_dumps
    with open(path_prefix + ".qasm", "w") as f:
        f.write(qasm2_dumps(qc))
