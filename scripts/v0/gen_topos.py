"""生成 v0 所需的 4/6/7 qubit 拓扑 JSON（line / ring），格式与现存 topo 一致。

用法:
    PYTHONPATH=src python3 scripts/v0/gen_topos.py
输出: traindata/topo/line_4q.json, line_6q.json, ring_6q.json, line_7q.json, ring_7q.json
默认噪声参数与 ibmq_5_line.json 一致（v0 纯路由不依赖噪声，仅需 coupling_map 一致）。
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TOPO_DIR = os.path.join(ROOT, "traindata", "topo")

DEFAULT_DEVICE_PARAMS = {
    "t1_times": None,          # 按 n 填充
    "t2_times": None,
    "freq_ghz": None,
    "readout_error": None,
    "single_q_gate_error": 0.001,
    "two_q_gate_error": 0.01,
    "shots": 1024,
}


def line_coupling(n):
    return [[i, i + 1] for i in range(n - 1)]


def ring_coupling(n):
    return line_coupling(n) + [[n - 1, 0]]


def make_topo(name, n, coupling):
    dp = {
        "t1_times": [50.0] * n,
        "t2_times": [70.0] * n,
        "freq_ghz": [5.0] * n,
        "readout_error": [0.02] * n,
        "single_q_gate_error": 0.001,
        "two_q_gate_error": 0.01,
        "shots": 1024,
    }
    return {
        "name": name,
        "description": f"{name} (v0 生成, line/ring {n}q)",
        "num_qubits": n,
        "coupling_map": coupling,
        "device_params": dp,
        "crosstalk_strength": None,
    }


def main():
    os.makedirs(TOPO_DIR, exist_ok=True)
    specs = {
        "line_4q": (4, line_coupling(4)),
        "line_6q": (6, line_coupling(6)),
        "ring_6q": (6, ring_coupling(6)),
        "line_7q": (7, line_coupling(7)),
        "ring_7q": (7, ring_coupling(7)),
    }
    for name, (n, cm) in specs.items():
        path = os.path.join(TOPO_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(make_topo(name, n, cm), f, indent=2)
        print(f"wrote {path} ({n}q, {len(cm)} edges)")


if __name__ == "__main__":
    main()
