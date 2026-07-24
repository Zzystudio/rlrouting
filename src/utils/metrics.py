# ============================================================================
# metrics.py
# 保真度计算与电路距离度量工具。
# ============================================================================

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from qiskit.quantum_info import Statevector, DensityMatrix
except Exception:  # pragma: no cover - qiskit 可选依赖
    Statevector = None
    DensityMatrix = None


def state_fidelity(ideal_sv: np.ndarray, noisy_dm: np.ndarray) -> float:
    """计算理想态矢量与含噪密度矩阵之间的保真度。

    F = <psi| rho |psi>  （对纯态目标的保真度，取值 [0, 1]）

    Parameters
    ----------
    ideal_sv : 一维复向量，理想态矢量
    noisy_dm : 二维复矩阵，含噪密度矩阵
    """
    psi = np.asarray(ideal_sv).astype(complex).reshape(-1)
    rho = np.asarray(noisy_dm).astype(complex)
    val = float(np.real(np.vdot(psi, rho @ psi)))
    return float(np.clip(val, 0.0, 1.0))


def counts_fidelity(ideal_counts: Dict[str, int], noisy_counts: Dict[str, int]) -> float:
    """基于测量计数的保真度（经典保真度 / 命中率）。"""
    ideal_total = sum(ideal_counts.values())
    noisy_total = sum(noisy_counts.values())
    if ideal_total == 0 or noisy_total == 0:
        return 0.0
    overlap = 0.0
    for bitstring, cnt in noisy_counts.items():
        ideal_p = ideal_counts.get(bitstring, 0) / ideal_total
        noisy_p = cnt / noisy_total
        overlap += np.sqrt(ideal_p * noisy_p)
    return float(np.clip(overlap ** 2, 0.0, 1.0))


def circuit_fidelity_from_simulator(ideal_sv: np.ndarray, noisy_dm) -> float:
    """兼容 qiskit DensityMatrix 对象的包装。"""
    if hasattr(noisy_dm, "data"):
        noisy_dm = np.asarray(noisy_dm.data)
    return state_fidelity(ideal_sv, noisy_dm)


def coupling_distance(phys_q1: int, phys_q2: int, dist_matrix: np.ndarray) -> float:
    """两物理比特在耦合图中的最短路径距离。"""
    return float(dist_matrix[phys_q1, phys_q2])
