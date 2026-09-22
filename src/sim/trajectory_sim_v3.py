"""v3 事件级模拟器：v2 基础上把 swap 也作为动态串扰源。

物理依据：swap 由 3 个 CX 组成（0.9µs = 3×0.3µs），与并发 1-hop 门/swap 的
ZZ 动态串扰与 CX 同量级——v2 默认 `swap_xtalk=False`（历史口径，与旧 timing.py
A2 一致），v3 显式开启，向真实硬件靠近。

实现：完全复用 v2（config 浅拷贝 + 动态设 `swap_xtalk=True`，触发 v2 内部
`_reduce_phys_circuit_for_fidelity_v2` 的 `getattr(cfg, "swap_xtalk", ...)` 分支），
不改动 v2 的任何行为（既有基准不受影响）。

用法：`--fidelity-sim trajectory_v3`；评估脚本可经 `trajectory_circuit_fidelity_events_v3`
或 v3 版 fidelity_fn 使用。
"""
from dataclasses import replace

from .trajectory_sim_v2 import (
    trajectory_circuit_fidelity_events as _v2_fidelity_events,
    make_event_fidelity_fn as _v2_make_event_fidelity_fn,
)


def _swap_xtalk_config(config):
    """浅拷贝 config 并开启 swap 动态串扰（NoiseConfig 非 frozen/slots，可动态加属性）。"""
    cfg3 = replace(config)
    cfg3.swap_xtalk = True
    return cfg3


def trajectory_circuit_fidelity_events_v3(phys_circuit, config,
                                          num_trajectories: int = 16,
                                          seed=None, durations=None,
                                          backend: str = "auto"):
    """v3 单电路保真度：v2 口径 + swap 动态串扰（swap=3×CX 物理真实）。"""
    return _v2_fidelity_events(phys_circuit, _swap_xtalk_config(config),
                               num_trajectories=num_trajectories, seed=seed,
                               durations=durations, backend=backend)


def make_event_fidelity_fn_v3(config, num_trajectories: int = 16, seed=None,
                              durations=None, backend: str = "auto"):
    """v3 env fidelity_fn hook（训练/评估用，swap 动态串扰开启）。"""
    return _v2_make_event_fidelity_fn(_swap_xtalk_config(config),
                                      num_trajectories=num_trajectories,
                                      seed=seed, durations=durations,
                                      backend=backend)
