# ============================================================================
# test_predictor.py
# 验证 Multi-GNN 预测器前向传播形状与训练一步不报错。
# ============================================================================

import torch

from routing.graph.circuit_dag import CircuitDAG, build_routing_graph
from routing.graph.features import HardwareFeatures
from routing.gnn.predictor import MultiGNNTidelityPredictor, fidelity_loss


def _sample(n=4, depth=4, seed=0):
    from sim.sim import NoiseConfig
    from utils.data_gen import random_circuit
    coupling = [(i, i + 1) for i in range(n - 1)]
    config = NoiseConfig(
        t1_times=[50.0] * n, t2_times=[70.0] * n, freq_ghz=[5.0] * n,
        single_q_gate_error=0.001, two_q_gate_error=0.01,
        coupling_map=coupling, readout_error=[0.02] * n,
    )
    qc = random_circuit(n, depth, seed=seed)
    dag = CircuitDAG.from_circuit(qc)
    hw = HardwareFeatures.from_noise_config(config)
    data = build_routing_graph(dag, list(range(n)), hw, coupling).to_pyg()
    return data


def test_forward_shape():
    model = MultiGNNTidelityPredictor()
    data = _sample()
    out = model(data)
    assert out.shape == (1, 1 + 4)  # 总保真度 + 4 分项


def test_predict_fidelity_range():
    model = MultiGNNTidelityPredictor()
    model.eval()
    data = _sample()
    with torch.no_grad():
        fid = model.predict_fidelity(data)
    assert 0.0 <= float(fid.item()) <= 1.0


def test_training_step():
    model = MultiGNNTidelityPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = _sample()
    target = torch.tensor([0.9])
    opt.zero_grad()
    pred = model(data)
    loss = fidelity_loss(pred, target)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()
