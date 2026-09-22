"""把 LA287 起点扩宽为 E17 组合轮用的起点（edge_mlp 64→128 / 32→64，输入 158→267）。

方案：
- edge_mlp.0 [64,158]→[128,267]：旧权重放左上角，新列/新行 0；新 bias=0.01（微正，
  ReLU 活性保梯度可训；下游新列全 0 → 初始 logits 与原 LA287 逐位等价）
- edge_mlp.2 [32,64]→[64,128]、edge_mlp.4 [1,32]→[1,64]：同左上加宽
- edge_score/commit_head [1,158]→[1,267]：旧放前 158 列，新列 0
- critic_*.0 [64,180]→[64,289]：edge 段前 158 列 + map/progress/phase 段移到新位置
  （旧 158:180 → 新 267:289），中间新特征列 0
- gnn / rew_norm 原样
产物: policy_LA287wide.pt
"""
import numpy as np
import torch


def main():
    src = torch.load("models/policy_LA287.pt", map_location="cpu", weights_only=False)
    ac = src["ac"]
    old_ef = ac["edge_mlp.0.weight"].shape[1]  # 158
    new_ef = 267  # out*3 + sabre5 + look6 + noise5 + global101 + sabre_core6
    new_h1, new_h2 = 128, 64

    out = {}
    # edge_mlp.0: [h1, ef]
    w0 = torch.zeros(new_h1, new_ef)
    w0[:64, :old_ef] = ac["edge_mlp.0.weight"]
    b0 = torch.zeros(new_h1)
    b0[:64] = ac["edge_mlp.0.bias"]
    b0[64:] = 0.01  # 保 ReLU 活性
    out["edge_mlp.0.weight"], out["edge_mlp.0.bias"] = w0, b0
    # edge_mlp.2: [h2, h1]
    w2 = torch.zeros(new_h2, new_h1)
    w2[:32, :64] = ac["edge_mlp.2.weight"]
    b2 = torch.zeros(new_h2)
    b2[:32] = ac["edge_mlp.2.bias"]
    b2[32:] = 0.01
    out["edge_mlp.2.weight"], out["edge_mlp.2.bias"] = w2, b2
    # edge_mlp.4: [1, h2]
    w4 = torch.zeros(1, new_h2)
    w4[0, :32] = ac["edge_mlp.4.weight"]
    out["edge_mlp.4.weight"], out["edge_mlp.4.bias"] = w4, ac["edge_mlp.4.bias"]
    # edge_score / commit_head: [1, ef]
    for k in ("edge_score.weight", "commit_head.weight"):
        w = torch.zeros(1, new_ef)
        w[0, :old_ef] = ac[k]
        out[k] = w
    out["edge_score.bias"] = ac["edge_score.bias"]
    if "commit_head.bias" in ac:
        out["commit_head.bias"] = ac["commit_head.bias"]

    # critic_*.0: [64, ef + num_qubits + 2]  (180 → 289)
    nq = 20
    old_cin = old_ef + nq + 2
    new_cin = new_ef + nq + 2
    for prefix in ("critic_route", "critic_fid", "critic_la"):
        if f"{prefix}.0.weight" not in ac:
            continue
        w = torch.zeros(64, new_cin)
        w[:, :old_ef] = ac[f"{prefix}.0.weight"][:, :old_ef]           # edge 段
        w[:, new_ef:new_ef + (old_cin - old_ef)] = \
            ac[f"{prefix}.0.weight"][:, old_ef:]                        # map/prog/phase 段
        out[f"{prefix}.0.weight"] = w
        out[f"{prefix}.0.bias"] = ac[f"{prefix}.0.bias"]
        out[f"{prefix}.2.weight"] = ac[f"{prefix}.2.weight"]
        out[f"{prefix}.2.bias"] = ac[f"{prefix}.2.bias"]
        # critic_la 可能有额外结构（如 critic_la.2/3）
        for k in ac:
            if k.startswith(prefix + ".") and k not in out:
                out[k] = ac[k]

    new_ac = {k: v for k, v in ac.items() if k.startswith("critic") is False
              and k not in out}
    for k, v in out.items():
        new_ac[k] = v

    dst = {"ac": new_ac, "rew_norm": src["rew_norm"]}
    if "gnn" in src:
        dst["gnn"] = src["gnn"]
    torch.save(dst, "models/policy_LA287wide.pt")
    print(f"写入 models/policy_LA287wide.pt")
    print(f"edge_mlp.0: {tuple(ac['edge_mlp.0.weight'].shape)} -> "
          f"{tuple(new_ac['edge_mlp.0.weight'].shape)}")
    print(f"critic_route.0: {tuple(ac['critic_route.0.weight'].shape)} -> "
          f"{tuple(new_ac['critic_route.0.weight'].shape)}")


if __name__ == "__main__":
    main()
