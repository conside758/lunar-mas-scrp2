"""用 Gazebo 收集的 6 机器人示范数据（pickle）训练 VarRoleAwareActor BC。

数据格式：{"names": [...], "roles": {...}, "samples": [encode_item(...), ...]}（由
lunar_env.collect_gazebo_data / collect_dagger_gazebo 产出）。BC 步骤与 `train_vars.py`
的 `_bc_step_var` 完全一致（drive=MSE，tool=CE，逐机器人），避免与替身 BC 训练逻辑发散。

用法：python3 -m lunar_rl.train_gazebo_bc_vars <data.pkl> <out.pt> [--epochs 30] [--lr 1e-3] [--batch 256]
"""
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn

from lunar_rl.networks import VarRoleAwareActor
from lunar_rl.obs import FIXED_DIM
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE, ROLE_TOOL_DIM, set_robots


def compute_tool_weights(data, mode="sqrt"):
    """按逆频率给每个机器人 tool 类别加权（缓解 idle 类占绝对多数导致的类不平衡）。

    mode: 'balanced'=完整逆频率（可能 80x 过猛、伤共享 drive 特征）；'sqrt'=开方软化（默认，温和）。
    返回 {name: torch.Tensor(weight,)}，仅对 tool 维度 >1 的机器人（excavator=3 类 / hauler=2 类）。
    """
    weights = {}
    for j, name in enumerate(ROBOT_NAMES):
        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
        if k <= 1:
            continue
        counts = np.zeros(k, dtype=np.float64)
        for d in data:
            counts[int(d["tools"][j])] += 1.0
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (k * counts)  # 逆频率
        if mode == "sqrt":
            w = np.sqrt(w)
        weights[name] = torch.tensor(w, dtype=torch.float32)
    return weights


def _bc_step_var(actors, batch_items, opt, mse, ce_by_name, device):
    """VarRoleAwareActor 的一步 BC（drive=MSE，tool=类加权 CE，逐机器人）。"""
    loss = 0.0
    for j, name in enumerate(ROBOT_NAMES):
        fixed = torch.tensor(np.stack([d["fixed"][j] for d in batch_items]),
                             dtype=torch.float32, device=device)
        res_ent = torch.tensor(np.stack([d["res_ent"][j] for d in batch_items]),
                               dtype=torch.float32, device=device)
        depot_ent = torch.tensor(np.stack([d["depot_ent"][j] for d in batch_items]),
                                 dtype=torch.float32, device=device)
        team_ent = torch.tensor(np.stack([d["team_ent"][j] for d in batch_items]),
                                dtype=torch.float32, device=device)
        res_mask = torch.tensor(np.stack([d["res_mask"][j] for d in batch_items]),
                                dtype=torch.float32, device=device)
        depot_mask = torch.tensor(np.stack([d["depot_mask"][j] for d in batch_items]),
                                  dtype=torch.float32, device=device)
        team_mask = torch.tensor(np.stack([d["team_mask"][j] for d in batch_items]),
                                 dtype=torch.float32, device=device)
        drive_t = torch.tensor(np.stack([d["drive"][j] for d in batch_items]),
                               dtype=torch.float32, device=device)
        dm, _, tl = actors[name](fixed, res_ent, team_ent, res_mask, team_mask,
                                 depot_ent, depot_mask)
        loss = loss + mse(dm, drive_t)
        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
        if k > 0:
            tools_t = torch.tensor([d["tools"][j] for d in batch_items],
                                   dtype=torch.long, device=device)
            loss = loss + ce_by_name[name](tl[:, :k], tools_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train_bc_var(actors, data, epochs=30, lr=1e-3, batch=256, device="cpu",
                 tool_weight="sqrt"):
    params = [p for a in actors.values() for p in a.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    if tool_weight in ("balanced", "sqrt"):
        tw = compute_tool_weights(data, mode=tool_weight)
        ce_by_name = {n: nn.CrossEntropyLoss(weight=w) for n, w in tw.items()}
        # 无工具机器人占位（不会被调用）
        for n in ROBOT_NAMES:
            if ROLE_TOOL_DIM[ROBOT_ROLE[n]] > 0 and n not in ce_by_name:
                ce_by_name[n] = nn.CrossEntropyLoss()
        for n, w in tw.items():
            print(f"[bc] tool class weights {n} = {w.numpy().round(3).tolist()}", flush=True)
    else:
        ce_by_name = {n: nn.CrossEntropyLoss() for n in ROBOT_NAMES}
    n = len(data)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        nbatch = 0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            total += _bc_step_var(actors, [data[k] for k in b], opt, mse, ce_by_name, device)
            nbatch += 1
        if (ep + 1) % 5 == 0:
            print(f"[bc] epoch={ep+1} loss={total/max(1, nbatch):.4f}", flush=True)
    return actors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_pkl")
    ap.add_argument("out_pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tool-weight", type=str, default="sqrt",
                    choices=["none", "balanced", "sqrt"],
                    help="tool 类别加权：none=无 / balanced=完整逆频率 / sqrt=开方软化(默认)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data_pkl, "rb") as f:
        blob = pickle.load(f)
    names = blob["names"]
    roles = blob["roles"]
    samples = blob["samples"]
    set_robots(names, roles)
    print(f"[train_gazebo_bc_vars] loaded {len(samples)} samples, names={names}", flush=True)

    actors = {n: VarRoleAwareActor(fixed_dim=FIXED_DIM, hidden=128) for n in names}
    train_bc_var(actors, samples, epochs=args.epochs, lr=args.lr, batch=args.batch,
                 tool_weight=args.tool_weight)

    torch.save({"actors": {k: v.state_dict() for k, v in actors.items()}}, args.out_pt)
    print(f"[train_gazebo_bc_vars] saved checkpoint to {args.out_pt}", flush=True)


if __name__ == "__main__":
    main()
