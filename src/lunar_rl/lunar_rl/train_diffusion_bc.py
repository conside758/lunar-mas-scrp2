"""用 Gazebo 收集的示范数据训练 Diffusion Policy drive 头（替换高斯 MSE）。

数据格式同 collect_gazebo_data（encode_item）：每样本含 fixed/res_ent/depot_ent/team_ent/各 mask/drive/tools。
只训练 drive 的去噪扩散（DDPM），tool 部署时用规则状态机。

用法：python3 -m lunar_rl.train_diffusion_bc <data.pkl> <out.pt> [--epochs 100] [--lr 1e-3] [--batch 256]
"""
import argparse
import pickle

import numpy as np
import torch

from lunar_rl.diffusion_actor import DiffusionDriveActor
from lunar_rl.obs import FIXED_DIM
from lunar_rl.registry import ROBOT_NAMES, set_robots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_pkl")
    ap.add_argument("out_pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-ckpt", type=str, default=None,
                    help="从既有 checkpoint 初始化权重后微调（Gazebo 少量 rollout 微调用）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data_pkl, "rb") as f:
        blob = pickle.load(f)
    names = blob["names"]
    roles = blob["roles"]
    samples = blob["samples"]
    set_robots(names, roles)
    print(f"[train_diffusion_bc] loaded {len(samples)} samples, names={names}, T={args.T}",
          flush=True)

    actors = {n: DiffusionDriveActor(fixed_dim=FIXED_DIM, hidden=128, attn_dim=64,
                                     action_dim=2, T=args.T) for n in names}
    if args.init_ckpt:
        init = torch.load(args.init_ckpt, map_location="cpu")
        for n, a in actors.items():
            if n in init["actors"]:
                a.load_state_dict(init["actors"][n])
        print(f"[train_diffusion_bc] initialized weights from {args.init_ckpt}", flush=True)
    params = [p for a in actors.values() for p in a.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    n = len(samples)

    for ep in range(args.epochs):
        idx = np.random.permutation(n)
        total = 0.0
        nbatch = 0
        for i in range(0, n, args.batch):
            b = [samples[k] for k in idx[i:i + args.batch]]
            loss = 0.0
            for j, name in enumerate(ROBOT_NAMES):
                fixed = torch.tensor(np.stack([d["fixed"][j] for d in b]), dtype=torch.float32)
                res_ent = torch.tensor(np.stack([d["res_ent"][j] for d in b]), dtype=torch.float32)
                depot_ent = torch.tensor(np.stack([d["depot_ent"][j] for d in b]), dtype=torch.float32)
                team_ent = torch.tensor(np.stack([d["team_ent"][j] for d in b]), dtype=torch.float32)
                res_mask = torch.tensor(np.stack([d["res_mask"][j] for d in b]), dtype=torch.float32)
                depot_mask = torch.tensor(np.stack([d["depot_mask"][j] for d in b]), dtype=torch.float32)
                team_mask = torch.tensor(np.stack([d["team_mask"][j] for d in b]), dtype=torch.float32)
                x0 = torch.tensor(np.stack([d["drive"][j] for d in b]), dtype=torch.float32)
                cond = actors[name].encode(fixed, res_ent, team_ent, res_mask, team_mask,
                                           depot_ent, depot_mask)
                loss = loss + actors[name].diffusion_loss(cond, x0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)  # 稳定 DDPM 训练，防止 loss 爆炸
            opt.step()
            total += loss.item()
            nbatch += 1
        if (ep + 1) % 10 == 0:
            print(f"[diffusion] epoch={ep+1} loss={total/max(1, nbatch):.4f}", flush=True)

    torch.save({"actors": {k: v.state_dict() for k, v in actors.items()}}, args.out_pt)
    print(f"[train_diffusion_bc] saved checkpoint to {args.out_pt}", flush=True)


if __name__ == "__main__":
    main()
