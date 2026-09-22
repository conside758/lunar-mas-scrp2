"""用 Gazebo 收集的示范数据（pickle）训练 BC，保存 3-actor checkpoint。

用法：python3 -m lunar_rl.train_gazebo_bc <data.pkl> <out.pt> [--epochs 30] [--lr 1e-3]
"""
import argparse
import pickle

import numpy as np
import torch

from lunar_rl.bc import train_bc
from lunar_rl.networks import Actor, Critic, ROBOT_NAMES
from lunar_rl.obs import obs_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_pkl")
    parser.add_argument("out_pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data_pkl, "rb") as f:
        data = pickle.load(f)
    print(f"[train_gazebo_bc] loaded {len(data)} samples", flush=True)

    odim = obs_dim()
    actors = {n: Actor(obs_dim=odim, hidden=128) for n in ROBOT_NAMES}
    critic = Critic(global_dim=3 * odim, hidden=128)
    train_bc(actors, data, epochs=args.epochs, lr=args.lr)

    torch.save({"actors": {k: v.state_dict() for k, v in actors.items()},
                "critic": critic.state_dict()}, args.out_pt)
    print(f"[train_gazebo_bc] saved checkpoint to {args.out_pt}", flush=True)


if __name__ == "__main__":
    main()
