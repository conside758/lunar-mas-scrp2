"""变规模/动态重规划场景的 BC 训练 + 对比（RoleAware 注意力 vs 普通 FlatActor）。

在可变资源场景（每回合随机采样 min~max 个资源，资源耗尽后动态重生）里收集规则示范，
分别用 VarRoleAwareActor（掩码注意力）与普通 FlatActor（展平定长）做行为克隆，再在同一组
评估种子下对比交付量——验证「注意力在变规模场景是否优于展平」。
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from lunar_rl.networks import ROBOT_NAMES, Actor, VarRoleAwareActor
from lunar_rl.obs import encode_obs_vars, flat_vars, flat_vars_dim
from lunar_rl.registry import ROBOT_ROLE, ROLE_TOOL_DIM
from lunar_rl.rule_policy import rule_action
from lunar_rl.runlog import plot_bars, run_dir, write_summary
from lunar_rl.surrogate import SurrogateEnv

MAX_RES = 8
MAX_TEAM = 5
MAX_DEPOT = 2


# ---------- 数据收集 ----------
def collect_var_data(env, episodes):
    data = []
    for _ in range(episodes):
        obs, _ = env.reset()
        while True:
            action = rule_action(obs)
            enc = encode_obs_vars(obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
            item = {
                "fixed": np.stack([enc[n]["fixed"] for n in ROBOT_NAMES]).astype(np.float32),
                "res_ent": np.stack([enc[n]["res_ent"] for n in ROBOT_NAMES]).astype(np.float32),
                "depot_ent": np.stack([enc[n]["depot_ent"] for n in ROBOT_NAMES]).astype(np.float32),
                "team_ent": np.stack([enc[n]["team_ent"] for n in ROBOT_NAMES]).astype(np.float32),
                "res_mask": np.stack([enc[n]["res_mask"] for n in ROBOT_NAMES]).astype(np.float32),
                "depot_mask": np.stack([enc[n]["depot_mask"] for n in ROBOT_NAMES]).astype(np.float32),
                "team_mask": np.stack([enc[n]["team_mask"] for n in ROBOT_NAMES]).astype(np.float32),
                "flat": np.stack([flat_vars(enc[n]) for n in ROBOT_NAMES]).astype(np.float32),
                "drive": np.array([action[n]["drive"] for n in ROBOT_NAMES], dtype=np.float32),
                "tools": [action[n].get("tool", 0) for n in ROBOT_NAMES],
            }
            data.append(item)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    return data


# ---------- BC 训练 ----------
def _bc_step_var(actors, batch_items, opt, mse, ce, device):
    """VarRoleAwareActor 的一步 BC。"""
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
        dm, _, tl = actors[name](fixed, res_ent, team_ent, res_mask, team_mask, depot_ent, depot_mask)
        loss = loss + mse(dm, drive_t)
        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
        if k > 0:
            tools_t = torch.tensor([d["tools"][j] for d in batch_items],
                                   dtype=torch.long, device=device)
            loss = loss + ce(tl[:, :k], tools_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def _bc_step_flat(actors, batch_items, opt, mse, ce, device):
    """普通 FlatActor 的一步 BC。"""
    loss = 0.0
    for j, name in enumerate(ROBOT_NAMES):
        flat = torch.tensor(np.stack([d["flat"][j] for d in batch_items]),
                            dtype=torch.float32, device=device)
        drive_t = torch.tensor(np.stack([d["drive"][j] for d in batch_items]),
                               dtype=torch.float32, device=device)
        dm, _, tl = actors[name](flat)
        loss = loss + mse(dm, drive_t)
        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
        if k > 0:
            tools_t = torch.tensor([d["tools"][j] for d in batch_items],
                                   dtype=torch.long, device=device)
            loss = loss + ce(tl[:, :k], tools_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train_bc_var(actors, data, epochs=20, lr=1e-3, batch=256, device="cpu", var=True):
    params = [p for a in actors.values() for p in a.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    n = len(data)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            items = [data[k] for k in b]
            if var:
                total += _bc_step_var(actors, items, opt, mse, ce, device)
            else:
                total += _bc_step_flat(actors, items, opt, mse, ce, device)
        if (ep + 1) % 5 == 0:
            print(f"[bc] epoch={ep+1} loss={total/max(1, n//batch):.4f}", flush=True)
    return actors


# ---------- 确定性动作 / 评估 ----------
def _det_action(actors, obs, var):
    enc = encode_obs_vars(obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
    action = {}
    with torch.no_grad():
        for j, name in enumerate(ROBOT_NAMES):
            if var:
                fixed = torch.tensor(enc[name]["fixed"][None], dtype=torch.float32)
                res_ent = torch.tensor(enc[name]["res_ent"][None], dtype=torch.float32)
                depot_ent = torch.tensor(enc[name]["depot_ent"][None], dtype=torch.float32)
                team_ent = torch.tensor(enc[name]["team_ent"][None], dtype=torch.float32)
                res_mask = torch.tensor(enc[name]["res_mask"][None], dtype=torch.float32)
                depot_mask = torch.tensor(enc[name]["depot_mask"][None], dtype=torch.float32)
                team_mask = torch.tensor(enc[name]["team_mask"][None], dtype=torch.float32)
                dm, _, tl = actors[name](fixed, res_ent, team_ent, res_mask, team_mask,
                                         depot_ent, depot_mask)
            else:
                flat = torch.tensor(flat_vars(enc[name])[None], dtype=torch.float32)
                dm, _, tl = actors[name](flat)
            sub = {"drive": torch.clamp(dm[0], -1.0, 1.0).numpy().astype(np.float32)}
            k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
            if k > 0:
                sub["tool"] = int(torch.argmax(tl[0, :k]).item())
            action[name] = sub
    return action


def evaluate(env, actors, var, episodes, seed0):
    totals = []
    for e in range(episodes):
        np.random.seed(seed0 + e)
        obs, _ = env.reset()
        while True:
            action = _det_action(actors, obs, var)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        totals.append(info["delivered_value"])
    return float(np.mean(totals)), totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--bc-episodes", type=int, default=20)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eval-seed0", type=int, default=1000)
    ap.add_argument("--out-prefix", type=str, default="/home/admina/MAS_ws/.tmp/vars")
    ap.add_argument("--run-name", type=str, default=None,
                    help="结果目录里本次运行的文件前缀（默认 run；结果存 results/<当天日期>/）")
    ap.add_argument("--app-weight", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = SurrogateEnv(args.scenario, stage="full")
    if args.app_weight is not None:
        env.w["app"] = args.app_weight

    print(f"[collect] rule data episodes={args.bc_episodes}...", flush=True)
    data = collect_var_data(env, args.bc_episodes)
    print(f"[collect] {len(data)} samples", flush=True)

    # 两种 actor 用同一份数据训练
    var_actors = {n: VarRoleAwareActor(fixed_dim=13, hidden=128) for n in ROBOT_NAMES}
    flat_actors = {n: Actor(obs_dim=flat_vars_dim(MAX_RES, MAX_TEAM), hidden=128)
                   for n in ROBOT_NAMES}
    print("[bc] training VarRoleAwareActor...", flush=True)
    train_bc_var(var_actors, data, epochs=args.bc_epochs, lr=args.bc_lr, var=True)
    print("[bc] training FlatActor...", flush=True)
    train_bc_var(flat_actors, data, epochs=args.bc_epochs, lr=args.bc_lr, var=False)

    # 同一组评估种子对比
    d_var, tv = evaluate(env, var_actors, var=True, episodes=args.eval_episodes,
                         seed0=args.eval_seed0)
    d_flat, tf = evaluate(env, flat_actors, var=False, episodes=args.eval_episodes,
                          seed0=args.eval_seed0)
    print(f"[eval] VarRoleAwareActor delivered={d_var:.1f}  per-episode={tv}", flush=True)
    print(f"[eval] FlatActor         delivered={d_flat:.1f}  per-episode={tf}", flush=True)

    torch.save({"actors": {k: v.state_dict() for k, v in var_actors.items()}},
               args.out_prefix + "_var.pt")
    torch.save({"actors": {k: v.state_dict() for k, v in flat_actors.items()}},
               args.out_prefix + "_flat.pt")
    print(f"saved {args.out_prefix}_var.pt / {args.out_prefix}_flat.pt")

    # 结果目录：对比柱状图 + checkpoint 副本 + 摘要
    rdir, prefix = run_dir(args.run_name)
    plot_bars(["VarRoleAware", "FlatActor"], [d_var, d_flat],
              os.path.join(rdir, prefix + "_curves.png"),
              title=f"变规模 BC 对比 seed={args.seed} (bc_eps={args.bc_episodes})")
    torch.save({"actors": {k: v.state_dict() for k, v in var_actors.items()}},
               os.path.join(rdir, prefix + "_var.pt"))
    torch.save({"actors": {k: v.state_dict() for k, v in flat_actors.items()}},
               os.path.join(rdir, prefix + "_flat.pt"))
    write_summary(os.path.join(rdir, prefix + ".txt"),
                  f"scenario={args.scenario}\nseed={args.seed} bc_episodes={args.bc_episodes} "
                  f"bc_epochs={args.bc_epochs}\n"
                  f"VarRoleAware delivered={d_var:.1f}\nFlatActor delivered={d_flat:.1f}\n")
    print(f"saved artifacts to {rdir} (prefix={prefix})")


if __name__ == "__main__":
    main()
