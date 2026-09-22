"""AWAC（优势加权回归）离线精修：用 HAPPO 反事实优势 + 指数加权 BC，替代 PPO ratio-clip。

思路：BC 预热后，用 BC 策略做**随机 rollout 收集多样示范数据**，拟合集中 QCritic Q(s,a)；
再用逐 Agent 反事实优势 A_i = Q(s,a) − Q(s,a_cf_i) 做 **exp(A_i/λ) 加权回归**：
    L_i = −E[ exp(A_i/λ) · log π_i(a_i|s) ]
使策略在不离开示范分布的前提下向高优势动作靠拢，缓解 PPO 精修「洗掉 BC」。
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from lunar_rl.mappo_vars import _actor_forward, _counterfactual, encode_agents, FLAT_DIM
from lunar_rl.networks import (ROBOT_NAMES, ROBOT_ROLE, ROLE_TOOL_DIM, Actor, QCritic,
                               VarRoleAwareActor, actions_to_dict, encode_joint_action)
from lunar_rl.registry import joint_action_dim
from lunar_rl.runlog import plot_curves, run_dir, write_summary
from lunar_rl.surrogate import SurrogateEnv


def build_actors(var, std_init=-0.7):
    if var:
        return {n: VarRoleAwareActor(fixed_dim=13, hidden=128, drive_logstd_init=std_init)
                for n in ROBOT_NAMES}
    return {n: Actor(obs_dim=FLAT_DIM, hidden=128, drive_logstd_init=std_init)
            for n in ROBOT_NAMES}


def sample_actions(actors, enc_batch, var):
    """从当前策略采样（stochastic，用于收集多样数据）。enc_batch 来自 encode_agents。"""
    fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask, flat, _ = enc_batch
    f = torch.tensor(fixed[None], dtype=torch.float32)
    re = torch.tensor(res_ent[None], dtype=torch.float32)
    de = torch.tensor(depot_ent[None], dtype=torch.float32)
    te = torch.tensor(team_ent[None], dtype=torch.float32)
    rm = torch.tensor(res_mask[None], dtype=torch.float32)
    dm = torch.tensor(depot_mask[None], dtype=torch.float32)
    tm = torch.tensor(team_mask[None], dtype=torch.float32)
    fl = torch.tensor(flat[None], dtype=torch.float32)
    drives, tools, logps = [], [], []
    with torch.no_grad():
        for j, name in enumerate(ROBOT_NAMES):
            dmu, ds, tl = _actor_forward(actors[name], f[:, j], re[:, j], de[:, j], te[:, j],
                                         rm[:, j], dm[:, j], tm[:, j], fl[:, j], var)
            dist = Normal(dmu, ds)
            u = dist.sample()
            drive = torch.clamp(u, -1.0, 1.0)
            lp = dist.log_prob(u).sum(-1)
            tool = torch.zeros(1, dtype=torch.long)
            k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
            if k > 0:
                d = Categorical(logits=tl[:, :k])
                tool = d.sample()
                lp = lp + d.log_prob(tool)
            drives.append(drive); tools.append(tool); logps.append(lp)
    drive_t = torch.stack(drives, dim=1)  # (1, N, 2)
    tools_l = [int(t[0]) for t in tools]
    return actions_to_dict(drive_t[0], tools_l), drive_t[0].numpy(), tools_l


def collect_data(env, actors, var, episodes):
    data = []
    for _ in range(episodes):
        obs, _ = env.reset()
        traj = []
        while True:
            enc = encode_agents(obs)
            action, drive, tools = sample_actions(actors, enc, var)
            obs2, reward, term, trunc, info = env.step(action)
            enc2 = encode_agents(obs2)
            traj.append({"enc": enc, "drive": drive, "tools": tools,
                         "reward": float(reward), "enc2": enc2, "done": bool(term or trunc)})
            obs = obs2
            if term or trunc:
                break
        # 计算 MC 回报（沿轨迹反向）
        G = 0.0
        for t in reversed(traj):
            G = t["reward"] + 0.9 * G * (1 - int(t["done"]))
            t["G"] = G
        data += traj
    return data


def fit_q(qcritic, data, gamma, epochs, lr, batch, device):
    opt = torch.optim.Adam(qcritic.parameters(), lr=lr)
    mse = nn.MSELoss()
    g = torch.tensor(np.stack([d["enc"][-1] for d in data]), dtype=torch.float32, device=device)
    ja = torch.stack([encode_joint_action(torch.tensor(d["drive"], dtype=torch.float32, device=device),
                                          d["tools"]) for d in data])
    y = torch.tensor(np.array([d["G"] for d in data], dtype=np.float32), device=device)
    n = len(data)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            pred = qcritic(g[b], ja[b])
            loss = mse(pred, y[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(qcritic.parameters(), 0.5)
            opt.step(); total += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"[fit_q] epoch={ep+1} loss={total/max(1, n//batch):.4f}", flush=True)
    return qcritic


def awac_update(actors, qcritic, data, lam, lr, batch, epochs, device, var, kl_coef=0.0,
                ref_actors=None):
    actor_opts = {n: torch.optim.Adam(a.parameters(), lr=lr) for n, a in actors.items()}
    n = len(data)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            items = [data[k] for k in b]
            g = torch.tensor(np.stack([it["enc"][-1] for it in items]), dtype=torch.float32,
                             device=device)
            ja = torch.stack([encode_joint_action(torch.tensor(it["drive"], dtype=torch.float32,
                                                               device=device), it["tools"])
                              for it in items])
            with torch.no_grad():
                q_actual = qcritic(g, ja)
            for j, name in enumerate(ROBOT_NAMES):
                fixed = torch.tensor(np.stack([it["enc"][0][j] for it in items]),
                                     dtype=torch.float32, device=device)
                res_ent = torch.tensor(np.stack([it["enc"][1][j] for it in items]),
                                       dtype=torch.float32, device=device)
                depot_ent = torch.tensor(np.stack([it["enc"][2][j] for it in items]),
                                         dtype=torch.float32, device=device)
                team_ent = torch.tensor(np.stack([it["enc"][3][j] for it in items]),
                                        dtype=torch.float32, device=device)
                res_mask = torch.tensor(np.stack([it["enc"][4][j] for it in items]),
                                        dtype=torch.float32, device=device)
                depot_mask = torch.tensor(np.stack([it["enc"][5][j] for it in items]),
                                          dtype=torch.float32, device=device)
                team_mask = torch.tensor(np.stack([it["enc"][6][j] for it in items]),
                                         dtype=torch.float32, device=device)
                flat = torch.tensor(np.stack([it["enc"][7][j] for it in items]),
                                    dtype=torch.float32, device=device)
                drive_t = torch.tensor(np.stack([it["drive"][j] for it in items]),
                                       dtype=torch.float32, device=device)
                dm, ds, tl = _actor_forward(actors[name], fixed, res_ent, depot_ent, team_ent,
                                            res_mask, depot_mask, team_mask, flat, var)
                dist = Normal(dm, ds)
                new_lp = dist.log_prob(drive_t).sum(-1)
                k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
                if k > 0:
                    tools_t = torch.tensor([it["tools"][j] for it in items],
                                           dtype=torch.long, device=device)
                    new_lp = new_lp + Categorical(logits=tl[:, :k]).log_prob(tools_t)
                # 反事实优势
                with torch.no_grad():
                    advs = []
                    for ii, it in enumerate(items):
                        tool_idx = int(it["tools"][j]) if k > 0 else 0
                        a_cf = _counterfactual(ja[ii], j, dm[ii].detach(), tool_idx)
                        advs.append((q_actual[ii] - qcritic(g[ii:ii + 1], a_cf.unsqueeze(0))).detach())
                    adv = torch.stack(advs)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                w = torch.exp(adv / lam)
                loss = -(w * new_lp).mean()
                if ref_actors is not None and kl_coef > 0.0:
                    with torch.no_grad():
                        dm_ref, _, tl_ref = _actor_forward(ref_actors[name], fixed, res_ent,
                                                           depot_ent, team_ent, res_mask, depot_mask,
                                                           team_mask, flat, var)
                    kl = ((dm - dm_ref) ** 2).mean(-1)
                    if k > 0:
                        p = torch.softmax(tl[:, :k], dim=-1)
                        q = torch.softmax(tl_ref[:, :k], dim=-1)
                        kl = kl + (p * (p.log() - q.log())).sum(-1)
                    loss = loss + kl_coef * kl.mean()
                actor_opts[name].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actors[name].parameters(), 0.5)
                actor_opts[name].step()
                total += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"[awac] epoch={ep+1} loss={total/max(1, n//batch):.4f}", flush=True)
    return actors


def evaluate(env, actors, var, episodes=10, seed0=1000):
    from lunar_rl.train_vars import _det_action
    vals = []
    for e in range(episodes):
        np.random.seed(seed0 + e)
        obs, _ = env.reset()
        while True:
            obs, _, term, trunc, info = env.step(_det_action(actors, obs, var))
            if term or trunc:
                break
        vals.append(info["delivered_value"])
    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--actor-type", type=str, default="var", choices=["var", "flat"])
    ap.add_argument("--load-actor", type=str, required=True)
    ap.add_argument("--collect-episodes", type=int, default=20)
    ap.add_argument("--awac-epochs", type=int, default=30)
    ap.add_argument("--awac-lam", type=float, default=1.0)
    ap.add_argument("--awac-lr", type=float, default=1e-4)
    ap.add_argument("--q-epochs", type=int, default=30)
    ap.add_argument("--q-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--out", type=str, default="/home/admina/MAS_ws/.tmp/awac.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    var = args.actor_type == "var"
    env = SurrogateEnv(args.scenario, stage="full")
    env.w["app"] = 0.3

    actors = build_actors(var)
    ck = torch.load(args.load_actor, map_location="cpu")
    for n in ROBOT_NAMES:
        actors[n].load_state_dict(ck["actors"][n])
    ref_actors = {k: copy.deepcopy(v) for k, v in actors.items()} if args.kl_coef > 0 else None

    # 1. 多样示范数据（BC 策略随机 rollout）
    print(f"[collect] {args.collect_episodes} episodes (stochastic BC rollout)...", flush=True)
    data = collect_data(env, actors, var, args.collect_episodes)
    print(f"[collect] {len(data)} transitions", flush=True)

    gdim = len(ROBOT_NAMES) * FLAT_DIM
    qcritic = QCritic(global_dim=gdim, action_dim=joint_action_dim(), hidden=128)

    # 2. 拟合 Q
    print("[fit_q] ...", flush=True)
    fit_q(qcritic, data, args.gamma, args.q_epochs, args.q_lr, args.batch, "cpu")

    # 3. AWAC 优势加权回归
    t0 = time.time()
    print("[awac] ...", flush=True)
    awac_update(actors, qcritic, data, args.awac_lam, args.awac_lr, args.batch, args.awac_epochs,
                "cpu", var, kl_coef=args.kl_coef, ref_actors=ref_actors)
    print(f"[awac] done ({time.time()-t0:.1f}s)", flush=True)

    # 4. 评估
    d, per = evaluate(env, actors, var, episodes=args.eval_episodes)
    print(f"[final] delivered_value={d:.1f} per-episode={per}", flush=True)

    ckpt = {"actors": {k: v.state_dict() for k, v in actors.items()},
            "qcritic": qcritic.state_dict()}
    torch.save(ckpt, args.out)
    rdir, prefix = run_dir(args.run_name)
    torch.save(ckpt, os.path.join(rdir, prefix + ".pt"))
    write_summary(os.path.join(rdir, prefix + ".txt"),
                  f"scenario={args.scenario}\nload_actor={args.load_actor}\n"
                  f"collect_episodes={args.collect_episodes} awac_epochs={args.awac_epochs} "
                  f"lam={args.awac_lam} q_epochs={args.q_epochs}\nfinal_value={d:.1f}\n")
    print(f"saved {args.out} and artifacts to {rdir} (prefix={prefix})")


if __name__ == "__main__":
    main()
