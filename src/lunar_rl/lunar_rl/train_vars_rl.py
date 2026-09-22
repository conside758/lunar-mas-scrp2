"""变规模场景的 BC 预热 + HAPPO 反事实（或 MAPPO）RL 精修入口。

用法：python3 -m lunar_rl.train_vars_rl <scenario_var.yaml> --actor-type var|flat \
        --load-actor <bc.pt> [--hap] [--critic-warmup 20] ...
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

from lunar_rl.mappo_vars import MAPPOVars, FLAT_DIM
from lunar_rl.networks import (ROBOT_NAMES, Actor, Critic, QCritic,
                               VarRoleAwareActor)
from lunar_rl.registry import joint_action_dim
from lunar_rl.runlog import plot_curves, run_dir, write_summary
from lunar_rl.surrogate import SurrogateEnv


def build_actors(var, std_init=-0.7):
    if var:
        return {n: VarRoleAwareActor(fixed_dim=13, hidden=128, drive_logstd_init=std_init)
                for n in ROBOT_NAMES}
    return {n: Actor(obs_dim=FLAT_DIM, hidden=128, drive_logstd_init=std_init)
            for n in ROBOT_NAMES}


def evaluate(env, agent, episodes=10, seed0=1000):
    totals, vals, rewards = [], [], []
    for e in range(episodes):
        np.random.seed(seed0 + e)
        obs, _ = env.reset()
        while True:
            action = agent.act(obs)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        totals.append(info["delivered"])
        vals.append(info["delivered_value"])
        rewards.append(info["episode_reward"])
    return float(np.mean(vals)), vals, float(np.mean(rewards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--actor-type", type=str, default="var", choices=["var", "flat"])
    ap.add_argument("--load-actor", type=str, default=None)
    ap.add_argument("--hap", action="store_true")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--critic-warmup", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--cql-coef", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--app-weight", type=float, default=None)
    ap.add_argument("--std-init", type=float, default=-0.7)
    ap.add_argument("--reset-std", action="store_true")
    ap.add_argument("--run-name", type=str, default=None,
                    help="结果目录里本次运行的文件前缀（默认 run；结果存 results/<当天日期>/）")
    ap.add_argument("--out", type=str, default="/home/admina/MAS_ws/.tmp/vars_rl.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    var = args.actor_type == "var"
    env = SurrogateEnv(args.scenario, stage="full")
    if args.app_weight is not None:
        env.w["app"] = args.app_weight
        print(f"[reward] app weight overridden to {args.app_weight}", flush=True)

    actors = build_actors(var, std_init=args.std_init)
    if args.load_actor:
        ckpt = torch.load(args.load_actor, map_location="cpu")
        for n in ROBOT_NAMES:
            actors[n].load_state_dict(ckpt["actors"][n])
        if args.reset_std:
            for n in ROBOT_NAMES:
                actors[n].drive_logstd.data.fill_(args.std_init)
            print(f"[load] {args.load_actor} (reset std -> {args.std_init})", flush=True)
        else:
            print(f"[load] {args.load_actor} (keep ckpt std)", flush=True)

    gdim = len(ROBOT_NAMES) * FLAT_DIM
    critic = Critic(global_dim=gdim, hidden=128)
    qcritic = QCritic(global_dim=gdim, action_dim=joint_action_dim(), hidden=128) if args.hap else None
    ref_actors = {k: copy.deepcopy(v) for k, v in actors.items()} if args.kl_coef > 0 else None
    agent = MAPPOVars(actors, critic, lr=args.lr, gamma=args.gamma,
                      ref_actors=ref_actors, kl_coef=args.kl_coef, hap=args.hap,
                      qcritic=qcritic, critic_lr=args.critic_lr, var=var,
                      cql_coef=args.cql_coef)

    if args.critic_warmup > 0:
        agent.warmup_critic(env, episodes=args.critic_warmup, epochs=30)

    t0 = time.time()
    metrics = {"delivered": [], "reward": [], "value_loss": [], "entropy": []}
    for it in range(args.iters):
        buf = agent.rollout(env, args.steps)
        stats = agent.update(buf)
        metrics["value_loss"].append((it, stats["value_loss"]))
        metrics["entropy"].append((it, stats["entropy"]))
        if it % args.eval_every == 0 or it == args.iters - 1:
            delivered, _, reward = evaluate(env, agent, episodes=3)
            metrics["delivered"].append((it, delivered))
            metrics["reward"].append((it, reward))
            print(f"[iter {it:4d}] policy_loss={stats['policy_loss']:.4f} "
                  f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.3f} "
                  f"delivered={delivered:.1f} ({time.time()-t0:.1f}s)", flush=True)

    ckpt = {"actors": {k: v.state_dict() for k, v in actors.items()},
            "critic": critic.state_dict(),
            "qcritic": qcritic.state_dict() if qcritic is not None else None}
    torch.save(ckpt, args.out)
    # 最终用同一组种子评估
    d, per, _ = evaluate(env, agent, episodes=args.eval_episodes)
    print(f"[final] delivered={d:.1f} per-episode={per}", flush=True)

    rdir, prefix = run_dir(args.run_name)
    plot_curves(metrics, os.path.join(rdir, prefix + "_curves.png"),
                title=f"Var-scale RL seed={args.seed} (actor={args.actor_type}, hap={args.hap}, cql={args.cql_coef})")
    torch.save(ckpt, os.path.join(rdir, prefix + ".pt"))
    write_summary(os.path.join(rdir, prefix + ".txt"),
                  f"scenario={args.scenario}\nseed={args.seed} actor_type={args.actor_type} hap={args.hap}\n"
                  f"iters={args.iters} steps={args.steps} lr={args.lr} "
                  f"critic_lr={args.critic_lr} cql_coef={args.cql_coef}\n"
                  f"final_delivered={d:.1f}\n")
    print(f"saved checkpoint to {args.out}")
    print(f"saved artifacts to {rdir} (prefix={prefix})")


if __name__ == "__main__":
    main()
