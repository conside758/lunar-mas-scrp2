"""HASAC 训练入口（在快速替身环境上离线训练）。

用法：python3 -m lunar_rl.train_hasac <scenario_file> [--iters N] [--steps S]
与 train.py 的区别：HASAC 是离线（回放缓冲）+ 逐 Agent 软 Q + 熵正则 + 逐角色奖励 shaping。
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

from lunar_rl.bc import collect_rule_data, dagger, train_bc
from lunar_rl.hasac import HASAC
from lunar_rl.networks import (AGENT_ACTION_DIM, ROBOT_NAMES, Actor, QCritic,
                               RoleAwareActor)
from lunar_rl.obs import obs_dim
from lunar_rl.runlog import plot_curves, run_dir, write_summary
from lunar_rl.surrogate import SurrogateEnv


def evaluate(env, agent, episodes=3):
    totals, rewards = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        while True:
            action = agent.act(obs)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        totals.append(info["delivered"])
        rewards.append(info["episode_reward"])
    return float(np.mean(totals)), float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="训练前先收集的回放缓冲步数")
    parser.add_argument("--rlpd-steps", type=int, default=0,
                        help="RLPD 式预填：用规则专家示范 transitions 预填回放缓冲的步数（0=关闭）")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-iter", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-episodes", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="熵正则温度（SAC）")
    parser.add_argument("--clip-eps", type=float, default=0.0,
                        help="镜像学习信任域 clip 系数（>0 用 soft PPO clip ratio 替代重参数化 SAC）")
    parser.add_argument("--stage", type=str, default="full")
    parser.add_argument("--load-actor", type=str, default=None)
    parser.add_argument("--reset-std", action="store_true",
                        help="加载 checkpoint 后把 drive_logstd 重置为 --std-init")
    parser.add_argument("--std-init", type=float, default=-0.7)
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help="对 BC 参考策略的 KL 正则系数（>0 防漂移）")
    parser.add_argument("--app-weight", type=float, default=None)
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--role-aware", action="store_true")
    parser.add_argument("--run-name", type=str, default=None,
                        help="结果目录里本次运行的文件前缀（默认 run；结果存 results/<当天日期>/）")
    parser.add_argument("--out", type=str, default="/home/admina/MAS_ws/.tmp/hasac.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = SurrogateEnv(args.scenario, stage=args.stage, domain_rand=args.domain_rand)
    if args.app_weight is not None:
        env.w["app"] = args.app_weight
        print(f"[reward] app weight overridden to {args.app_weight}", flush=True)
    odim = obs_dim()
    if args.role_aware:
        actors = {name: RoleAwareActor(obs_dim=odim, hidden=128,
                                       drive_logstd_init=args.std_init)
                  for name in ROBOT_NAMES}
    else:
        actors = {name: Actor(obs_dim=odim, hidden=128, drive_logstd_init=args.std_init)
                  for name in ROBOT_NAMES}

    if args.load_actor:
        ckpt = torch.load(args.load_actor, map_location="cpu")
        for name in ROBOT_NAMES:
            actors[name].load_state_dict(ckpt["actors"][name])
        if args.reset_std:
            for name in ROBOT_NAMES:
                actors[name].drive_logstd.data.fill_(args.std_init)
            print(f"[load] warm-started from {args.load_actor} "
                  f"(reset drive_logstd -> {args.std_init})", flush=True)
        else:
            print(f"[load] warm-started from {args.load_actor} (keep ckpt std)", flush=True)

    if args.bc_episodes > 0:
        if args.dagger_rounds > 0:
            print(f"[dagger] init={args.bc_episodes} rounds={args.dagger_rounds} "
                  f"(stage={args.stage})...", flush=True)
            dagger(actors, env, rounds=args.dagger_rounds,
                   episodes=args.bc_episodes, epochs=args.bc_epochs,
                   init_episodes=args.bc_episodes)
        else:
            print(f"[bc] collecting {args.bc_episodes} rule episodes...", flush=True)
            data = collect_rule_data(env, args.bc_episodes)
            train_bc(actors, data, epochs=args.bc_epochs)

    # 逐 Agent 软 Q Critic：输入全局状态(96) + 自身动作(drive 2 + tool one-hot)
    qcritics = {name: QCritic(global_dim=3 * odim, action_dim=AGENT_ACTION_DIM[name],
                              hidden=128)
                for name in ROBOT_NAMES}
    ref_actors = {k: copy.deepcopy(v) for k, v in actors.items()} if args.kl_coef > 0 else None
    agent = HASAC(actors, qcritics, lr=args.lr, critic_lr=args.critic_lr,
                  gamma=args.gamma, tau=args.tau, alpha=args.alpha,
                  ref_actors=ref_actors, kl_coef=args.kl_coef,
                  clip_eps=args.clip_eps)

    # RLPD 式预填：规则示范 transitions 常驻缓冲（防止在线精修洗掉 BC）
    if args.rlpd_steps > 0:
        n = agent.prefill_rule(env, args.rlpd_steps)
        print(f"[rlpd] prefilled {n} rule transitions (buffer={len(agent.buffer)})", flush=True)

    # 预热回放缓冲（在线随机采样）
    if args.warmup_steps > 0:
        n = agent.rollout(env, args.warmup_steps)
        print(f"[warmup] collected {n} transitions (buffer={len(agent.buffer)})", flush=True)

    t0 = time.time()
    metrics = {"delivered": [], "reward": [], "value_loss": [], "entropy": []}
    final_delivered = 0.0
    for it in range(args.iters):
        agent.rollout(env, args.steps)
        stats = agent.update(batch_size=args.batch_size, updates=args.updates_per_iter)
        metrics["value_loss"].append((it, stats["q_loss"]))
        metrics["entropy"].append((it, stats["entropy"]))
        if it % args.eval_every == 0 or it == args.iters - 1:
            delivered, reward = evaluate(env, agent)
            metrics["delivered"].append((it, delivered))
            metrics["reward"].append((it, reward))
            final_delivered = delivered
            print(f"[iter {it:4d}] q_loss={stats['q_loss']:.4f} "
                  f"policy_loss={stats['policy_loss']:.4f} entropy={stats['entropy']:.3f} "
                  f"delivered={delivered:.1f} reward={reward:.1f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    ckpt = {"actors": {k: v.state_dict() for k, v in actors.items()},
            "qcritics": {k: v.state_dict() for k, v in qcritics.items()}}
    torch.save(ckpt, args.out)

    rdir, prefix = run_dir(args.run_name)
    plot_curves(metrics, os.path.join(rdir, prefix + "_curves.png"),
                title=f"HASAC seed={args.seed} (role_aware={args.role_aware}, alpha={args.alpha})")
    torch.save(ckpt, os.path.join(rdir, prefix + ".pt"))
    write_summary(os.path.join(rdir, prefix + ".txt"),
                  f"scenario={args.scenario}\nstage={args.stage} seed={args.seed} "
                  f"role_aware={args.role_aware}\n"
                  f"iters={args.iters} steps={args.steps} lr={args.lr} "
                  f"critic_lr={args.critic_lr} alpha={args.alpha} clip_eps={args.clip_eps}\n"
                  f"final_delivered={final_delivered:.1f}\n")
    print(f"saved checkpoint to {args.out}")
    print(f"saved artifacts to {rdir} (prefix={prefix})")


if __name__ == "__main__":
    main()
