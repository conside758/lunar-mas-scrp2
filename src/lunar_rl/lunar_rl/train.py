"""MAPPO 基线训练入口（在快速替身环境上训练）。

用法：python3 -m lunar_rl.train <scenario_file> [--iters N] [--steps S]
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

from lunar_rl.bc import collect_rule_data, dagger, train_bc
from lunar_rl.mappo import MAPPO
from lunar_rl.networks import ROBOT_NAMES, Actor, Critic, QCritic, RoleAwareActor
from lunar_rl.obs import obs_dim
from lunar_rl.runlog import plot_curves, run_dir, write_summary
from lunar_rl.surrogate import SurrogateEnv


def evaluate(env, agent, episodes=3):
    """确定性评估，返回 (平均交付量, 平均回合回报)。"""
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
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-episodes", type=int, default=0,
                        help="规则基线 BC 预热的回合数（0 表示跳过）")
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--dagger-rounds", type=int, default=0,
                        help="DAgger 迭代轮数（0 表示用普通 BC）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=None,
                        help="Critic/QCritic 独立学习率（None=与 --lr 相同；HAPPO 反事实 Q 建议更高，如 3e-4）")
    parser.add_argument("--cql-coef", type=float, default=0.0,
                        help="Cal-QL/CQL 保守 Q 惩罚系数（对 OOD 均值动作的 Q 加惩罚，防漂移；>0 启用）")
    parser.add_argument("--stage", type=str, default="full",
                        help="课程阶段：c0 / c1 / c2 / full")
    parser.add_argument("--load-actor", type=str, default=None,
                        help="从已有 checkpoint 加载 actor/critic 作为预热")
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="RL 前用确定性 rollout 预拟合 critic 的回合数（0 表示跳过）")
    parser.add_argument("--std-init", type=float, default=-0.7,
                        help="drive logstd 初值（-0.7→std≈0.5 探索；-2.0→std≈0.14 细调）")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help="对 BC 参考策略的 drive 均值 L2 正则系数（>0 启用防漂移）")
    parser.add_argument("--app-weight", type=float, default=None,
                        help="覆盖吸引奖励权重（微调时可调低以降低回报方差）")
    parser.add_argument("--domain-rand", action="store_true",
                        help="训练时对速度/转向做域随机（模拟 Gazebo 摩擦效率下降）")
    parser.add_argument("--role-aware", action="store_true",
                        help="用 RoleAwareActor（资源/队友注意力池化）替代普通 Actor")
    parser.add_argument("--hap", action="store_true",
                        help="用 HAPPO 顺序更新（逐 Agent 独立优化器 + 各自信任域）替代 MAPPO 同时更新")
    parser.add_argument("--run-name", type=str, default=None,
                        help="结果目录里本次运行的文件前缀（默认 run；结果存 results/<当天日期>/）")
    parser.add_argument("--out", type=str, default="/home/admina/MAS_ws/.tmp/mappo_baseline.pt")
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
    critic = Critic(global_dim=3 * odim, hidden=128)
    qcritic = QCritic(global_dim=3 * odim, action_dim=11, hidden=128) if args.hap else None

    if args.load_actor:
        ckpt = torch.load(args.load_actor, map_location="cpu")
        for name in ROBOT_NAMES:
            actors[name].load_state_dict(ckpt["actors"][name])
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        if qcritic is not None and ckpt.get("qcritic") is not None:
            qcritic.load_state_dict(ckpt["qcritic"])
        print(f"[load] warm-started from {args.load_actor}", flush=True)

    if args.bc_episodes > 0:
        if args.dagger_rounds > 0:
            print(f"[dagger] init={args.bc_episodes} rounds={args.dagger_rounds} "
                  f"(stage={args.stage})...", flush=True)
            dagger(actors, env, rounds=args.dagger_rounds,
                   episodes=args.bc_episodes, epochs=args.bc_epochs,
                   init_episodes=args.bc_episodes)
        else:
            print(f"[bc] collecting {args.bc_episodes} rule episodes (stage={args.stage})...", flush=True)
            data = collect_rule_data(env, args.bc_episodes)
            train_bc(actors, data, epochs=args.bc_epochs)

    ref_actors = {k: copy.deepcopy(v) for k, v in actors.items()} if args.kl_coef > 0 else None
    agent = MAPPO(actors, critic, lr=args.lr, gamma=args.gamma,
                  entropy_coef=args.entropy_coef, ref_actors=ref_actors,
                  kl_coef=args.kl_coef, hap=args.hap, qcritic=qcritic,
                  critic_lr=args.critic_lr, cql_coef=args.cql_coef)

    if args.critic_warmup > 0:
        agent.warmup_critic(env, episodes=args.critic_warmup, epochs=30)

    t0 = time.time()
    metrics = {"delivered": [], "reward": [], "value_loss": [], "entropy": []}
    final_delivered = 0.0
    for it in range(args.iters):
        buf = agent.rollout(env, args.steps)
        stats = agent.update(buf)
        metrics["value_loss"].append((it, stats["value_loss"]))
        metrics["entropy"].append((it, stats["entropy"]))
        if it % args.eval_every == 0 or it == args.iters - 1:
            delivered, reward = evaluate(env, agent)
            metrics["delivered"].append((it, delivered))
            metrics["reward"].append((it, reward))
            final_delivered = delivered
            print(f"[iter {it:4d}] policy_loss={stats['policy_loss']:.4f} "
                  f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.3f} "
                  f"delivered={delivered:.1f} reward={reward:.1f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    ckpt = {"actors": {k: v.state_dict() for k, v in actors.items()},
            "critic": critic.state_dict(),
            "qcritic": qcritic.state_dict() if qcritic is not None else None}
    torch.save(ckpt, args.out)

    # 结果目录：曲线图 + checkpoint 副本 + 摘要
    rdir, prefix = run_dir(args.run_name)
    plot_curves(metrics, os.path.join(rdir, prefix + "_curves.png"),
                title=f"MAPPO/HAPPO seed={args.seed} (role_aware={args.role_aware}, hap={args.hap})")
    torch.save(ckpt, os.path.join(rdir, prefix + ".pt"))
    write_summary(os.path.join(rdir, prefix + ".txt"),
                  f"scenario={args.scenario}\nstage={args.stage} seed={args.seed} "
                  f"role_aware={args.role_aware} hap={args.hap}\n"
                  f"iters={args.iters} steps={args.steps} lr={args.lr} "
                  f"critic_lr={args.critic_lr} cql_coef={args.cql_coef}\n"
                  f"final_delivered={final_delivered:.1f}\n")
    print(f"saved checkpoint to {args.out}")
    print(f"saved artifacts to {rdir} (prefix={prefix})")


if __name__ == "__main__":
    main()
