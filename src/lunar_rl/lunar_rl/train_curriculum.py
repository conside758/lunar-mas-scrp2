"""课程训练驱动：C0 → C1 → C2 → full 逐级推进。

每级：
  1. （可选）加载上一级 checkpoint 作为 actor/critic 初值（课程迁移）
  2. 用规则基线做 BC 预热（重放本级的规则 rollout）
  3. 预拟合集中 Critic（确定性 rollout 的蒙特卡洛回报）
  4. 低 lr RL 微调
  5. 保存本级 checkpoint

用法：python3 -m lunar_rl.train_curriculum <scenario> [--seed 1] ...
"""
import argparse
import os
import subprocess
import sys

STAGES = ["c0", "c1", "c2", "full"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bc-episodes", type=int, default=20)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--critic-warmup", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--out-dir", type=str,
                        default="/home/admina/MAS_ws/.tmp/curriculum")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prev_ckpt = None
    for stage in STAGES:
        out = os.path.join(args.out_dir, f"{stage}.pt")
        log = os.path.join(args.out_dir, f"{stage}.log")
        cmd = [sys.executable, "-m", "lunar_rl.train", args.scenario,
               "--stage", stage,
               "--seed", str(args.seed),
               "--bc-episodes", str(args.bc_episodes),
               "--bc-epochs", str(args.bc_epochs),
               "--critic-warmup", str(args.critic_warmup),
               "--lr", str(args.lr),
               "--iters", str(args.iters),
               "--steps", str(args.steps),
               "--eval-every", str(args.eval_every),
               "--out", out]
        if prev_ckpt:
            cmd += ["--load-actor", prev_ckpt]
        print(f"\n========== STAGE {stage} ==========", flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[stage {stage}] exit={rc.returncode} log={log} ckpt={out}", flush=True)
        if rc.returncode != 0:
            print(f"[stage {stage}] FAILED, stopping.", flush=True)
            sys.exit(rc.returncode)
        prev_ckpt = out

    print(f"\ncurriculum done. final checkpoint: {prev_ckpt}", flush=True)


if __name__ == "__main__":
    main()
