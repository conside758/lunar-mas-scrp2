"""Gazebo 部署验证：跑 Diffusion Policy（drive）+ 规则 tool，逐步记录并输出目的性/地形指标。

报告：delivered_value、每车 max roll/pitch、目标切换次数、路径迂回比（阶段段）、方向反转次数。
指标定义与 `lunar_rl.eval_purpose` 完全一致（surrogate 与 Gazebo 可比）。

用法：ros2 run lunar_env eval_deploy <scenario_file> <checkpoint.pt> <out.json> [--alpha 0.4]
"""
import json
import math
import sys

import numpy as np
import rclpy

from lunar_env.env import LunarEnv
from lunar_env.run_policy_diffusion import diffusion_action
from lunar_rl.diffusion_actor import DiffusionDriveActor
from lunar_rl.eval_purpose import _metrics, _phase_of
from lunar_rl.obs import FIXED_DIM
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE
from lunar_rl.rewards import goal_key_for


def load_actors(ckpt_path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    actors = {}
    for name, sd in ckpt["actors"].items():
        a = DiffusionDriveActor(fixed_dim=FIXED_DIM, hidden=128, attn_dim=64, action_dim=2)
        a.load_state_dict(sd)
        a.eval()
        actors[name] = a
    return actors


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
    out_json = sys.argv[3] if len(sys.argv) > 3 else "/home/admina/MAS_ws/.tmp/deploy_metrics.json"
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4

    env = LunarEnv(scenario_file=scenario)
    actors = load_actors(ckpt_path)
    roles = dict(ROBOT_ROLE)
    print(f"[eval_deploy] loaded {ckpt_path} ({len(actors)} actors), alpha={alpha}", flush=True)

    obs, info = env.reset()
    poses = {n: [env.pose[n]] for n in ROBOT_NAMES}
    keys = {n: [goal_key_for(env, n)] for n in ROBOT_NAMES}
    goals = {n: [] for n in ROBOT_NAMES}
    roll_hist = {n: [] for n in ROBOT_NAMES}
    for n in ROBOT_NAMES:
        goals[n].append(tuple(obs["goals"][n]))
        roll_hist[n].append(tuple(env.roll_pitch.get(n, (0.0, 0.0))))
    prev_drive = {}

    while True:
        action = diffusion_action(actors, obs, prev_drive=prev_drive, alpha=alpha)
        obs, _, term, trunc, info = env.step(action)
        for n in ROBOT_NAMES:
            poses[n].append(env.pose[n])
            keys[n].append(goal_key_for(env, n))
            goals[n].append(tuple(obs["goals"][n]))
            roll_hist[n].append(tuple(env.roll_pitch.get(n, (0.0, 0.0))))
        if term or trunc:
            break

    # 指标（与 surrogate eval_purpose 同款）
    metrics = {n: _metrics(poses[n], keys[n], goals[n], roles[n]) for n in ROBOT_NAMES}
    max_roll = {n: float(max(abs(r[0]) for r in roll_hist[n])) for n in ROBOT_NAMES}
    max_pitch = {n: float(max(abs(r[1]) for r in roll_hist[n])) for n in ROBOT_NAMES}

    summary = {
        "delivered": float(info["delivered"]),
        "delivered_value": float(info["delivered_value"]),
        "detected": int(info.get("detected", -1)),
        "steps": int(info.get("step", 0)),
        "max_roll_deg": {n: math.degrees(v) for n, v in max_roll.items()},
        "max_pitch_deg": {n: math.degrees(v) for n, v in max_pitch.items()},
        "max_roll_deg_team": float(math.degrees(max(max_roll.values()))),
        "max_pitch_deg_team": float(math.degrees(max(max_pitch.values()))),
        "metrics": metrics,
    }

    task = [n for n in ROBOT_NAMES if roles[n] in ("excavator", "hauler")]
    sw = {n: metrics[n]["goal_switches"] for n in ROBOT_NAMES}
    spr = {n: metrics[n]["same_phase_retargets"] for n in task}
    rev = {n: metrics[n]["reversals"] for n in task}
    det = {n: metrics[n]["detour_ratio_seg"] for n in task}

    print("\n[eval_deploy] ===== 部署指标 =====")
    print(f"  delivered_value = {summary['delivered_value']:.2f}  (delivered={summary['delivered']:.2f}, "
          f"det={summary['detected']}, steps={summary['steps']})")
    print(f"  team max roll/pitch = {summary['max_roll_deg_team']:.2f}deg / "
          f"{summary['max_pitch_deg_team']:.2f}deg")
    for n in ROBOT_NAMES:
        print(f"    {n:12s} max_roll={summary['max_roll_deg'][n]:5.2f}deg "
              f"max_pitch={summary['max_pitch_deg'][n]:5.2f}deg "
              f"goal_switches={sw[n]:3d} same_phase={spr.get(n, 0):3d} "
              f"detour={det.get(n, float('nan')) if det.get(n) is not None else float('nan'):.3f} "
              f"reversals={rev.get(n, 0):3d}")
    task_sw = float(np.mean([sw[n] for n in task]))
    task_spr = float(np.mean([spr[n] for n in task]))
    task_det = float(np.mean([det[n] for n in task if det[n] is not None])) \
        if any(det[n] is not None for n in task) else float("nan")
    task_rev = float(np.mean([rev[n] for n in task]))
    print(f"  [task] goal_switches={task_sw:.1f} same_phase_retargets={task_spr:.1f} "
          f"detour_ratio={task_det:.3f} reversals={task_rev:.1f}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval_deploy] saved {out_json}")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
