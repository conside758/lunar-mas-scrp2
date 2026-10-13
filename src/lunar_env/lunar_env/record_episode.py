"""录制一个 Gazebo 回合的轨迹（供 2D 可视化 demo 渲染），并输出目的性/地形指标。

用 Diffusion Policy drive + 规则 tool 跑满一个回合，逐步记录：机器人位置、资源剩余量、depot、
交付量/价值、铲斗/车斗、工具动作、探测掩码、目标键（goal_key）。保存为 pkl（供 render_demo.py
渲染 GIF/图），并打印/落盘目的性指标（delivered_value、每车 max roll/pitch、目标切换、
路径迂回比、方向反转）。若 gzserver 中途崩溃（SIGINT），也会保存已录轨迹。

用法：ros2 run lunar_env record_episode <scenario_file> <checkpoint.pt> <out.pkl>
"""
import json
import math
import pickle
import sys

import numpy as np
import torch
import rclpy

from lunar_env.env import LunarEnv
from lunar_env.run_policy_diffusion import diffusion_action
from lunar_rl.diffusion_actor import DiffusionDriveActor
from lunar_rl.eval_purpose import _metrics
from lunar_rl.obs import FIXED_DIM
from lunar_rl.registry import ROBOT_ROLE
from lunar_rl.rewards import goal_key_for


def load_diffusion_actors(ckpt_path):
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
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/episode_traj.pkl"

    env = LunarEnv(scenario_file=scenario)
    actors = load_diffusion_actors(ckpt_path)
    roles = dict(ROBOT_ROLE)
    print(f"[record] loaded {ckpt_path} ({len(actors)} actors)", flush=True)

    traj = []
    obs, info = env.reset()
    prev_drive = {}
    # 目的性指标跟踪（与 surrogate eval_purpose 同款）
    names = list(roles.keys())
    poses = {n: [env.pose[n]] for n in names}
    keys = {n: [goal_key_for(env, n)] for n in names}
    goals = {n: [tuple(obs["goals"][n])] for n in names}
    roll_hist = {n: [tuple(env.roll_pitch.get(n, (0.0, 0.0)))] for n in names}

    try:
        while True:
            action = diffusion_action(actors, obs, prev_drive=prev_drive, alpha=0.8)
            frame = {
                "step": int(info.get("step", 0)),
                "robots": {n: [float(obs[n][0]), float(obs[n][1])] for n in roles},
                "roll_pitch": {n: [float(env.roll_pitch[n][0]), float(env.roll_pitch[n][1])]
                               for n in roles},  # 倾倒监测
                "resources": [[float(r[0]), float(r[1]), float(r[2]), float(r[3])]
                              for r in obs["resources"]],
                "depots": [[float(d[0]), float(d[1])] for d in obs["depots"]],
                "detected": [int(d) for d in obs["detected"]],
                "delivered": float(info["delivered"]),
                "delivered_value": float(info["delivered_value"]),
                "buckets": {n: float(obs[n][5]) for n in roles if roles[n] == "excavator"},
                "bins": {n: float(obs[n][5]) for n in roles if roles[n] == "hauler"},
                "tools": {n: action[n].get("tool", -1) for n in roles},
            }
            traj.append(frame)
            obs, reward, terminated, truncated, info = env.step(action)
            for n in names:
                poses[n].append(env.pose[n])
                keys[n].append(goal_key_for(env, n))
                goals[n].append(tuple(obs["goals"][n]))
                roll_hist[n].append(tuple(env.roll_pitch.get(n, (0.0, 0.0))))
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        pass
    finally:
        with open(out_path, "wb") as f:
            pickle.dump({"roles": roles, "traj": traj}, f)
        print(f"[record] saved {len(traj)} frames to {out_path} "
              f"(final delivered_value={traj[-1]['delivered_value'] if traj else 0:.1f})",
              flush=True)

        # 目的性/地形指标汇总
        metrics = {n: _metrics(poses[n], keys[n], goals[n], roles[n]) for n in names}
        max_roll = {n: float(max(abs(r[0]) for r in roll_hist[n])) for n in names}
        max_pitch = {n: float(max(abs(r[1]) for r in roll_hist[n])) for n in names}
        task = [n for n in names if roles[n] in ("excavator", "hauler")]
        sw = {n: metrics[n]["goal_switches"] for n in names}
        spr = {n: metrics[n]["same_phase_retargets"] for n in task}
        det = {n: metrics[n]["detour_ratio_seg"] for n in task}
        rev = {n: metrics[n]["reversals"] for n in task}
        summary = {
            "delivered_value": float(traj[-1]["delivered_value"]) if traj else 0.0,
            "delivered": float(traj[-1]["delivered"]) if traj else 0.0,
            "detected": int(info.get("detected", -1)),
            "steps": int(info.get("step", 0)),
            "max_roll_deg": {n: math.degrees(v) for n, v in max_roll.items()},
            "max_pitch_deg": {n: math.degrees(v) for n, v in max_pitch.items()},
            "max_roll_deg_team": float(math.degrees(max(max_roll.values(), default=0.0))),
            "max_pitch_deg_team": float(math.degrees(max(max_pitch.values(), default=0.0))),
            "goal_switches": sw,
            "same_phase_retargets": spr,
            "detour_ratio_seg": {n: (None if det[n] is None else float(det[n])) for n in task},
            "reversals": rev,
        }
        print("\n[record] ===== 部署指标 =====")
        print(f"  delivered_value = {summary['delivered_value']:.2f}  (delivered={summary['delivered']:.2f}, "
              f"det={summary['detected']}, steps={summary['steps']})")
        print(f"  team max roll/pitch = {summary['max_roll_deg_team']:.2f}deg / "
              f"{summary['max_pitch_deg_team']:.2f}deg")
        for n in names:
            dv = det.get(n)
            print(f"    {n:12s} max_roll={summary['max_roll_deg'][n]:5.2f}deg "
                  f"max_pitch={summary['max_pitch_deg'][n]:5.2f}deg "
                  f"goal_switches={sw[n]:3d} same_phase={spr.get(n, 0):3d} "
                  f"detour={(f'{dv:.3f}' if dv is not None else 'n/a'):>6s} "
                  f"reversals={rev.get(n, 0):3d}")
        task_det = [det[n] for n in task if det[n] is not None]
        print(f"  [task] goal_switches={np.mean([sw[n] for n in task]):.1f} "
              f"same_phase_retargets={np.mean([spr[n] for n in task]):.1f} "
              f"detour_ratio={(float(np.mean(task_det)) if task_det else float('nan')):.3f} "
              f"reversals={np.mean([rev[n] for n in task]):.1f}")
        with open(out_path + ".metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[record] saved {out_path}.metrics.json", flush=True)
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
