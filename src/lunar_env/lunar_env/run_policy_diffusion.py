"""运行 Diffusion Policy 驱动 + 规则 tool 的混合策略。

drive 用 DDPM 反向采样生成（替换高斯 MSE 均值），tool 用规则状态机（BC tool 类不平衡）。
用法：ros2 run lunar_env run_policy_diffusion <scenario_file> <checkpoint.pt>
"""
import sys

import numpy as np
import torch
import rclpy

from lunar_env.env import LunarEnv
from lunar_rl.diffusion_actor import DiffusionDriveActor
from lunar_rl.obs import encode_obs_vars, FIXED_DIM
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE, ROLE_TOOL_DIM, set_robots
from lunar_rl.rule_policy import rule_action

MAX_RES = 8
MAX_TEAM = 5
MAX_DEPOT = 2


def diffusion_action(actors, obs, prev_drive=None, alpha=0.4):
    """DDPM 采样 drive + 规则 tool；对 drive 做 EMA 时序平滑（抑制逐步抖动/来回踱步）。

    prev_drive: {name: ndarray(2,)} 上一帧平滑后的 drive（原地更新）。
    alpha: 平滑系数（越大越贴近原始采样，越小越平滑）。
    """
    enc = encode_obs_vars(obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
    rule = rule_action(obs)
    action = {}
    with torch.no_grad():
        for name in ROBOT_NAMES:
            e = enc[name]
            fixed = torch.tensor(e["fixed"][None], dtype=torch.float32)
            res_ent = torch.tensor(e["res_ent"][None], dtype=torch.float32)
            depot_ent = torch.tensor(e["depot_ent"][None], dtype=torch.float32)
            team_ent = torch.tensor(e["team_ent"][None], dtype=torch.float32)
            res_mask = torch.tensor(e["res_mask"][None], dtype=torch.float32)
            depot_mask = torch.tensor(e["depot_mask"][None], dtype=torch.float32)
            team_mask = torch.tensor(e["team_mask"][None], dtype=torch.float32)
            cond = actors[name].encode(fixed, res_ent, team_ent, res_mask, team_mask,
                                       depot_ent, depot_mask)
            drive = actors[name].sample(cond)[0].numpy().astype(np.float32)
            if prev_drive is not None and name in prev_drive:
                drive = alpha * drive + (1.0 - alpha) * prev_drive[name]
            if prev_drive is not None:
                prev_drive[name] = drive.copy()
            sub = {"drive": np.clip(drive, -1.0, 1.0).astype(np.float32)}
            if "tool" in rule.get(name, {}):
                sub["tool"] = rule[name]["tool"]
            action[name] = sub
    return action


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None

    env = LunarEnv(scenario_file=scenario)  # 内部 set_robots(6 名字/角色)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    actors = {}
    for name, sd in ckpt["actors"].items():
        a = DiffusionDriveActor(fixed_dim=FIXED_DIM, hidden=128, attn_dim=64, action_dim=2)
        a.load_state_dict(sd)
        a.eval()
        actors[name] = a
    print(f"[diffusion] loaded {ckpt_path} ({len(actors)} actors)", flush=True)

    obs, info = env.reset()
    total = 0.0
    steps = 0
    prev_drive = {}
    while True:
        action = diffusion_action(actors, obs, prev_drive=prev_drive, alpha=0.4)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if steps % 20 == 0:
            buckets = {n: round(obs[n][5], 1) for n in ("excavator_0", "excavator_1")}
            bins = {n: round(obs[n][5], 1) for n in ("hauler_0", "hauler_1")}
            tools = {n: action[n].get('tool', '-') for n in ("excavator_0", "excavator_1",
                                                             "hauler_0", "hauler_1")}
            print(f"[diffusion] step={steps} delivered={info['delivered']:.1f} "
                  f"delivered_value={info['delivered_value']:.1f} det={info.get('detected','?')} "
                  f"tools={tools} buckets={buckets} bins={bins}", flush=True)
        if terminated or truncated:
            break
    print(f"[diffusion] finished: steps={steps} reward={total:.2f} "
          f"delivered={info['delivered']:.1f} delivered_value={info['delivered_value']:.1f}",
          flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
