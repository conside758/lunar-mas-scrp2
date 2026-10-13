"""在 surrogate 上评估 Diffusion Policy（drive）+ 规则 tool 的交付量与行为（快速回归）。

与 Gazebo 部署共用同一推理逻辑（encode_obs_vars + DDPM 采样 + 规则 tool），
用于训练后先做替身回归（预期 delivered_value 接近专家），再上 Gazebo。

用法：python3 -m lunar_rl.eval_diffusion_surrogate <scenario.yaml> <checkpoint.pt> [--episodes 5] [--alpha 0.4]
"""
import argparse

import numpy as np
import torch

from lunar_rl.diffusion_actor import DiffusionDriveActor
from lunar_rl.obs import FIXED_DIM, encode_obs_vars
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE, ROLE_TOOL_DIM
from lunar_rl.rule_policy import rule_action
from lunar_rl.surrogate import SurrogateEnv

MAX_RES = 8
MAX_TEAM = 5
MAX_DEPOT = 2


def load_actors(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    actors = {}
    for name, sd in ckpt["actors"].items():
        a = DiffusionDriveActor(fixed_dim=FIXED_DIM, hidden=128, attn_dim=64, action_dim=2)
        a.load_state_dict(sd)
        a.eval()
        actors[name] = a
    return actors


def diffusion_action(actors, obs, prev_drive=None, alpha=0.4):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()

    actors = load_actors(args.ckpt)
    env = SurrogateEnv(args.scenario, committed=True)
    totals = []
    for ep in range(args.episodes):
        np.random.seed(1000 + ep)
        obs, _ = env.reset()
        prev = {}
        while True:
            action = diffusion_action(actors, obs, prev_drive=prev, alpha=args.alpha)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        totals.append(info["delivered_value"])
        print(f"[surrogate-diffusion] ep {ep + 1} delivered_value={info['delivered_value']:.1f} "
              f"max_roll={info.get('max_roll_deg', 0):.1f}deg max_pitch={info.get('max_pitch_deg', 0):.1f}deg",
              flush=True)
    print(f"[surrogate-diffusion] mean delivered_value={float(np.mean(totals)):.1f} "
          f"per-ep={totals}")


if __name__ == "__main__":
    main()
