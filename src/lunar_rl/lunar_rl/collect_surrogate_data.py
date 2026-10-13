"""在 surrogate 上运行「承诺式规则专家」收集示范数据（含地形感知观测）。

数据格式与 collect_gazebo_data / var_policy.encode_item 完全一致：
    {"names": [...], "roles": {...}, "samples": [{fixed/res_ent/depot_ent/team_ent/各 mask/drive/tools}, ...]}
供 train_diffusion_bc.py 直接训练 Diffusion Policy。

用法：python3 -m lunar_rl.collect_surrogate_data <scenario.yaml> <out.pkl> <n_episodes> [--seed0 0]
"""
import argparse
import pickle

import numpy as np

from lunar_rl.obs import encode_obs_vars
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE
from lunar_rl.rule_policy import rule_action
from lunar_rl.surrogate import SurrogateEnv

MAX_RES = 8
MAX_TEAM = 5
MAX_DEPOT = 2


def encode_item(obs, action):
    enc = encode_obs_vars(obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
    return {
        "fixed": np.stack([enc[n]["fixed"] for n in ROBOT_NAMES]).astype(np.float32),
        "res_ent": np.stack([enc[n]["res_ent"] for n in ROBOT_NAMES]).astype(np.float32),
        "depot_ent": np.stack([enc[n]["depot_ent"] for n in ROBOT_NAMES]).astype(np.float32),
        "team_ent": np.stack([enc[n]["team_ent"] for n in ROBOT_NAMES]).astype(np.float32),
        "res_mask": np.stack([enc[n]["res_mask"] for n in ROBOT_NAMES]).astype(np.float32),
        "depot_mask": np.stack([enc[n]["depot_mask"] for n in ROBOT_NAMES]).astype(np.float32),
        "team_mask": np.stack([enc[n]["team_mask"] for n in ROBOT_NAMES]).astype(np.float32),
        "drive": np.array([action[n]["drive"] for n in ROBOT_NAMES], dtype=np.float32),
        "tools": [action[n].get("tool", 0) for n in ROBOT_NAMES],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("out_pkl")
    ap.add_argument("n_episodes", type=int)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--domain-rand", action="store_true",
                    help="开启速度/转向域随机化（使训练分布贴近 Gazebo 物理噪声）")
    args = ap.parse_args()

    env = SurrogateEnv(args.scenario, committed=True, domain_rand=args.domain_rand)
    names = list(ROBOT_NAMES)
    roles = dict(ROBOT_ROLE)
    data = []
    for ep in range(args.n_episodes):
        np.random.seed(args.seed0 + ep)
        obs, _ = env.reset()
        while True:
            action = rule_action(obs)
            data.append(encode_item(obs, action))
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        print(f"[collect] ep {ep + 1}/{args.n_episodes} done, "
              f"delivered_value={info['delivered_value']:.1f}, samples={len(data)}",
              flush=True)

    with open(args.out_pkl, "wb") as f:
        pickle.dump({"names": names, "roles": roles, "samples": data}, f)
    print(f"[collect] saved {len(data)} samples to {args.out_pkl}", flush=True)


if __name__ == "__main__":
    main()
