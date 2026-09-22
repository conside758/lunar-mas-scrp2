"""DAgger 采集（Gazebo）：用当前 BC 策略滚动，把访问状态按规则重标注，修复复合误差。

用法：ros2 run lunar_env collect_dagger_gazebo <scenario> <ckpt.pt> <out.pkl> <n_episodes>
"""
import pickle
import sys

import numpy as np
import torch
import rclpy

from lunar_env.env import LunarEnv
from lunar_env.rule_policy import rule_action
from lunar_rl.mappo import MAPPO
from lunar_rl.networks import ROBOT_NAMES, Actor, Critic
from lunar_rl.obs import encode_obs, obs_dim


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/dagger.pkl"
    n_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    odim = obs_dim()
    actors = {n: Actor(obs_dim=odim, hidden=128) for n in ROBOT_NAMES}
    critic = Critic(global_dim=3 * odim, hidden=128)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for n in ROBOT_NAMES:
        actors[n].load_state_dict(ckpt["actors"][n])
    agent = MAPPO(actors, critic, lr=1e-4)

    env = LunarEnv(scenario_file=scenario)
    data = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            # 专家重标注：规则动作作为标签
            rule = rule_action(obs)
            enc = encode_obs(obs)
            obs_batch = np.stack([enc[n] for n in ROBOT_NAMES]).astype(np.float32)
            drive = np.array([rule[n]["drive"] for n in ROBOT_NAMES], dtype=np.float32)
            tools = [rule[n].get("tool", None) for n in ROBOT_NAMES]
            data.append((obs_batch, drive, tools))
            # 用当前策略滚动（on-policy）
            action = agent.act(obs)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        print(f"[dagger] episode {ep + 1}/{n_episodes} done, samples={len(data)}",
              flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[dagger] saved {len(data)} samples to {out_path}", flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
