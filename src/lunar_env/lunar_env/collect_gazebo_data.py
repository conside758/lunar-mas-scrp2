"""在 Gazebo 中运行规则基线，收集 (encoded_obs, rule_action) 示范数据用于 BC 预热。

绕开 sim-to-sim 迁移差距：直接用真实 Gazebo 动力学下的规则轨迹做 BC。
用法：ros2 run lunar_env collect_gazebo_data <scenario> <out.pkl> <n_episodes>
"""
import pickle
import sys

import numpy as np
import rclpy

from lunar_env.env import LunarEnv
from lunar_env.rule_policy import rule_action
from lunar_rl.networks import ROBOT_NAMES
from lunar_rl.obs import encode_obs


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gazebo_bc.pkl"
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    env = LunarEnv(scenario_file=scenario)
    data = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            action = rule_action(obs)
            enc = encode_obs(obs)
            obs_batch = np.stack([enc[n] for n in ROBOT_NAMES]).astype(np.float32)
            drive = np.array([action[n]["drive"] for n in ROBOT_NAMES],
                             dtype=np.float32)
            tools = [action[n].get("tool", None) for n in ROBOT_NAMES]
            data.append((obs_batch, drive, tools))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        print(f"[collect] episode {ep + 1}/{n_episodes} done, "
              f"samples={len(data)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[collect] saved {len(data)} samples to {out_path}", flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
