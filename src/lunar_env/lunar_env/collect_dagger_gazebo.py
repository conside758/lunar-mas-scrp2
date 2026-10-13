"""DAgger 采集（Gazebo）：用当前 BC 策略（VarRoleAwareActor）on-policy 滚动，按规则重标注，修复复合误差。

用法：ros2 run lunar_env collect_dagger_gazebo <scenario> <ckpt.pt> <out.pkl> <n_episodes>
输出 pkl：{"names": [...], "roles": {...}, "samples": [encode_item(...), ...]}
"""
import pickle
import sys

import rclpy

from lunar_env.env import LunarEnv
from lunar_env.var_policy import det_action, encode_item, load_var_actors
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE
from lunar_rl.rule_policy import rule_action


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
    out_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/6r_dagger.pkl"
    n_episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    actors = load_var_actors(ckpt_path)
    print(f"[dagger] loaded policy {ckpt_path} ({len(actors)} actors)", flush=True)

    env = LunarEnv(scenario_file=scenario)
    names = list(ROBOT_NAMES)
    roles = dict(ROBOT_ROLE)
    data = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            # 专家重标注：规则动作作为标签
            rule = rule_action(obs)
            data.append(encode_item(obs, rule))
            # 用当前策略滚动（on-policy），把策略访问到的状态纳入下一轮训练
            action = det_action(actors, obs)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        print(f"[dagger] episode {ep + 1}/{n_episodes} done, samples={len(data)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"names": names, "roles": roles, "samples": data}, f)
    print(f"[dagger] saved {len(data)} samples to {out_path}", flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
