"""在 Gazebo 中运行规则基线（6 机器人拍卖版），收集 (VarRoleAwareActor 编码观测, rule_action) 示范数据。

绕开 sim-to-sim 迁移差距：直接用真实 Gazebo 动力学下的规则轨迹做 BC（见 docs/9.1-9.8.md）。
用法：ros2 run lunar_env collect_gazebo_data <scenario_file> <out.pkl> <n_episodes>
输出 pkl：{"names": [...], "roles": {...}, "samples": [encode_item(...), ...]}
"""
import pickle
import sys

import rclpy

from lunar_env.env import LunarEnv
from lunar_env.var_policy import encode_item
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE
from lunar_rl.rule_policy import rule_action


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/6r_gazebo_bc.pkl"
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    env = LunarEnv(scenario_file=scenario)  # 内部 set_robots(6 名字/角色)
    names = list(ROBOT_NAMES)
    roles = dict(ROBOT_ROLE)
    data = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            action = rule_action(obs)
            data.append(encode_item(obs, action))
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        print(f"[collect] episode {ep + 1}/{n_episodes} done, "
              f"delivered_value={info['delivered_value']:.1f}, samples={len(data)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"names": names, "roles": roles, "samples": data}, f)
    print(f"[collect] saved {len(data)} samples to {out_path}", flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
