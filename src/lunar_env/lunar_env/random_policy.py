"""随机策略冒烟测试：验证 LunarEnv 可 reset/step 且不崩溃。

用法（需先启动 bringup）：
    ros2 run lunar_env random_policy <scenario_file>
"""
import sys

import rclpy

from lunar_env.env import LunarEnv


def main(args=None):
    rclpy.init(args=args)
    scenario = None
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    env = LunarEnv(scenario_file=scenario)
    try:
        for ep in range(3):
            obs, info = env.reset()
            total = 0.0
            steps = 0
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            print(f"[random_policy] episode {ep}: reward={total:.2f} "
                  f"delivered={info['delivered']:.1f} steps={steps}")
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
