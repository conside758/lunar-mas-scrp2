"""运行规则基线：用 rule_policy 驱动 lunar_env 跑一个完整回合并记录指标。

在 bringup 进程组内运行（绕开沙箱跨进程 DDS 限制）。默认跑满 max_steps。
用法：ros2 run lunar_env run_baseline <scenario_file>
"""
import sys

import rclpy
from lunar_env.env import LunarEnv
from lunar_env.rule_policy import rule_action


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    env = LunarEnv(scenario_file=scenario)

    obs, info = env.reset()
    total = 0.0
    steps = 0
    while True:
        action = rule_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if steps % 20 == 0:
            print(f"[baseline] step={steps} delivered={info['delivered']:.1f} "
                  f"excavator=({obs['excavator'][0]:.2f},{obs['excavator'][1]:.2f},"
                  f"yaw={obs['excavator'][2]:.2f}) bucket={obs['excavator'][5]:.1f} "
                  f"hauler=({obs['hauler'][0]:.2f},{obs['hauler'][1]:.2f}) "
                  f"bin={obs['hauler'][5]:.1f}", flush=True)
        if terminated or truncated:
            break
    print(f"[baseline] finished: steps={steps} reward={total:.2f} "
          f"delivered={info['delivered']:.1f}")
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
