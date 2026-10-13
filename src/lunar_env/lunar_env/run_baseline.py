"""运行规则基线（6 机器人拍卖版）：用 lunar_rl.rule_policy.rule_action 驱动 lunar_env。

目标点取自 obs["goals"]（由 lunar_rl.rewards.goal_for 计算：scout→最近未发现资源；
excavator→拍卖分配资源；hauler→最近 depot / 有货 excavator）。在 bringup 进程组内运行。
用法：ros2 run lunar_env run_baseline <scenario_file>
"""
import sys

import rclpy
from lunar_env.env import LunarEnv
from lunar_rl.rule_policy import rule_action


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
            det = info.get('detected', '?')
            buckets = {n: round(obs[n][5], 1) for n in ("excavator_0", "excavator_1")}
            bins = {n: round(obs[n][5], 1) for n in ("hauler_0", "hauler_1")}
            pos = {n: (round(obs[n][0], 2), round(obs[n][1], 2))
                   for n in ("excavator_0", "excavator_1", "hauler_0", "hauler_1")}
            goals = {n: (round(obs['goals'][n][0], 2), round(obs['goals'][n][1], 2))
                     for n in ("excavator_0", "excavator_1", "hauler_0", "hauler_1")}
            print(f"[baseline] step={steps} delivered={info['delivered']:.1f} "
                  f"delivered_value={info['delivered_value']:.1f} det={det} "
                  f"buckets={buckets} bins={bins}", flush=True)
            print(f"    pos={pos} goals={goals}", flush=True)
        if terminated or truncated:
            break
    print(f"[baseline] finished: steps={steps} reward={total:.2f} "
          f"delivered={info['delivered']:.1f} delivered_value={info['delivered_value']:.1f}",
          flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
