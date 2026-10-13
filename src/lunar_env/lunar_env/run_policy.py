"""运行训练好的 6 机器人角色感知 BC 策略（VarRoleAwareActor）：加载 checkpoint 驱动 lunar_env。

在 bringup 进程组内运行（绕开沙箱跨进程 DDS 限制）。
用法：ros2 run lunar_env run_policy <scenario_file> <checkpoint.pt>

checkpoint 结构：{"actors": {机器人名: VarRoleAwareActor.state_dict()}}。观测用 encode_obs_vars（与替身一致），
确定性推理与 DAgger 采集共用 `lunar_env.var_policy.det_action`。
"""
import sys

import rclpy

from lunar_env.env import LunarEnv
from lunar_env.var_policy import det_action, load_var_actors


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None

    env = LunarEnv(scenario_file=scenario)  # 内部已 set_robots(6 名字/角色)

    if not ckpt_path:
        raise RuntimeError("checkpoint path required")
    actors = load_var_actors(ckpt_path)
    print(f"[policy] loaded {ckpt_path} ({len(actors)} actors: "
          f"{sorted(actors.keys())})", flush=True)

    obs, info = env.reset()
    total = 0.0
    steps = 0
    while True:
        action = det_action(actors, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if steps % 20 == 0:
            det = info.get('detected', '?')
            tools = {n: action[n].get('tool', '-') for n in ("excavator_0", "excavator_1",
                                                             "hauler_0", "hauler_1")}
            buckets = {n: round(obs[n][5], 1) for n in ("excavator_0", "excavator_1")}
            bins = {n: round(obs[n][5], 1) for n in ("hauler_0", "hauler_1")}
            print(f"[policy] step={steps} delivered={info['delivered']:.1f} "
                  f"delivered_value={info['delivered_value']:.1f} det={det} "
                  f"tools={tools} buckets={buckets} bins={bins}", flush=True)
        if terminated or truncated:
            break
    print(f"[policy] finished: steps={steps} reward={total:.2f} "
          f"delivered={info['delivered']:.1f} delivered_value={info['delivered_value']:.1f}",
          flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
