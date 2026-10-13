"""混合/残差策略：drive 在「规则 drive」与「BC drive」间按 alpha 混合，tool 用规则状态机。

drive = clip(rule_drive + alpha * (bc_drive - rule_drive), -1, 1)
- alpha=0 → 纯规则 drive（基线）
- alpha=1 → 纯 BC drive（原混合策略）
- 0<alpha<1 → 残差混合（BC 只对规则 drive 做有界修正）
tool 始终用规则状态机（BC 的 softmax tool 类不平衡，见 docs/9.23-9.28.md §七）。
用法：ros2 run lunar_env run_policy_hybrid <scenario_file> <checkpoint.pt> [alpha]
"""
import sys

import numpy as np
import rclpy

from lunar_env.env import LunarEnv
from lunar_env.var_policy import det_action, load_var_actors
from lunar_rl.rule_policy import rule_action


def hybrid_action(actors, obs, alpha=1.0):
    """drive 按 alpha 混合规则与 BC，tool 用规则。"""
    bc = det_action(actors, obs)
    rule = rule_action(obs)
    action = {}
    for name in bc:
        rd = np.asarray(rule[name]["drive"], dtype=np.float32)
        bd = np.asarray(bc[name]["drive"], dtype=np.float32)
        drive = np.clip(rd + alpha * (bd - rd), -1.0, 1.0)
        sub = {"drive": drive.astype(np.float32)}
        if "tool" in rule.get(name, {}):
            sub["tool"] = rule[name]["tool"]
        action[name] = sub
    return action


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    env = LunarEnv(scenario_file=scenario)
    actors = load_var_actors(ckpt_path)
    print(f"[hybrid] loaded {ckpt_path} ({len(actors)} actors), alpha={alpha}", flush=True)

    obs, info = env.reset()
    total = 0.0
    steps = 0
    while True:
        action = hybrid_action(actors, obs, alpha=alpha)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if steps % 20 == 0:
            buckets = {n: round(obs[n][5], 1) for n in ("excavator_0", "excavator_1")}
            bins = {n: round(obs[n][5], 1) for n in ("hauler_0", "hauler_1")}
            tools = {n: action[n].get('tool', '-') for n in ("excavator_0", "excavator_1",
                                                             "hauler_0", "hauler_1")}
            print(f"[hybrid] step={steps} delivered={info['delivered']:.1f} "
                  f"delivered_value={info['delivered_value']:.1f} det={info.get('detected','?')} "
                  f"tools={tools} buckets={buckets} bins={bins}", flush=True)
        if terminated or truncated:
            break
    print(f"[hybrid] finished: steps={steps} reward={total:.2f} "
          f"delivered={info['delivered']:.1f} delivered_value={info['delivered_value']:.1f}",
          flush=True)
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
