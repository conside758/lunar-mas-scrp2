"""运行训练好的 3-actor 策略：加载 checkpoint 驱动 lunar_env 跑完整回合并记录指标。

在 bringup 进程组内运行（绕开沙箱跨进程 DDS 限制）。
用法：ros2 run lunar_env run_policy <scenario_file> <checkpoint.pt>
"""
import sys

import numpy as np
import torch
import rclpy

from lunar_env.env import LunarEnv
from lunar_rl.networks import ROBOT_NAMES, Actor, Critic
from lunar_rl.obs import obs_dim
from lunar_rl.mappo import MAPPO


def main(args=None):
    rclpy.init(args=args)
    scenario = sys.argv[1] if len(sys.argv) > 1 else None
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
    env = LunarEnv(scenario_file=scenario)

    odim = obs_dim()
    actors = {n: Actor(obs_dim=odim, hidden=128) for n in ROBOT_NAMES}
    critic = Critic(global_dim=3 * odim, hidden=128)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for n in ROBOT_NAMES:
            actors[n].load_state_dict(ckpt["actors"][n])
        if "critic" in ckpt:
            critic.load_state_dict(ckpt["critic"])
        print(f"[policy] loaded {ckpt_path}", flush=True)
    agent = MAPPO(actors, critic, lr=1e-4)

    obs, info = env.reset()
    total = 0.0
    steps = 0
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if steps % 20 == 0:
            print(f"[policy] step={steps} delivered={info['delivered']:.1f} "
                  f"det={info.get('detected','?')}/3 "
                  f"scout=({obs['scout'][0]:.2f},{obs['scout'][1]:.2f}) "
                  f"excavator=({obs['excavator'][0]:.2f},{obs['excavator'][1]:.2f}) "
                  f"bucket={obs['excavator'][5]:.1f} extool={action['excavator'].get('tool')} "
                  f"hauler=({obs['hauler'][0]:.2f},{obs['hauler'][1]:.2f}) "
                  f"bin={obs['hauler'][5]:.1f} hatool={action['hauler'].get('tool')}",
                  flush=True)
        if terminated or truncated:
            break
    print(f"[policy] finished: steps={steps} reward={total:.2f} "
          f"delivered={info['delivered']:.1f}")
    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
