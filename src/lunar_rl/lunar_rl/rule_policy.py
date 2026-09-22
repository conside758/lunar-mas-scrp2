"""规则基线策略：拍卖式任务分配 + 会合/装卸状态机（支持任意数量机器人 / 多 depot / 异构资源价值）。

将 obs 映射为 env 的动作 dict（drive + tool）。目标点直接取自 obs["goals"]（由 rewards.goal_for
计算：scout→最近未发现资源；excavator→拍卖分配资源；hauler→最近 depot / 有货 excavator）。
"""
import math

import numpy as np

from lunar_rl.registry import ROBOT_ROLE


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def go_to(robot_obs, tx, ty):
    """去目标点的 go-to-goal 控制器（平滑版），返回 drive=[vx_norm, wz_norm]。"""
    x, y, yaw = robot_obs[0], robot_obs[1], robot_obs[2]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist < 0.08:
        return [0.0, 0.0]
    err = _norm_angle(math.atan2(dy, dx) - yaw)
    vx = _clip(1.0 * dist, 0.0, 1.0) * max(0.0, math.cos(err)) ** 2
    wz = _clip(1.2 * err, -1.0, 1.0)
    return [vx, wz]


def rule_action(obs, dig_radius=0.6, load_radius=0.7, dump_radius=0.7,
                bucket_cap=4.0, bin_cap=10.0):
    """根据观测返回规则动作 dict。目标点用 obs["goals"]（拍卖分配），工具按角色状态机决定。"""
    goals = obs.get("goals", {})
    action = {}
    for name, o in obs.items():
        if name in ("resources", "detected", "depots", "depot", "goals"):
            continue
        role = ROBOT_ROLE.get(name, "scout")
        goal = goals.get(name)
        gx, gy = (float(goal[0]), float(goal[1])) if goal is not None else (o[0], o[1])
        d = math.hypot(o[0] - gx, o[1] - gy)
        cargo = float(o[5]) if len(o) >= 6 else 0.0

        if role == "excavator":
            if cargo >= bucket_cap - 0.05:
                # 满斗 → 去装载（goal=最近 hauler）
                if d < load_radius:
                    action[name] = {"drive": [0.0, 0.0], "tool": 2}
                else:
                    action[name] = {"drive": go_to(o, gx, gy), "tool": 0}
            else:
                # 挖斗未满 → 去挖（goal=拍卖分配资源）
                if d < dig_radius:
                    action[name] = {"drive": [0.0, 0.0], "tool": 1}
                else:
                    action[name] = {"drive": go_to(o, gx, gy), "tool": 0}
        elif role == "hauler":
            if cargo > 0.05:
                if d < dump_radius:
                    action[name] = {"drive": [0.0, 0.0], "tool": 1}
                else:
                    action[name] = {"drive": go_to(o, gx, gy), "tool": 0}
            else:
                action[name] = {"drive": go_to(o, gx, gy), "tool": 0}
        else:  # scout：探索（goal=最近未发现资源），无工具
            action[name] = {"drive": go_to(o, gx, gy)}
    return action
