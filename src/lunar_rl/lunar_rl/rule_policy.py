"""规则基线策略：承诺式任务分配 + 会合/装卸状态机（支持任意数量机器人 / 多 depot / 异构资源价值）。

将 obs 映射为 env 的动作 dict（drive + tool）。目标点直接取自 obs["goals"]（由 rewards.TaskAllocator
承诺式计算：scout→探测未发现资源；excavator→承诺挖掘资源/固定配对装载；hauler→固定配对服务/承诺 depot）。

go_to / go_to_smooth 带「转向滞回 + 到达死区」：
  - 转向滞回：|heading 误差|>0.5 rad 才进入原地转向，<0.3 rad 才退出，抑制临界抖动；
  - 到达死区：距离 < arrive_radius 即锁定停止，不再微调。
历史经验（docs/9.1-9.8.md）：纯转向效率 ~50%，「边前进边转向」效率骤降，故 diff-drive 大航向误差先原地转向。
"""
import math

import numpy as np

from lunar_rl.registry import ROBOT_ROLE
from lunar_rl.terrain import get_terrain

# 斜坡减速增益：坡度越大，前进速度越低（坡度过陡时几乎停下，避免打滑/翻倒）
SLOPE_SLOW_GAIN = 2.0
SLOPE_SLOW_FLOOR = 0.35  # 最大减速到 35% 前进速度


def _slope_speed_factor(x, y, yaw):
    """根据当前位姿处地形坡度返回前进速度缩放因子（≤1）。"""
    try:
        fwd, lat = get_terrain().slopes(x, y, yaw)
    except Exception:
        return 1.0
    mag = math.hypot(fwd, lat)  # tan 坡度（≈坡度角弧度）
    return max(SLOPE_SLOW_FLOOR, 1.0 - SLOPE_SLOW_GAIN * mag)


# 每车 go-to 控制器状态（转向滞回），模块级持久；reset_rule_state() 在 env.reset() 时清空
_CTRL = {}


def reset_rule_state():
    """清空 go-to 控制器状态（每个回合开始时调用，避免跨回合污染滞回状态）。"""
    _CTRL.clear()


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ctrl_state(name):
    if name is None:
        return None
    st = _CTRL.get(name)
    if st is None:
        st = {"turning": False}
        _CTRL[name] = st
    return st


def go_to(robot_obs, tx, ty, name=None, arrive_radius=0.25,
          turn_enter=0.5, turn_exit=0.3):
    """去目标点的 go-to-goal 控制器（差分轮式，含原地转向 + 转向滞回 + 到达死区）。

    返回 drive=[vx_norm, wz_norm]。适用于 excavator / hauler（diff-drive）。
    """
    x, y, yaw = robot_obs[0], robot_obs[1], robot_obs[2]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    st = _ctrl_state(name)
    if dist < arrive_radius:
        if st is not None:
            st["turning"] = False
        return [0.0, 0.0]
    err = _norm_angle(math.atan2(dy, dx) - yaw)
    # 转向滞回：进入/退出阈值分离，避免在临界航向误差处振荡
    if st is not None:
        if not st["turning"] and abs(err) > turn_enter:
            st["turning"] = True
        elif st["turning"] and abs(err) < turn_exit:
            st["turning"] = False
    turning = bool(st["turning"]) if st is not None else (abs(err) > turn_enter)
    if turning:
        # 原地转向为主（增益较小，避免大角度转向过冲振荡）
        return [0.0, _clip(0.9 * err, -1.0, 1.0)]
    # 前进 + 微调航向（死区抑制抖动）；斜坡减速（坡度越大越慢，避免打滑/翻倒）
    vx = _clip(1.0 * dist, 0.12, 1.0)
    vx *= _slope_speed_factor(x, y, yaw)
    wz = 0.0 if abs(err) < 0.08 else _clip(0.8 * err, -1.0, 1.0)
    return [vx, wz]


def go_to_smooth(robot_obs, tx, ty, name=None, arrive_radius=0.25):
    """去目标点的 go-to-goal 控制器（平滑版，边走边转，带到达死区）。

    用于 Scout（四足步态）——其步态增益依赖前进速度（target=|vx|），vx=0 时无法原地转向，
    因此必须保持「前进 + 微调航向」的平滑运动，否则会原地卡死。
    """
    x, y, yaw = robot_obs[0], robot_obs[1], robot_obs[2]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist < arrive_radius:
        return [0.0, 0.0]
    err = _norm_angle(math.atan2(dy, dx) - yaw)
    vx = _clip(1.0 * dist, 0.0, 1.0) * max(0.0, math.cos(err)) ** 2
    vx *= _slope_speed_factor(x, y, yaw)
    wz = _clip(1.2 * err, -1.0, 1.0)
    return [vx, wz]


def rule_action(obs, dig_radius=0.6, load_radius=1.5, dump_radius=1.5,
                bucket_cap=4.0, bin_cap=10.0):
    """根据观测返回规则动作 dict。目标点用 obs["goals"]（承诺式分配），工具按角色状态机决定。"""
    goals = obs.get("goals", {})
    action = {}
    for name, o in obs.items():
        if name in ("resources", "detected", "depots", "depot", "goals", "roll_pitch",
                    "terrain"):
            continue
        role = ROBOT_ROLE.get(name, "scout")
        goal = goals.get(name)
        gx, gy = (float(goal[0]), float(goal[1])) if goal is not None else (o[0], o[1])
        d = math.hypot(o[0] - gx, o[1] - gy)
        cargo = float(o[5]) if len(o) >= 6 else 0.0

        if role == "excavator":
            if cargo >= bucket_cap - 0.05:
                # 满斗 → 去装载（goal=承诺配对的 hauler）
                if d < load_radius:
                    action[name] = {"drive": [0.0, 0.0], "tool": 2}
                else:
                    action[name] = {"drive": go_to(o, gx, gy, name=name), "tool": 0}
            else:
                # 挖斗未满 → 去挖（goal=承诺分配资源）
                if d < dig_radius:
                    action[name] = {"drive": [0.0, 0.0], "tool": 1}
                else:
                    action[name] = {"drive": go_to(o, gx, gy, name=name), "tool": 0}
        elif role == "hauler":
            if cargo > 0.05:
                if d < dump_radius:
                    action[name] = {"drive": [0.0, 0.0], "tool": 1}
                else:
                    action[name] = {"drive": go_to(o, gx, gy, name=name), "tool": 0}
            else:
                action[name] = {"drive": go_to(o, gx, gy, name=name), "tool": 0}
        else:  # scout：探索（goal=最近未发现资源），无工具（步态无法原地转向，用平滑 go-to）
            action[name] = {"drive": go_to_smooth(o, gx, gy, name=name)}
    return action
