"""规则基线策略：就近拍卖式任务分配 + 会合/装卸状态机。

将 obs 映射为 env 的动作 dict（drive + tool）。Stage 2：资源位置未知，Scout 负责探索
（朝最近「未发现」资源移动，靠探测发现），Excavator 只用「已发现」资源挖掘。

动作语义（与 lunar_env 一致）：
    excavator tool: 0 idle / 1 dig / 2 load
    hauler    tool: 0 idle / 1 dump
"""
import math

import numpy as np


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def go_to(robot_obs, tx, ty):
    """去目标点的 go-to-goal 控制器，返回 drive=[vx_norm, wz_norm]。"""
    x, y, yaw = robot_obs[0], robot_obs[1], robot_obs[2]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist < 0.08:
        return [0.0, 0.0]
    err = _norm_angle(math.atan2(dy, dx) - yaw)
    if abs(err) > 0.5:
        # 转向为主（增益较小，避免大角度转向过冲振荡）
        return [0.0, _clip(0.9 * err, -1.0, 1.0)]
    # 前进 + 微调航向（死区抑制抖动）
    vx = _clip(1.0 * dist, 0.12, 1.0)
    wz = 0.0 if abs(err) < 0.08 else _clip(0.8 * err, -1.0, 1.0)
    return [vx, wz]


def _nearest_resource(resources, x, y, min_amount=0.01):
    cand = [r for r in resources if r[2] > min_amount]
    if not cand:
        return None
    return min(cand, key=lambda r: (r[0] - x) ** 2 + (r[1] - y) ** 2)


def rule_action(obs, dig_radius=0.6, load_radius=0.7, dump_radius=0.7,
                bucket_cap=4.0, bin_cap=10.0):
    """根据观测返回规则动作 dict（Stage 2：Scout 探索未发现，Excavator 只用已发现）。"""
    resources = np.asarray(obs["resources"], dtype=np.float32)  # (n,3) = [x, y, amount]
    depot = obs["depot"]          # [x, y]
    ex = obs["excavator"]         # [x, y, yaw, vx, wz, bucket]
    ha = obs["hauler"]            # [x, y, yaw, vx, wz, bin]
    sc = obs["scout"]             # [x, y, yaw, vx, wz]

    # 已发现 / 未发现资源子集（无 detected 字段时视为全部已知）
    det = obs.get("detected", None)
    if det is None:
        det_mask = np.ones(len(resources), dtype=bool)
    else:
        det_mask = np.asarray(det) > 0.5
    det_res = resources[det_mask]
    undet_res = resources[~det_mask]

    ex_bucket = ex[5]
    ha_bin = ha[5]

    action = {
        "scout": {"drive": [0.0, 0.0]},
        "excavator": {"drive": [0.0, 0.0], "tool": 0},
        "hauler": {"drive": [0.0, 0.0], "tool": 0},
    }

    # ---- Scout：探索，朝最近「未发现」有量资源移动；都发现后待命 ----
    r = _nearest_resource(undet_res, sc[0], sc[1])
    if r is not None:
        action["scout"] = {"drive": go_to(sc, r[0], r[1])}

    # ---- Excavator：挖掘「已发现」资源（铲斗不满且有已发现资源）→ 会合装载 ----
    r = _nearest_resource(det_res, ex[0], ex[1])
    if ex_bucket < bucket_cap - 0.05 and r is not None:
        d = math.hypot(r[0] - ex[0], r[1] - ex[1])
        if d < dig_radius:
            action["excavator"] = {"drive": [0.0, 0.0], "tool": 1}  # dig
        else:
            action["excavator"] = {"drive": go_to(ex, r[0], r[1]), "tool": 0}
    else:
        d = math.hypot(ha[0] - ex[0], ha[1] - ex[1])
        if d < load_radius:
            action["excavator"] = {"drive": [0.0, 0.0], "tool": 2}  # load
        else:
            action["excavator"] = {"drive": go_to(ex, ha[0], ha[1]), "tool": 0}

    # ---- Hauler：有货→去卸载；空→会合等待装载 ----
    if ha_bin > 0.05:
        d = math.hypot(depot[0] - ha[0], depot[1] - ha[1])
        if d < dump_radius:
            action["hauler"] = {"drive": [0.0, 0.0], "tool": 1}  # dump
        else:
            action["hauler"] = {"drive": go_to(ha, depot[0], depot[1]), "tool": 0}
    else:
        d = math.hypot(ex[0] - ha[0], ex[1] - ha[1])
        if d > load_radius:
            action["hauler"] = {"drive": go_to(ha, ex[0], ex[1]), "tool": 0}
        else:
            action["hauler"] = {"drive": [0.0, 0.0], "tool": 0}  # 等待装载

    return action
