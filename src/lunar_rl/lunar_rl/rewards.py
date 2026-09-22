"""奖励塑形 + 目标/任务分配：交付 + 探测 + 吸引（势函数近似）+ 步进代价，权重可配置。

支持任意数量机器人（同角色多实例）与多个 depot；资源带异构 value；excavator 用「贪婪拍卖」
分配到单位距离价值最高的已发现资源（作为规则基线的任务分配）。
"""
import math

from lunar_rl.registry import ROBOT_ROLE


def _is_detected(env, i):
    det = getattr(env, "detected", None)
    if det is None:
        return True
    return bool(det[i])


def _dist(env, name, x, y):
    return math.hypot(env.pose[name][0] - x, env.pose[name][1] - y)


def _dist2(env, a, b):
    return math.hypot(env.pose[a][0] - env.pose[b][0], env.pose[a][1] - env.pose[b][1])


def detected_resources(env):
    return [r for i, r in enumerate(env.resources)
            if r["amount"] > 0.01 and _is_detected(env, i)]


def undetected_resources(env):
    return [r for i, r in enumerate(env.resources)
            if r["amount"] > 0.01 and not _is_detected(env, i)]


def nearest_detected_resource(env, name):
    cand = detected_resources(env)
    if not cand:
        return None
    return min(cand, key=lambda r: _dist(env, name, r["x"], r["y"]))


def nearest_undetected_resource(env, name):
    cand = undetected_resources(env)
    if not cand:
        return None
    return min(cand, key=lambda r: _dist(env, name, r["x"], r["y"]))


def nearest_depot(env, name):
    return min(env.depots, key=lambda d: math.hypot(env.pose[name][0] - d[0],
                                                    env.pose[name][1] - d[1]))


def nearest_hauler(env, name):
    cand = [n for n in env.robot_names
            if ROBOT_ROLE[n] == "hauler" and env.bin[n] < env.bin_capacity - 0.01]
    if not cand:
        return None
    return min(cand, key=lambda n: _dist2(env, name, n))


def nearest_loaded_excavator(env, name):
    cand = [n for n in env.robot_names
            if ROBOT_ROLE[n] == "excavator" and env.bucket[n] > 0.01]
    if not cand:
        return None
    return min(cand, key=lambda n: _dist2(env, name, n))


def assign_excavators(env):
    """贪婪拍卖：把已发现资源按「value/(1+distance)」互斥分配给各 excavator（挖斗未满者）。"""
    exs = [n for n in env.robot_names
           if ROBOT_ROLE[n] == "excavator" and env.bucket[n] < env.bucket_capacity - 0.05]
    available = list(detected_resources(env))
    assignment = {}
    for ex in exs:
        best, best_score = None, -1.0
        for r in available:
            d = _dist(env, ex, r["x"], r["y"])
            score = float(r.get("value", 1.0)) / (1.0 + d)
            if score > best_score:
                best_score, best = score, r
        if best is not None:
            assignment[ex] = best
            available.remove(best)
    return assignment


def goal_for(env, name):
    """返回机器人 name 的当前目标点 (x, y)。"""
    role = ROBOT_ROLE[name]
    if role == "scout":
        r = nearest_undetected_resource(env, name)
        return (r["x"], r["y"]) if r is not None else env.pose[name][:2]
    if role == "excavator":
        if env.bucket[name] >= env.bucket_capacity - 0.05:
            ha = nearest_hauler(env, name)
            return env.pose[ha][:2] if ha is not None else env.pose[name][:2]
        r = assign_excavators(env).get(name)
        if r is None:
            r = nearest_detected_resource(env, name)
        return (r["x"], r["y"]) if r is not None else env.pose[name][:2]
    if role == "hauler":
        if env.bin[name] > 0.05:
            d = nearest_depot(env, name)
            return (d[0], d[1])
        ex = nearest_loaded_excavator(env, name)
        return env.pose[ex][:2] if ex is not None else env.pose[name][:2]
    return env.pose[name][:2]


def goal_key_for(env, name):
    """目标身份键（目标切换时重置势函数基线，避免虚假势能跳跃）。"""
    role = ROBOT_ROLE[name]
    if role == "scout":
        r = nearest_undetected_resource(env, name)
        return f"res{r['id']}" if r is not None else "idle"
    if role == "excavator":
        if env.bucket[name] >= env.bucket_capacity - 0.05:
            return "hauler"
        r = assign_excavators(env).get(name) or nearest_detected_resource(env, name)
        return f"res{r['id']}" if r is not None else "none"
    if role == "hauler":
        if env.bin[name] > 0.05:
            d = nearest_depot(env, name)
            return f"depot{env.depots.index(d)}"
        ex = nearest_loaded_excavator(env, name)
        return ex if ex is not None else "idle"
    return "idle"


DEFAULT_WEIGHTS = {"deliver": 1.0, "dig": 0.5, "load": 0.5, "detect": 1.0,
                   "app": 1.0, "step": 0.02}
