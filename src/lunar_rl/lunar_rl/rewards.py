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


def _find_resource(env, rid):
    """按 id 查找资源，返回 (index, resource) 或 (None, None)。"""
    for i, r in enumerate(env.resources):
        if r["id"] == rid:
            return i, r
    return None, None


def _score(env, name, r):
    return float(r.get("value", 1.0)) / (1.0 + _dist(env, name, r["x"], r["y"]))


class TaskAllocator:
    """承诺式任务分配器：给每台车一个任务序列，仅「任务完成/资源耗尽/队友失联」时重分配（带滞回）。

    消除旧 `goal_for` 每步重算 nearest/auction 造成的目标抖动，是「行为目的性」的核心。
    承诺序列：
      scout      → 未发现资源（逐个探测，检测到即切下一个）
      excavator  → dig(资源，滞回持有) ↔ load(固定配对 hauler)
      hauler     → serve(固定配对 excavator) ↔ dump(承诺 depot)
    """

    def __init__(self, hysteresis=0.3):
        self.hysteresis = hysteresis
        self.commit = {}       # name -> 当前阶段承诺（goal_key_for 用）
        self.dig_commit = {}   # excavator -> 跨装载循环持久持有的挖掘资源 id
        self.depot_commit = {}  # hauler -> 跨卸载循环持久持有的 depot

    def reset(self):
        self.commit.clear()
        self.dig_commit.clear()
        self.depot_commit.clear()

    # ---- 固定配对（人工调度式：按索引 1:1 配对）----
    def _pair_hauler(self, env, excavator):
        exs = sorted(env._excavators())
        has = sorted(env._haulers())
        if excavator in exs and len(has) == len(exs) and len(has) > 0:
            return has[exs.index(excavator)]
        return nearest_hauler(env, excavator)

    def _pair_excavator(self, env, hauler):
        exs = sorted(env._excavators())
        has = sorted(env._haulers())
        if hauler in has and len(has) == len(exs) and len(exs) > 0:
            return exs[has.index(hauler)]
        return nearest_loaded_excavator(env, hauler)

    def goal_for(self, env, name):
        role = ROBOT_ROLE[name]
        if role == "scout":
            return self._scout_goal(env, name)
        if role == "excavator":
            return self._excavator_goal(env, name)
        if role == "hauler":
            return self._hauler_goal(env, name)
        return env.pose[name][:2]

    def goal_key_for(self, env, name):
        c = self.commit.get(name)
        if c is None:
            return "none"
        k = c.get("kind")
        if k in ("explore", "dig"):
            return f"res{c['res_id']}"
        if k == "load":
            return f"load:{c['hauler']}"
        if k == "dump":
            return f"depot{c['depot_id']}"
        if k == "serve":
            return f"serve:{c['excavator']}"
        return "idle"

    def _scout_goal(self, env, name):
        undet = undetected_resources(env)
        c = self.commit.get(name)
        if c and c.get("kind") == "explore":
            rid = c["res_id"]
            r = next((rr for rr in undet if rr["id"] == rid), None)
            if r is not None:
                return (r["x"], r["y"])
        if not undet:
            self.commit[name] = {"kind": "idle"}
            return env.pose[name][:2]
        r = min(undet, key=lambda rr: _dist(env, name, rr["x"], rr["y"]))
        self.commit[name] = {"kind": "explore", "res_id": r["id"]}
        return (r["x"], r["y"])

    def _excavator_goal(self, env, name):
        # 满斗 → 装载阶段（固定配对 hauler）
        if env.bucket[name] >= env.bucket_capacity - 0.05:
            ha = self._pair_hauler(env, name)
            if ha is None:
                return env.pose[name][:2]
            self.commit[name] = {"kind": "load", "hauler": ha}
            return env.pose[ha][:2]

        # 未满 → 挖掘阶段：跨装载循环持久持有已承诺资源（耗尽/未发现才放弃；换资源需滞回）
        rid = self.dig_commit.get(name)
        if rid is not None:
            i, r = _find_resource(env, rid)
            if r is not None and r["amount"] > 0.01 and i is not None and _is_detected(env, i):
                newr = assign_excavators(env).get(name)
                if newr is not None and newr["id"] != rid:
                    if _score(env, name, newr) > _score(env, name, r) * (1.0 + self.hysteresis):
                        rid = newr["id"]
                        self.dig_commit[name] = rid
                        self.commit[name] = {"kind": "dig", "res_id": rid}
                        return (newr["x"], newr["y"])
                self.commit[name] = {"kind": "dig", "res_id": rid}
                return (r["x"], r["y"])

        r = assign_excavators(env).get(name) or nearest_detected_resource(env, name)
        if r is None:
            self.commit[name] = {"kind": "idle"}
            return env.pose[name][:2]
        self.dig_commit[name] = r["id"]
        self.commit[name] = {"kind": "dig", "res_id": r["id"]}
        return (r["x"], r["y"])

    def _hauler_goal(self, env, name):
        # 有货 → 卸载阶段（跨卸载循环持久持有 depot，换 depot 需滞回）
        if env.bin[name] > 0.05:
            c = self.commit.get(name)
            if c and c.get("kind") == "dump":
                return c["depot"]
            d = self.depot_commit.get(name)
            if d is None:
                d = nearest_depot(env, name)
                self.depot_commit[name] = d
            else:
                nd = nearest_depot(env, name)
                if _dist(env, name, nd[0], nd[1]) < _dist(env, name, d[0], d[1]) * (1.0 - self.hysteresis):
                    self.depot_commit[name] = nd
                    d = nd
            dpos = (d[0], d[1])
            did = env.depots.index(d) if d in env.depots else 0
            self.commit[name] = {"kind": "dump", "depot": dpos, "depot_id": did}
            return dpos
        # 空车 → 服务阶段（固定配对 excavator）
        ex = self._pair_excavator(env, name)
        if ex is None:
            self.commit[name] = {"kind": "idle"}
            return env.pose[name][:2]
        self.commit[name] = {"kind": "serve", "excavator": ex}
        return env.pose[ex][:2]


def goal_for(env, name):
    """返回机器人 name 的当前目标点 (x, y)。有 allocator 时走承诺式分配，否则走旧贪婪。"""
    alloc = getattr(env, "allocator", None)
    if alloc is not None:
        return alloc.goal_for(env, name)

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
    alloc = getattr(env, "allocator", None)
    if alloc is not None:
        return alloc.goal_key_for(env, name)

    role = ROBOT_ROLE[name]
    if role == "scout":
        r = nearest_undetected_resource(env, name)
        return f"res{r['id']}" if r is not None else "idle"
    if role == "excavator":
        if env.bucket[name] >= env.bucket_capacity - 0.05:
            ha = nearest_hauler(env, name)
            return f"load:{ha}" if ha is not None else "idle"
        r = assign_excavators(env).get(name) or nearest_detected_resource(env, name)
        return f"res{r['id']}" if r is not None else "none"
    if role == "hauler":
        if env.bin[name] > 0.05:
            d = nearest_depot(env, name)
            return f"depot{env.depots.index(d)}"
        ex = nearest_loaded_excavator(env, name)
        return f"serve:{ex}" if ex is not None else "idle"
    return "idle"


DEFAULT_WEIGHTS = {"deliver": 1.0, "dig": 0.5, "load": 0.5, "detect": 1.0,
                   "app": 1.0, "step": 0.02, "slope": 0.5}

# 陡坡惩罚阈值（坡度 tan）：超过该值才惩罚，让策略学会避开陡坡但不必绕开所有缓坡
SLOPE_PENALTY_TAN = math.tan(math.radians(10.0))
# roll/pitch 终止阈值（rad）：超过判翻倒终止。取 25° 以区分「真实翻倒」与「腿式车瞬时晃动」，
# 避免单帧晃动误终止（验收指标是 max roll/pitch < 15°，另行报告）。
TIP_LIMIT_RAD = math.radians(25.0)
