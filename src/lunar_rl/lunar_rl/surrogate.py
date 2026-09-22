"""快速替身环境：与 lunar_env 同接口（reset/step/obs/action/reward），纯 Python 运动学模型。

支持任意数量机器人（同角色多实例）与多个 depot；资源带异构 value；deliver 奖励按 value 加权，
使「任务选择/目的地选择」成为真正的决策。
"""
import math

import numpy as np
import yaml

from lunar_rl.registry import set_robots
from lunar_rl.rewards import DEFAULT_WEIGHTS, goal_for, goal_key_for


class SurrogateEnv:
    """与 LunarEnv 接口一致的快速替身环境（纯 Python，无 ROS/Gazebo）。"""

    def __init__(self, scenario_file, max_lin=0.4, max_ang=0.8, stage="full",
                 domain_rand=False):
        self.stage = stage
        self.domain_rand = domain_rand
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = yaml.safe_load(f)
        ep = self.scenario["episode"]
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.control_horizon = float(ep["control_horizon_s"])
        self.max_steps = int(float(ep["max_time_s"]) / self.control_horizon)
        self.dig_radius = float(ep["dig_radius"])
        self.load_radius = float(ep["load_radius"])
        self.dump_radius = float(ep["dump_radius"])
        self.detect_radius = float(ep.get("detect_radius", 1.5))
        self.bucket_capacity = float(ep["bucket_capacity"])
        self.bin_capacity = float(ep["bin_capacity"])
        self.dig_rate = float(ep["dig_rate_per_s"])
        self.dump_rate = float(ep.get("dump_rate_per_s", self.dig_rate))
        self.robot_names = list(self.scenario["robots"].keys())
        # 机器人名 → 角色（scout/excavator/hauler，小写）
        self.role_of = {n: str(self.scenario["robots"][n].get("role", n)).lower()
                        for n in self.robot_names}
        set_robots(self.robot_names, self.role_of)  # 同步全局注册表
        self.w = dict(DEFAULT_WEIGHTS)
        # 资源池（变规模用）：value 缺省 1.0
        self.resource_pool = [
            {"id": int(r["id"]), "x": float(r["x"]), "y": float(r["y"]),
             "amount": float(r["amount"]), "value": float(r.get("value", 1.0))}
            for r in self.scenario["resources"]
        ]
        # 价值归一化系数：把 value 压到 [1/max_value, 1]，降低价值加权奖励的绝对幅度 → 降 Q 噪声
        self.value_norm = max((r["value"] for r in self.resource_pool), default=1.0)
        self.variable_resources = bool(ep.get("variable_resources", False))
        self.min_resources = int(ep.get("min_resources", len(self.resource_pool)))
        self.max_resources = int(ep.get("max_resources", len(self.resource_pool)))
        self.dynamic_respawn = bool(ep.get("dynamic_respawn", False))
        self.respawn_threshold = float(ep.get("respawn_threshold", 0.0))
        # 多个 depot：优先读 `depots` 列表，否则回退到单 `depot`
        if "depots" in self.scenario:
            self.depots = [(float(d["x"]), float(d["y"])) for d in self.scenario["depots"]]
        else:
            self.depots = [(float(self.scenario["depot"]["x"]), float(self.scenario["depot"]["y"]))]
        self.n_resources = len(self.resource_pool)
        self.reset()

    # ---------- 角色分组 ----------
    def _scouts(self):
        return [n for n in self.robot_names if self.role_of[n] == "scout"]

    def _excavators(self):
        return [n for n in self.robot_names if self.role_of[n] == "excavator"]

    def _haulers(self):
        return [n for n in self.robot_names if self.role_of[n] == "hauler"]

    # ---------- 与 lunar_env 一致的核心接口 ----------
    def reset(self):
        if self.variable_resources:
            k = int(np.random.randint(self.min_resources, self.max_resources + 1))
            k = min(k, len(self.resource_pool))
            idxs = list(np.random.choice(len(self.resource_pool), size=k, replace=False))
            self._active_pool = idxs
            self.resources = [dict(self.resource_pool[i], id=j) for j, i in enumerate(idxs)]
        else:
            self._active_pool = list(range(len(self.resource_pool)))
            self.resources = [dict(r) for r in self.resource_pool]
        self.n_resources = len(self.resources)
        self.detected = [False] * self.n_resources
        self.pose = {}
        self.vel = {}
        if self.domain_rand:
            self.speed_scale = {n: float(np.random.uniform(0.5, 0.9)) for n in self.robot_names}
            self.turn_scale = {n: float(np.random.uniform(0.5, 0.9)) for n in self.robot_names}
        else:
            self.speed_scale = {n: 1.0 for n in self.robot_names}
            self.turn_scale = {n: 1.0 for n in self.robot_names}
        for name, cfg in self.scenario["robots"].items():
            self.pose[name] = (cfg["x"], cfg["y"], cfg["yaw"])
            self.vel[name] = (0.0, 0.0)
        # 逐 excavator 铲斗（量 + 价值）、逐 hauler 车斗（量 + 价值）
        self.bucket = {n: 0.0 for n in self._excavators()}
        self.bucket_value = {n: 0.0 for n in self._excavators()}
        self.bin = {n: 0.0 for n in self._haulers()}
        self.bin_value = {n: 0.0 for n in self._haulers()}
        self.delivered = 0.0        # 交付量（吨）
        self.delivered_value = 0.0  # 交付价值（价值加权，优化目标）
        self.episode_reward = 0.0
        self.reward_breakdown = {"dig": 0.0, "load": 0.0, "deliver": 0.0, "detect": 0.0,
                                 "app": 0.0, "step": 0.0}
        self.app_by_agent = {n: 0.0 for n in self.robot_names}
        self.per_agent_reward = {n: 0.0 for n in self.robot_names}
        self.step_count = 0
        self.tool = {n: 0 for n in self.robot_names}
        self.prev_goal_dist = {}
        self.prev_goal_key = {}
        self._configure_stage()
        return self._obs(), self._info()

    def _configure_stage(self):
        if self.stage == "c0":
            for n in self._haulers():
                self.bin[n] = self.bin_capacity
                self.bin_value[n] = self.bin_capacity * 1.0
            for r in self.resources:
                r["amount"] = 0.0
        elif self.stage == "c1":
            for i, r in enumerate(self.resources):
                if i != 0:
                    r["amount"] = 0.0

    def step(self, action):
        self._apply_action(action)
        reward = self._update_task()
        self.step_count += 1
        self.episode_reward += reward
        terminated = self.step_count >= self.max_steps
        truncated = False
        return self._obs(), reward, terminated, truncated, self._info()

    # ---------- 运动学 ----------
    def _apply_action(self, action):
        dt = self.control_horizon
        for name in self.robot_names:
            a = action[name]
            drive = a["drive"]
            vx = float(np.clip(drive[0], -1.0, 1.0)) * self.max_lin * self.speed_scale[name]
            wz = float(np.clip(drive[1], -1.0, 1.0)) * self.max_ang * self.turn_scale[name]
            x, y, yaw = self.pose[name]
            yaw = yaw + wz * dt
            x = x + vx * math.cos(yaw) * dt
            y = y + vx * math.sin(yaw) * dt
            self.pose[name] = (x, y, yaw)
            self.vel[name] = (vx, wz)
            if isinstance(a, dict) and "tool" in a:
                self.tool[name] = int(a["tool"])

    # ---------- 任务逻辑 ----------
    def _update_task(self):
        dt = self.control_horizon
        rb = {"dig": 0.0, "load": 0.0, "deliver": 0.0, "detect": 0.0, "app": 0.0, "step": 0.0}

        # 探测：所有 scout（只更新发现状态，不再给离散一次性奖励——探索由 app 势函数驱动，
        # 去掉离散跳变以降 Q 噪声）
        for sc in self._scouts():
            p = self.pose[sc]
            for i, r in enumerate(self.resources):
                if not self.detected[i] and r["amount"] > 0.01 and \
                        self._dist(p, (r["x"], r["y"])) < self.detect_radius:
                    self.detected[i] = True

        # 挖掘：所有 excavator（价值加权 dig 奖励；bucket_value 存真实价值，奖励按 value_norm 归一化）
        for ex in self._excavators():
            if self.tool[ex] == 1 and self.bucket[ex] < self.bucket_capacity - 0.01:
                p = self.pose[ex]
                for r in self.resources:
                    if r["amount"] > 0 and self._dist(p, (r["x"], r["y"])) < self.dig_radius:
                        gain = min(self.dig_rate * dt, self.bucket_capacity - self.bucket[ex],
                                   r["amount"])
                        v = float(r.get("value", 1.0))
                        self.bucket[ex] += gain
                        self.bucket_value[ex] += v * gain
                        r["amount"] -= gain
                        rb["dig"] += self.w["dig"] * (v / self.value_norm) * gain
                        break  # 每步只挖一个资源

        # 装载：excavator(tool=2) → 最近的、有容量的 hauler
        for ex in self._excavators():
            if self.tool[ex] == 2 and self.bucket[ex] > 0.01:
                p = self.pose[ex]
                cands = [ha for ha in self._haulers()
                         if self.bin[ha] < self.bin_capacity - 0.01]
                if cands:
                    ha = min(cands, key=lambda h: self._dist(p, self.pose[h]))
                    if self._dist(p, self.pose[ha]) < self.load_radius:
                        transfer = min(self.bucket[ex], self.bin_capacity - self.bin[ha])
                        vd = self.bucket_value[ex] / max(self.bucket[ex], 1e-6)
                        self.bucket[ex] -= transfer
                        self.bucket_value[ex] -= vd * transfer
                        self.bin[ha] += transfer
                        self.bin_value[ha] += vd * transfer
                        rb["load"] += self.w["load"] * transfer

        # 卸载：所有 hauler → 就近 depot（价值加权 deliver 奖励，连续卸载）
        for ha in self._haulers():
            if self.tool[ha] == 1 and self.bin[ha] > 0.01:
                p = self.pose[ha]
                for dep in self.depots:
                    if self._dist(p, dep) < self.dump_radius:
                        dump = min(self.dump_rate * dt, self.bin[ha])
                        vd = self.bin_value[ha] / max(self.bin[ha], 1e-6)
                        self.bin[ha] -= dump
                        self.bin_value[ha] -= vd * dump
                        self.delivered += dump
                        self.delivered_value += vd * dump  # 真实价值（指标）
                        rb["deliver"] += self.w["deliver"] * (vd / self.value_norm) * dump  # 归一化奖励
                        break

        # 吸引奖励（势函数近似）
        app_by_agent = {n: 0.0 for n in self.robot_names}
        for name in self.robot_names:
            g = goal_for(self, name)
            if g is None:
                continue
            d = math.hypot(self.pose[name][0] - g[0], self.pose[name][1] - g[1])
            key = goal_key_for(self, name)
            prev = self.prev_goal_dist.get(name)
            prev_key = self.prev_goal_key.get(name)
            if prev is not None and prev_key == key:
                delta = self.w["app"] * (prev - d)
                rb["app"] += delta
                app_by_agent[name] = delta
            self.prev_goal_dist[name] = d
            self.prev_goal_key[name] = key

        rb["step"] = -self.w["step"] * dt
        reward = rb["dig"] + rb["load"] + rb["deliver"] + rb["detect"] + rb["app"] + rb["step"]
        for k, v in rb.items():
            self.reward_breakdown[k] += v
        for n in self.robot_names:
            self.app_by_agent[n] += app_by_agent[n]
        # 逐机器人奖励 shaping：团队 deliver + step + 自身 app + 角色专属贡献
        for name in self.robot_names:
            role = self.role_of[name]
            r_i = rb["deliver"] + rb["step"] + app_by_agent[name]
            if role == "scout":
                r_i += rb["detect"]
            elif role == "excavator":
                r_i += rb["dig"] + rb["load"]
            elif role == "hauler":
                r_i += rb["load"]
            self.per_agent_reward[name] = r_i

        # 动态重规划：资源耗尽后从池中重生新资源
        if self.dynamic_respawn:
            total = sum(r["amount"] for r in self.resources)
            if total < self.respawn_threshold and len(self.resources) < len(self.resource_pool):
                unused = [i for i in range(len(self.resource_pool)) if i not in self._active_pool]
                if unused:
                    i = int(np.random.choice(unused))
                    self.resources.append(dict(self.resource_pool[i], id=len(self.resources)))
                    self.detected.append(False)
                    self._active_pool.append(i)
                    self.n_resources = len(self.resources)
        return reward

    @staticmethod
    def _dist(pose, target):
        return math.hypot(pose[0] - target[0], pose[1] - target[1])

    # ---------- 观测 ----------
    def _obs(self):
        obs = {}
        for name in self.robot_names:
            cargo = 0.0
            if self.role_of[name] == "excavator":
                cargo = self.bucket[name]
            elif self.role_of[name] == "hauler":
                cargo = self.bin[name]
            obs[name] = self._robot_obs(name, with_cargo=(cargo is not None), cargo=cargo)
        res = np.zeros((self.n_resources, 4), dtype=np.float32)  # [x, y, amount, value]
        for i, r in enumerate(self.resources):
            res[i] = [r["x"], r["y"], r["amount"], float(r.get("value", 1.0))]
        obs["resources"] = res
        obs["detected"] = np.array([1.0 if d else 0.0 for d in self.detected], dtype=np.float32)
        obs["depots"] = np.array(self.depots, dtype=np.float32)      # (n_depots, 2)
        obs["depot"] = obs["depots"][0] if len(self.depots) > 0 else np.zeros(2, dtype=np.float32)
        obs["goals"] = {}
        for name in self.robot_names:
            g = goal_for(self, name)
            obs["goals"][name] = np.array(g if g is not None else [0.0, 0.0], dtype=np.float32)
        return obs

    def _robot_obs(self, name, with_cargo, cargo=0.0):
        x, y, yaw = self.pose[name]
        vx, wz = self.vel[name]
        if with_cargo:
            return np.array([x, y, yaw, vx, wz, cargo], dtype=np.float32)
        return np.array([x, y, yaw, vx, wz], dtype=np.float32)

    def _info(self):
        return {"delivered": float(self.delivered),
                "delivered_value": float(self.delivered_value),
                "episode_reward": float(self.episode_reward),
                "reward_breakdown": dict(self.reward_breakdown),
                "per_agent_reward": dict(self.per_agent_reward),
                "step": int(self.step_count)}
