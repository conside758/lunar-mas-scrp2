"""LunarEnv：包装 ROS2 + Gazebo 的 Gymnasium 环境（通用 N 机器人 / 多 depot / 异构价值）。

- reset() 重置机器人位姿与任务状态（资源量、铲斗/车斗载量、交付量/价值）。
- step() 下发各机器人动作（驱动 + 工具），按 control_horizon 推进仿真，返回 (obs, reward, done, info)。
- 任务语义：Scout 探测资源；Excavator 在资源区"挖掘"累加铲斗载量，在 Hauler 旁"装载"转移；
  Hauler 在卸载点"倾卸"产生交付（按资源 value 加权）。

与替身环境 `lunar_rl.surrogate.SurrogateEnv` 保持接口/观测一致（角色分组、多 depot、资源 (n,4)=[x,y,amount,value]、
逐 excavator 铲斗、逐 hauler 车斗、价值加权 delivered_value），使 `lunar_rl.obs.encode_obs_vars` 与
`lunar_rl.rewards.goal_for` 可直接复用，保证 Gazebo 观测与训练替身同分布（避免策略输入 OOD）。

action 结构（gymnasium.spaces.Dict，键为全部机器人名）：
    {name: {'drive': [vx_norm, wz_norm]},           # scout
     name: {'drive': [vx_norm, wz_norm], 'tool': int}}  # excavator tool∈{0 idle,1 dig,2 load} / hauler tool∈{0 idle,1 dump}
    vx_norm/wz_norm ∈ [-1,1]。
"""
import math
import time

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Trigger

from lunar_rl.registry import set_robots, ROBOT_ROLE
from lunar_rl.rewards import (DEFAULT_WEIGHTS, SLOPE_PENALTY_TAN, TIP_LIMIT_RAD,
                              TaskAllocator, goal_for, goal_key_for)
from lunar_rl.rule_policy import reset_rule_state
from lunar_rl.terrain import get_terrain


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _roll_pitch_yaw_from_quat(q):
    """从四元数计算 roll/pitch/yaw（ZYX 欧拉角），用于倾倒监测。"""
    w, x, y, z = q.w, q.x, q.y, q.z
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


class LunarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_file, node_name="lunar_env"):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()

        self.scenario_file = scenario_file
        self.scenario = self._load_scenario()

        ep = self.scenario["episode"]
        self.max_lin = 0.4
        self.max_ang = 0.5  # 降低最大转向角速度，配合加速度限制减小转向倾倒力矩
        self.control_horizon = float(ep["control_horizon_s"])
        self.max_steps = int(float(ep["max_time_s"]) / self.control_horizon)
        self.dig_radius = float(ep["dig_radius"])
        self.load_radius = float(ep["load_radius"])
        self.dump_radius = float(ep["dump_radius"])
        self.detect_radius = float(ep.get("detect_radius", 1.5))  # Scout 探测半径
        self.bucket_capacity = float(ep["bucket_capacity"])
        self.bin_capacity = float(ep["bin_capacity"])
        self.dig_rate = float(ep["dig_rate_per_s"])
        self.dump_rate = float(ep.get("dump_rate_per_s", self.dig_rate))

        self.robot_names = list(self.scenario["robots"].keys())
        self.role_of = {n: str(self.scenario["robots"][n].get("role", n)).lower()
                        for n in self.robot_names}
        set_robots(self.robot_names, self.role_of)  # 同步全局注册表（obs/rule_policy 依赖）
        self.entity_of = {n: self.scenario["robots"][n].get("entity", n)
                          for n in self.robot_names}

        self.w = dict(DEFAULT_WEIGHTS)
        self.value_norm = 1.0  # 由 reset 时按资源最大 value 更新
        self.terrain = get_terrain()
        self.allocator = TaskAllocator()  # 承诺式分配（与替身一致）

        self._init_spaces()

        # ROS
        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self.cmd_pubs = {}
        for name in self.robot_names:
            cfg = self.scenario["robots"][name]
            topic = cfg.get("cmd_vel_topic", f"/{cfg['namespace']}/cmd_vel")
            self.cmd_pubs[name] = self.node.create_publisher(Twist, topic, 10)

        # 每台 excavator 一个机械臂命令话题；每台 hauler 一个倾卸命令话题
        self.arm_pubs = {}
        for name in self._excavators():
            ns = self.scenario["robots"][name]["namespace"]
            self.arm_pubs[name] = self.node.create_publisher(
                Float64MultiArray, f"/{ns}/arm_controller/commands", 10)
        self.dump_pubs = {}
        for name in self._haulers():
            ns = self.scenario["robots"][name]["namespace"]
            self.dump_pubs[name] = self.node.create_publisher(
                Float64MultiArray, f"/{ns}/dump_controller/commands", 10)

        # 缓存 link_states 中的位姿/速度（按唯一 entity 名匹配）
        self.pose = {}
        self.vel = {}
        self.roll_pitch = {}
        for name, cfg in self.scenario["robots"].items():
            self.pose[name] = (cfg["x"], cfg["y"], cfg["yaw"])
            self.vel[name] = (0.0, 0.0)
            self.roll_pitch[name] = (0.0, 0.0)
        self.node.create_subscription(LinkStates, "/gazebo/link_states", self._link_cb, 10)

        self.set_state_client = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.reset_scenario_client = self.node.create_client(Trigger, "/reset_scenario")

        self.reset()

    # ---------- 角色分组 ----------
    def _scouts(self):
        return [n for n in self.robot_names if self.role_of[n] == "scout"]

    def _excavators(self):
        return [n for n in self.robot_names if self.role_of[n] == "excavator"]

    def _haulers(self):
        return [n for n in self.robot_names if self.role_of[n] == "hauler"]

    # ---------- 初始化 ----------
    def _load_scenario(self):
        with open(self.scenario_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_spaces(self):
        self.action_space = spaces.Dict({
            n: spaces.Dict({
                "drive": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                **({"tool": spaces.Discrete(3)} if self.role_of[n] == "excavator" else
                   ({"tool": spaces.Discrete(2)} if self.role_of[n] == "hauler" else {})),
            })
            for n in self.robot_names
        })
        inf = np.finfo(np.float32).max
        self.observation_space = spaces.Dict({
            **{n: spaces.Box(low=-inf, high=inf, shape=(6,), dtype=np.float32)
               for n in self._excavators() + self._haulers()},
            **{n: spaces.Box(low=-inf, high=inf, shape=(5,), dtype=np.float32)
               for n in self._scouts()},
            "resources": spaces.Box(low=-inf, high=inf,
                                    shape=(len(self.scenario["resources"]), 4), dtype=np.float32),
            "detected": spaces.Box(low=0.0, high=1.0,
                                   shape=(len(self.scenario["resources"]),), dtype=np.float32),
            "depots": spaces.Box(low=-inf, high=inf, shape=(len(self._depot_list()), 2),
                                 dtype=np.float32),
            "depot": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
            "roll_pitch": spaces.Dict({
                n: spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32)
                for n in self.robot_names
            }),
            "goals": spaces.Dict({
                n: spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32)
                for n in self.robot_names
            }),
        })

    def _depot_list(self):
        if "depots" in self.scenario:
            return self.scenario["depots"]
        return [self.scenario["depot"]]

    # ---------- ROS 回调 ----------
    def _link_cb(self, msg: LinkStates):
        for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
            for rname, entity in self.entity_of.items():
                if name == f"{entity}::base_link":
                    roll, pitch, yaw = _roll_pitch_yaw_from_quat(pose.orientation)
                    self.pose[rname] = (pose.position.x, pose.position.y, yaw)
                    self.roll_pitch[rname] = (roll, pitch)  # 倾倒监测
                    self.vel[rname] = (twist.linear.x, twist.angular.z)

    # ---------- 工具函数 ----------
    def _spin(self, duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.01)

    def _spin_until(self, predicate, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.01)
            if predicate():
                return True
        return False

    def _publish_cmd(self, name, vx, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pubs[name].publish(msg)

    def _set_entity_state(self, entity, x, y, z, yaw):
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            return False
        req = SetEntityState.Request()
        req.state.name = entity
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = "world"
        future = self.set_state_client.call_async(req)
        return self._spin_until(lambda: future.done())

    def _call_reset_scenario(self):
        if not self.reset_scenario_client.wait_for_service(timeout_sec=1.0):
            return False
        future = self.reset_scenario_client.call_async(Trigger.Request())
        self._spin_until(lambda: future.done())
        return True

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scenario = self._load_scenario()

        # 资源池（带 value）
        self.resources = [
            {"id": int(r["id"]), "x": float(r["x"]), "y": float(r["y"]),
             "amount": float(r["amount"]), "value": float(r.get("value", 1.0))}
            for r in self.scenario["resources"]
        ]
        self.n_resources = len(self.resources)
        # 多个 depot：优先 `depots` 列表，否则回退单 `depot`
        if "depots" in self.scenario:
            self.depots = [(float(d["x"]), float(d["y"])) for d in self.scenario["depots"]]
        else:
            self.depots = [(float(self.scenario["depot"]["x"]),
                            float(self.scenario["depot"]["y"]))]

        self.detected = [False] * self.n_resources
        self.value_norm = max((r["value"] for r in self.resources), default=1.0)

        # 逐 excavator 铲斗（量 + 价值）、逐 hauler 车斗（量 + 价值）
        self.bucket = {n: 0.0 for n in self._excavators()}
        self.bucket_value = {n: 0.0 for n in self._excavators()}
        self.bin = {n: 0.0 for n in self._haulers()}
        self.bin_value = {n: 0.0 for n in self._haulers()}
        self.delivered = 0.0        # 交付量（吨）
        self.delivered_value = 0.0  # 交付价值（价值加权，优化目标）
        self.episode_reward = 0.0
        self.reward_breakdown = {"dig": 0.0, "load": 0.0, "deliver": 0.0, "detect": 0.0,
                                 "app": 0.0, "step": 0.0, "slope": 0.0}
        self.app_by_agent = {n: 0.0 for n in self.robot_names}
        self.per_agent_reward = {n: 0.0 for n in self.robot_names}
        self.step_count = 0
        self.tool = {n: 0 for n in self.robot_names}
        self.prev_goal_dist = {}
        self.prev_goal_key = {}
        self.tip_streak = {n: 0 for n in self.robot_names}
        self.allocator.reset()
        reset_rule_state()

        # 复位机器人位姿
        for name, cfg in self.scenario["robots"].items():
            self._set_entity_state(cfg["entity"], cfg["x"], cfg["y"], cfg["z"], cfg["yaw"])
            self._publish_cmd(name, 0.0, 0.0)

        # 复位工具姿态（每台 excavator 机械臂 / 每台 hauler 倾卸斗）
        for name in self._excavators():
            self.arm_pubs[name].publish(Float64MultiArray(data=[0.2, -0.4, 0.2]))
        for name in self._haulers():
            self.dump_pubs[name].publish(Float64MultiArray(data=[0.0]))

        self._call_reset_scenario()
        self._spin(2.0)  # 让 teleport 后掉落的机器人充分沉降，避免首步误判瞬时 roll/pitch
        return self._obs(), self._info()

    def step(self, action):
        self._apply_action(action)
        self._spin(self.control_horizon)
        reward = self._update_task()
        self.step_count += 1
        self.episode_reward += reward
        terminated = self.step_count >= self.max_steps or self._tip_check()
        truncated = False
        return self._obs(), reward, terminated, truncated, self._info()

    def _tip_check(self):
        """|roll| 或 |pitch| 超阈值判翻倒终止（真实 link_states 四元数）。

        前 3 步为沉降宽限期（跳过），之后需「连续 2 步」超阈值才判翻倒，
        避免腿式车瞬时晃动误终止。
        """
        if self.step_count < 3:
            return False
        for n in self.robot_names:
            roll, pitch = self.roll_pitch.get(n, (0.0, 0.0))
            if abs(roll) > TIP_LIMIT_RAD or abs(pitch) > TIP_LIMIT_RAD:
                self.tip_streak[n] = self.tip_streak.get(n, 0) + 1
            else:
                self.tip_streak[n] = 0
            if self.tip_streak.get(n, 0) >= 2:
                return True
        return False

    def _collision_avoidance(self, action, min_dist=0.7, gain=0.6):
        """互斥势场避碰：邻车在前方且中心距 < min_dist 时，减速并转向远离。

        作用于归一化 drive（[-1,1]），对规则与学习策略统一生效；min_dist(0.7m) 小于
        装载半径(1.5m)，不会妨碍会合装载，只防止车体穿透/碰撞倾倒。
        """
        for name in self.robot_names:
            if name not in action:
                continue
            x, y, yaw = self.pose[name]
            rep_vx = 0.0
            rep_wz = 0.0
            for other in self.robot_names:
                if other == name:
                    continue
                ox, oy = self.pose[other][0], self.pose[other][1]
                dx = ox - x
                dy = oy - y
                d = math.hypot(dx, dy)
                if d < min_dist and d > 1e-3:
                    # 邻车方向转到车体系
                    c, s = math.cos(yaw), math.sin(yaw)
                    bx = dx * c + dy * s   # 前向分量
                    by = -dx * s + dy * c  # 左向分量
                    if bx > -0.15:  # 邻车在侧前/前方
                        rep_vx -= gain * max(0.0, bx) / min_dist   # 前方→减速
                        rep_wz -= gain * by / min_dist             # 左邻→右转远离
            drive = action[name]["drive"]
            action[name]["drive"] = [
                float(np.clip(drive[0] + rep_vx, -1.0, 1.0)),
                float(np.clip(drive[1] + rep_wz, -1.0, 1.0)),
            ]

    def _apply_action(self, action):
        # 车距排斥避碰（对规则/学习策略统一生效）：邻车在前方且过近时减速 + 转向远离
        self._collision_avoidance(action)

        for name in self.robot_names:
            a = action[name]
            drive = a["drive"]
            vx = float(np.clip(drive[0], -1.0, 1.0)) * self.max_lin
            wz = float(np.clip(drive[1], -1.0, 1.0)) * self.max_ang
            self._publish_cmd(name, vx, wz)
            if isinstance(a, dict) and "tool" in a:
                self.tool[name] = int(a["tool"])

        # 每台 excavator 机械臂姿态（idle / dig / load）
        for name in self._excavators():
            t = self.tool[name]
            if t == 1:
                arm = [0.9, -1.2, 0.4]
            elif t == 2:
                arm = [-0.9, -0.5, 0.5]
            else:
                arm = [0.2, -0.4, 0.2]
            self.arm_pubs[name].publish(Float64MultiArray(data=arm))

        # 每台 hauler 倾卸
        for name in self._haulers():
            self.dump_pubs[name].publish(Float64MultiArray(
                data=[-0.8 if self.tool[name] == 1 else 0.0]))

    def _update_task(self):
        dt = self.control_horizon
        rb = {"dig": 0.0, "load": 0.0, "deliver": 0.0, "detect": 0.0, "app": 0.0,
              "step": 0.0, "slope": 0.0}

        # 探测：所有 scout（只更新发现状态，不给离散一次性奖励）
        for sc in self._scouts():
            p = self.pose[sc]
            for i, r in enumerate(self.resources):
                if not self.detected[i] and r["amount"] > 0.01 and \
                        self._dist(p, (r["x"], r["y"])) < self.detect_radius:
                    self.detected[i] = True

        # 挖掘：所有 excavator（价值加权 dig 奖励，奖励按 value_norm 归一化）
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
                        self.delivered_value += vd * dump
                        rb["deliver"] += self.w["deliver"] * (vd / self.value_norm) * dump
                        break

        # 吸引奖励（势函数近似，与替身环境一致）
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
        # 陡坡惩罚：坡度超过阈值才罚（真实 roll/pitch 反推坡度，与替身量纲一致）
        slope_by_agent = {n: 0.0 for n in self.robot_names}
        for name in self.robot_names:
            roll, pitch = self.roll_pitch.get(name, (0.0, 0.0))
            mag = math.hypot(math.tan(roll), math.tan(pitch))
            if mag > SLOPE_PENALTY_TAN:
                slope_by_agent[name] = -self.w["slope"] * (mag - SLOPE_PENALTY_TAN) * dt
                rb["slope"] += slope_by_agent[name]
        reward = (rb["dig"] + rb["load"] + rb["deliver"] + rb["detect"] + rb["app"]
                  + rb["step"] + rb["slope"])
        for k, v in rb.items():
            self.reward_breakdown[k] += v
        for n in self.robot_names:
            self.app_by_agent[n] += app_by_agent[n]
        # 逐机器人奖励 shaping：团队 deliver + step + 自身 app + 自身坡度惩罚 + 角色专属贡献
        for name in self.robot_names:
            role = self.role_of[name]
            r_i = rb["deliver"] + rb["step"] + app_by_agent[name] + slope_by_agent[name]
            if role == "scout":
                r_i += rb["detect"]
            elif role == "excavator":
                r_i += rb["dig"] + rb["load"]
            elif role == "hauler":
                r_i += rb["load"]
            self.per_agent_reward[name] = r_i
        return reward

    @staticmethod
    def _dist(pose, target):
        return math.hypot(pose[0] - target[0], pose[1] - target[1])

    def _obs(self):
        obs = {}
        for name in self.robot_names:
            cargo = 0.0
            if self.role_of[name] == "excavator":
                cargo = self.bucket[name]
            elif self.role_of[name] == "hauler":
                cargo = self.bin[name]
            x, y, yaw = self.pose[name]
            vx, wz = self.vel[name]
            if self.role_of[name] in ("excavator", "hauler"):
                obs[name] = np.array([x, y, yaw, vx, wz, cargo], dtype=np.float32)
            else:
                obs[name] = np.array([x, y, yaw, vx, wz], dtype=np.float32)

        # 资源 (n,4) = [x, y, amount, value]（与替身环境一致）
        res = np.zeros((self.n_resources, 4), dtype=np.float32)
        for i, r in enumerate(self.resources):
            res[i] = [r["x"], r["y"], r["amount"], float(r.get("value", 1.0))]
        obs["resources"] = res
        obs["detected"] = np.array([1.0 if d else 0.0 for d in self.detected],
                                   dtype=np.float32)
        # 多 depot：`depots`(n_depots,2) + 兼容旧接口的 `depot`(首个 depot)
        obs["depots"] = np.array(self.depots, dtype=np.float32)
        obs["depot"] = obs["depots"][0] if len(self.depots) > 0 else np.zeros(2, dtype=np.float32)
        obs["roll_pitch"] = {n: np.array(self.roll_pitch.get(n, (0.0, 0.0)), dtype=np.float32)
                             for n in self.robot_names}
        obs["goals"] = {}
        for name in self.robot_names:
            g = goal_for(self, name)
            obs["goals"][name] = np.array(g if g is not None else [0.0, 0.0],
                                          dtype=np.float32)
        return obs

    def _info(self):
        max_roll = max((abs(self.roll_pitch.get(n, (0.0, 0.0))[0]) for n in self.robot_names),
                       default=0.0)
        max_pitch = max((abs(self.roll_pitch.get(n, (0.0, 0.0))[1]) for n in self.robot_names),
                        default=0.0)
        return {
            "delivered": float(self.delivered),
            "delivered_value": float(self.delivered_value),
            "episode_reward": float(self.episode_reward),
            "reward_breakdown": dict(self.reward_breakdown),
            "per_agent_reward": dict(self.per_agent_reward),
            "detected": int(sum(self.detected)),
            "max_roll_deg": float(math.degrees(max_roll)),
            "max_pitch_deg": float(math.degrees(max_pitch)),
            "step": int(self.step_count),
        }

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
