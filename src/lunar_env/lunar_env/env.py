"""LunarEnv：包装 ROS2 + Gazebo 的 Gymnasium 环境。

- reset() 重置机器人位姿与任务状态（资源量、载量、交付量）。
- step() 下发各机器人动作（驱动 + 工具），按 control_horizon 推进仿真，返回 (obs, reward, done, info)。
- 任务语义（Stage1 简化）：挖掘车在资源区"挖掘"累加铲斗载量，在运输车旁"装载"转移载量，
  运输车在卸载点"倾卸"产生交付回报。

使用方式（必须先 rclpy.init，或由本类自动 init）：
    from lunar_env.env import LunarEnv
    env = LunarEnv(scenario_file=...)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

action 结构（gymnasium.spaces.Dict）：
    {'scout':      {'drive': [vx_norm, wz_norm]},
     'excavator':  {'drive': [vx_norm, wz_norm], 'tool': int},
     'hauler':     {'drive': [vx_norm, wz_norm], 'tool': int}}
    vx_norm/wz_norm ∈ [-1,1]；tool: excavator {0 idle,1 dig,2 load}，hauler {0 idle,1 dump}。
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


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


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
        self.max_ang = 0.8
        self.control_horizon = float(ep["control_horizon_s"])
        self.max_steps = int(float(ep["max_time_s"]) / self.control_horizon)
        self.dig_radius = float(ep["dig_radius"])
        self.load_radius = float(ep["load_radius"])
        self.dump_radius = float(ep["dump_radius"])
        self.detect_radius = float(ep.get("detect_radius", 1.5))  # Scout 探测半径（Stage 2）
        self.bucket_capacity = float(ep["bucket_capacity"])
        self.bin_capacity = float(ep["bin_capacity"])
        self.dig_rate = float(ep["dig_rate_per_s"])

        self.robot_names = list(self.scenario["robots"].keys())
        self.n_resources = len(self.scenario["resources"])

        self._init_spaces()

        # ROS
        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self.cmd_pubs = {}
        for name in self.robot_names:
            cfg = self.scenario["robots"][name]
            ns = cfg["namespace"]
            topic = cfg.get("cmd_vel_topic", f"/{ns}/cmd_vel")
            self.cmd_pubs[name] = self.node.create_publisher(Twist, topic, 10)

        self.arm_pub = self.node.create_publisher(
            Float64MultiArray, "/excavator/arm_controller/commands", 10)
        self.dump_pub = self.node.create_publisher(
            Float64MultiArray, "/hauler/dump_controller/commands", 10)

        # 缓存 link_states 中的位姿/速度
        self._link_pose = {}
        self._link_twist = {}
        for name, cfg in self.scenario["robots"].items():
            self._link_pose[name] = (cfg["x"], cfg["y"], cfg["yaw"])
            self._link_twist[name] = (0.0, 0.0)
        self.node.create_subscription(LinkStates, "/gazebo/link_states", self._link_cb, 10)

        self.set_state_client = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.reset_scenario_client = self.node.create_client(Trigger, "/reset_scenario")

        self.reset()

    # ---------- 初始化 ----------
    def _load_scenario(self):
        with open(self.scenario_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_spaces(self):
        self.action_space = spaces.Dict({
            "scout": spaces.Dict({
                "drive": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }),
            "excavator": spaces.Dict({
                "drive": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "tool": spaces.Discrete(3),
            }),
            "hauler": spaces.Dict({
                "drive": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "tool": spaces.Discrete(2),
            }),
        })
        inf = np.finfo(np.float32).max
        self.observation_space = spaces.Dict({
            "scout": spaces.Box(low=-inf, high=inf, shape=(5,), dtype=np.float32),
            "excavator": spaces.Box(low=-inf, high=inf, shape=(6,), dtype=np.float32),
            "hauler": spaces.Box(low=-inf, high=inf, shape=(6,), dtype=np.float32),
            "resources": spaces.Box(low=-inf, high=inf, shape=(self.n_resources, 3), dtype=np.float32),
            "detected": spaces.Box(low=0.0, high=1.0, shape=(self.n_resources,), dtype=np.float32),
            "depot": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
            "goals": spaces.Dict({
                "scout": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
                "excavator": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
                "hauler": spaces.Box(low=-inf, high=inf, shape=(2,), dtype=np.float32),
            }),
        })

    # ---------- ROS 回调 ----------
    def _link_cb(self, msg: LinkStates):
        for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
            for rname in self.robot_names:
                if name == f"{rname}::base_link":
                    self._link_pose[rname] = (
                        pose.position.x, pose.position.y, _yaw_from_quat(pose.orientation))
                    self._link_twist[rname] = (twist.linear.x, twist.angular.z)

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

    def _get_pose(self, name):
        return self._link_pose.get(name, (0.0, 0.0, 0.0))

    def _get_twist(self, name):
        return self._link_twist.get(name, (0.0, 0.0))

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scenario = self._load_scenario()

        # 任务状态
        self.resources = [
            {"id": int(r["id"]), "x": float(r["x"]), "y": float(r["y"]),
             "amount": float(r["amount"]), "value": float(r.get("value", 1.0))}
            for r in self.scenario["resources"]
        ]
        self.depot = (float(self.scenario["depot"]["x"]), float(self.scenario["depot"]["y"]))
        self.detected = [False] * self.n_resources  # 资源是否已被 Scout 发现（Stage 2）
        self.bucket = 0.0
        self.bin = 0.0
        self.delivered = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.tool = {"scout": 0, "excavator": 0, "hauler": 0}

        # 复位机器人位姿
        for name, cfg in self.scenario["robots"].items():
            self._set_entity_state(cfg["entity"], cfg["x"], cfg["y"], cfg["z"], cfg["yaw"])
            self._publish_cmd(name, 0.0, 0.0)

        # 复位工具姿态
        self.arm_pub.publish(Float64MultiArray(data=[0.2, -0.4, 0.2]))
        self.dump_pub.publish(Float64MultiArray(data=[0.0]))

        self._call_reset_scenario()
        self._spin(0.2)
        return self._obs(), self._info()

    def step(self, action):
        self._apply_action(action)
        self._spin(self.control_horizon)
        reward = self._update_task()
        self.step_count += 1
        self.episode_reward += reward
        terminated = self.step_count >= self.max_steps
        truncated = False
        return self._obs(), reward, terminated, truncated, self._info()

    def _apply_action(self, action):
        for name in self.robot_names:
            a = action[name]
            drive = a["drive"]
            vx = float(np.clip(drive[0], -1.0, 1.0)) * self.max_lin
            wz = float(np.clip(drive[1], -1.0, 1.0)) * self.max_ang
            self._publish_cmd(name, vx, wz)
            if isinstance(a, dict) and "tool" in a:
                self.tool[name] = int(a["tool"])

        # 挖掘车机械臂姿态
        t = self.tool["excavator"]
        if t == 1:
            arm = [0.9, -1.2, 0.4]
        elif t == 2:
            arm = [-0.9, -0.5, 0.5]
        else:
            arm = [0.2, -0.4, 0.2]
        self.arm_pub.publish(Float64MultiArray(data=arm))

        # 运输车倾卸
        self.dump_pub.publish(Float64MultiArray(
            data=[-0.8 if self.tool["hauler"] == 1 else 0.0]))

    def _update_task(self):
        dt = self.control_horizon
        reward = 0.0
        p_ex = self._get_pose("excavator")
        p_ha = self._get_pose("hauler")

        # Scout 探测：进入探测半径的资源标记为「已发现」，并给探测奖励（Stage 2）
        p_sc = self._get_pose("scout")
        for i, r in enumerate(self.resources):
            if not self.detected[i] and r["amount"] > 0.01 and \
                    self._dist(p_sc, r) < self.detect_radius:
                self.detected[i] = True
                reward += 1.0

        # 挖掘
        if self.tool["excavator"] == 1:
            for r in self.resources:
                if r["amount"] > 0 and self._dist(p_ex, r) < self.dig_radius:
                    gain = min(self.dig_rate * dt, self.bucket_capacity - self.bucket, r["amount"])
                    self.bucket += gain
                    r["amount"] -= gain

        # 装载（挖掘车 → 运输车）
        if self.tool["excavator"] == 2 and self._dist(p_ex, p_ha) < self.load_radius:
            transfer = min(self.bucket, self.bin_capacity - self.bin)
            self.bucket -= transfer
            self.bin += transfer

        # 倾卸（运输车在卸载点）
        if self.tool["hauler"] == 1 and self._dist(p_ha, self.depot) < self.dump_radius:
            if self.bin > 0:
                reward += self.bin
                self.delivered += self.bin
                self.bin = 0.0

        reward -= 0.01 * dt
        return reward

    @staticmethod
    def _dist(pose, target):
        if isinstance(target, dict):
            tx, ty = target["x"], target["y"]
        else:
            tx, ty = target[0], target[1]
        return math.hypot(pose[0] - tx, pose[1] - ty)

    def _nearest_detected_resource(self, x, y):
        cand = [r for i, r in enumerate(self.resources)
                if r["amount"] > 0.01 and self.detected[i]]
        if not cand:
            return None
        return min(cand, key=lambda r: (r["x"] - x) ** 2 + (r["y"] - y) ** 2)

    def _nearest_undetected_resource(self, x, y):
        cand = [r for i, r in enumerate(self.resources)
                if r["amount"] > 0.01 and not self.detected[i]]
        if not cand:
            return None
        return min(cand, key=lambda r: (r["x"] - x) ** 2 + (r["y"] - y) ** 2)

    def _goal_for(self, name):
        """返回各车当前子目标点 (x, y)。与 lunar_rl.rewards.goal_for 保持一致（Stage 2）。"""
        if name == "excavator":
            ex_x, ex_y, _ = self._get_pose("excavator")
            if self.bucket < self.bucket_capacity - 0.05:
                r = self._nearest_detected_resource(ex_x, ex_y)
                if r is not None:
                    return (r["x"], r["y"])
            return self._get_pose("hauler")[:2]
        if name == "hauler":
            if self.bin > 0.05:
                return self.depot
            return self._get_pose("excavator")[:2]
        sc_x, sc_y, _ = self._get_pose("scout")
        r = self._nearest_undetected_resource(sc_x, sc_y)  # 探索未发现资源
        if r is not None:
            return (r["x"], r["y"])
        return (sc_x, sc_y)  # 都发现后原地待命

    def _obs(self):
        obs = {}
        obs["scout"] = self._robot_obs("scout", with_cargo=False)
        obs["excavator"] = self._robot_obs("excavator", with_cargo=True, cargo=self.bucket)
        obs["hauler"] = self._robot_obs("hauler", with_cargo=True, cargo=self.bin)

        res = np.zeros((self.n_resources, 3), dtype=np.float32)
        for i, r in enumerate(self.resources):
            res[i] = [r["x"], r["y"], r["amount"]]
        obs["resources"] = res
        obs["detected"] = np.array([1.0 if d else 0.0 for d in self.detected],
                                   dtype=np.float32)  # 资源发现掩码
        obs["depot"] = np.array(self.depot, dtype=np.float32)
        obs["goals"] = {}
        for name in self.robot_names:
            g = self._goal_for(name)
            obs["goals"][name] = np.array(g if g is not None else [0.0, 0.0],
                                         dtype=np.float32)
        return obs

    def _robot_obs(self, name, with_cargo, cargo=0.0):
        x, y, yaw = self._get_pose(name)
        vx, wz = self._get_twist(name)
        if with_cargo:
            return np.array([x, y, yaw, vx, wz, cargo], dtype=np.float32)
        return np.array([x, y, yaw, vx, wz], dtype=np.float32)

    def _info(self):
        return {
            "delivered": float(self.delivered),
            "episode_reward": float(self.episode_reward),
            "detected": int(sum(self.detected)),
            "step": int(self.step_count),
        }

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
