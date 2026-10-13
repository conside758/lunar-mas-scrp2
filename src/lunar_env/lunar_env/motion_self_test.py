"""运动自检节点：在 bringup 进程组内验证各台车能否被 cmd_vel 驱动移动（支持 N 机器人）。

用途：绕开沙箱「跨进程 DDS 投递受限」的限制——本节点由 launch 与 gzserver 同进程组启动，
因此能可靠收发 ROS 话题，用于验证车轮方向/速度接口/步态是否真正让机器人移动。

用法：ros2 run lunar_env motion_self_test <scenario_file>
流程（依次对每台机器人）：
  1) 记录 base_link 当前位姿；
  2) 发布 0.5 m/s 前进 cmd_vel 持续 4 s；
  3) 记录结束位姿并打印位移。
"""
import sys
import time

import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

TEST_DURATION = 4.0
LIN_VEL = 0.5


class MotionSelfTest(Node):
    def __init__(self, scenario_file):
        super().__init__("motion_self_test")
        with open(scenario_file, 'r', encoding='utf-8') as f:
            scenario = yaml.safe_load(f)
        self.robots = list(scenario['robots'].keys())
        self.topics = {n: scenario['robots'][n].get(
            'cmd_vel_topic', f"/{scenario['robots'][n]['namespace']}/cmd_vel")
            for n in self.robots}

        self.link_pose = {}
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(LinkStates, "/gazebo/link_states", self._cb, qos)
        self.pubs = {n: self.create_publisher(Twist, t, 10) for n, t in self.topics.items()}

        self.idx = 0
        self.phase = "record_before"
        self.before = None
        self.t0 = 0.0
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(f"motion self test started for {len(self.robots)} robots: "
                               f"{self.robots}")

    def _cb(self, msg):
        for name, pose, _twist in zip(msg.name, msg.pose, msg.twist):
            for rn in self.robots:
                if name == f"{rn}::base_link":
                    self.link_pose[rn] = (pose.position.x, pose.position.y)

    def _tick(self):
        if self.idx >= len(self.robots):
            self.get_logger().info("=== motion self test finished ===")
            self.destroy_node()
            rclpy.shutdown()
            return

        name = self.robots[self.idx]
        if name not in self.link_pose:
            self.get_logger().info(
                f"waiting for link_states... {len(self.link_pose)}/{len(self.robots)}")
            return

        if self.phase == "record_before":
            self.before = self.link_pose.get(name)
            self.t0 = time.time()
            self.phase = "drive"
            self._pub(name, LIN_VEL)
            return

        if self.phase == "drive":
            self._pub(name, LIN_VEL)
            if time.time() - self.t0 >= TEST_DURATION:
                self._pub(name, 0.0)
                after = self.link_pose.get(name)
                if self.before and after:
                    dx = after[0] - self.before[0]
                    dy = after[1] - self.before[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    self.get_logger().info(
                        f"[{name}] before={self.before} after={after} "
                        f"displacement={dist:.3f} m  (dx={dx:.3f}, dy={dy:.3f})")
                else:
                    self.get_logger().warn(f"[{name}] missing pose data")
                self.idx += 1
                self.phase = "record_before"
            return

    def _pub(self, name, lin):
        t = Twist()
        t.linear.x = float(lin)
        self.pubs[name].publish(t)


def main(args=None):
    rclpy.init(args=args)
    scenario_file = sys.argv[1] if len(sys.argv) > 1 else None
    node = MotionSelfTest(scenario_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
