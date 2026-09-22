"""转向自检：命令纯角速度（vx=0），测量 yaw 变化，量化转向效率与方向。"""
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates

ROBOT_TOPICS = {
    "excavator": "/excavator/drive_controller/cmd_vel_unstamped",
    "hauler": "/hauler/drive_controller/cmd_vel_unstamped",
}


def _yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class TurnTest(Node):
    def __init__(self):
        super().__init__("turn_test")
        self.yaw = {}
        self.create_subscription(LinkStates, "/gazebo/link_states", self._cb, 10)
        self.pubs = {n: self.create_publisher(Twist, t, 10) for n, t in ROBOT_TOPICS.items()}
        self.robots = list(ROBOT_TOPICS.keys())
        self.idx = 0
        self.phase = "before"
        self.before = None
        self.t0 = 0.0
        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info("turn test started")

    def _cb(self, msg):
        for name, pose, _ in zip(msg.name, msg.pose, msg.twist):
            for rn in self.robots:
                if name == f"{rn}::base_link":
                    self.yaw[rn] = _yaw(pose.orientation)

    def _tick(self):
        if self.idx >= len(self.robots):
            self.get_logger().info("=== turn test finished ===")
            self.destroy_node()
            rclpy.shutdown()
            return
        name = self.robots[self.idx]
        if name not in self.yaw:
            return
        if self.phase == "before":
            self.before = self.yaw[name]
            self.t0 = time.time()
            self.phase = "turn"
            self._pub(name, 0.0, 0.8)
            return
        if self.phase == "turn":
            self._pub(name, 0.0, 0.8)
            if time.time() - self.t0 >= 4.0:
                self._pub(name, 0.0, 0.0)
                d = self.yaw[name] - self.before
                eff = abs(d) / (0.8 * 4.0) * 100.0
                self.get_logger().info(
                    f"[{name}] yaw change={d:.3f} rad / 4s (cmd wz=0.8) "
                    f"-> efficiency={eff:.1f}%")
                self.idx += 1
                self.phase = "before"
            return

    def _pub(self, name, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.pubs[name].publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = TurnTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
