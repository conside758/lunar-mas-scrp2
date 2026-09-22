"""Scout 四足爬行/小跑步态节点（带速度闭环 + 指令看门狗）。

订阅 cmd_vel（geometry_msgs/Twist），向 legs_controller 的 commands 话题
发布 8 个关节位置（顺序与 scout_controllers.yaml 的 joints 一致）：
fl_hip, fl_knee, fr_hip, fr_knee, rl_hip, rl_knee, rr_hip, rr_knee。

- 静止（无 cmd_vel）时：输出零位（四腿竖直、足底着地），机器人保持静止。
- 收到 cmd_vel 后：以对角小跑（trot）的关节空间近似平滑过渡进入步态。
- 速度闭环：读取 /gazebo/link_states 中 base_link 的实际速度，按速度误差缩放
  步态增益，避免位置控制 + 高摩擦形成“能量泵”导致越走越快/失控冲刺。
- 指令看门狗：超过 CMD_TIMEOUT 未收到 cmd_vel 则自动归零，避免锁存导致持续行走。
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import LinkStates

JOINT_ORDER = [
    'fl_hip', 'fl_knee', 'fr_hip', 'fr_knee',
    'rl_hip', 'rl_knee', 'rr_hip', 'rr_knee',
]

# 对角配对：fl-rr 与 fr-rl 相位差 pi
PHASE_OFFSET = {
    'fl': 0.0,
    'rr': 0.0,
    'fr': math.pi,
    'rl': math.pi,
}

CMD_EPS = 0.01       # 速度指令低于该值视为静止
CMD_TIMEOUT = 0.5    # 指令看门狗：超时未收到 cmd_vel 则归零


class ScoutGait(Node):
    def __init__(self):
        super().__init__('scout_gait')

        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('hip_amplitude', 0.15)
        self.declare_parameter('knee_lift', 0.12)
        self.declare_parameter('stance_knee', 0.15)
        self.declare_parameter('step_freq_per_ms', 2.5)   # 每 m/s 的步频
        self.declare_parameter('turn_bias', 0.12)          # 转弯时左右髋偏置
        self.declare_parameter('max_lin', 0.4)
        self.declare_parameter('max_ang', 0.6)
        self.declare_parameter('ramp_time', 0.8)           # 进入/退出步态的平滑时间(s)
        self.declare_parameter('kp_speed', 2.0)            # 速度误差比例增益

        self.rate = self.get_parameter('update_rate').value
        self.hip_amp = self.get_parameter('hip_amplitude').value
        self.knee_lift = self.get_parameter('knee_lift').value
        self.stance_knee = self.get_parameter('stance_knee').value
        self.freq_per_ms = self.get_parameter('step_freq_per_ms').value
        self.turn_bias = self.get_parameter('turn_bias').value
        self.max_lin = self.get_parameter('max_lin').value
        self.max_ang = self.get_parameter('max_ang').value
        self.ramp_time = self.get_parameter('ramp_time').value
        self.kp_speed = self.get_parameter('kp_speed').value

        self.cmd_vel = Twist()
        self.t = 0.0
        self.gait_factor = 0.0     # 0 = 静止零位，1 = 完整步态
        self.actual_vx = 0.0       # base_link 实际前进速度（世界系 x）
        self.last_cmd_time = time.monotonic()

        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.link_sub = self.create_subscription(LinkStates, '/gazebo/link_states',
                                                 self.link_cb, 10)
        self.joint_pub = self.create_publisher(
            Float64MultiArray, 'legs_controller/commands', 10)

        self.timer = self.create_timer(1.0 / self.rate, self.update)

    def cmd_cb(self, msg: Twist):
        self.last_cmd_time = time.monotonic()
        vx = max(-self.max_lin, min(self.max_lin, msg.linear.x))
        wz = max(-self.max_ang, min(self.max_ang, msg.angular.z))
        self.cmd_vel.linear.x = vx
        self.cmd_vel.angular.z = wz

    def link_cb(self, msg: LinkStates):
        for name, twist in zip(msg.name, msg.twist):
            if name == 'scout::base_link':
                self.actual_vx = twist.linear.x
                break

    def update(self):
        dt = 1.0 / self.rate

        # 指令看门狗：超时未收到指令则归零，避免锁存
        if time.monotonic() - self.last_cmd_time > CMD_TIMEOUT:
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0

        vx_cmd = self.cmd_vel.linear.x
        wz_cmd = self.cmd_vel.angular.z
        target = abs(vx_cmd)
        moving = target > CMD_EPS or abs(wz_cmd) > CMD_EPS

        # 速度闭环：误差越大步态增益越大，达到/超过目标速度则趋于 0（滑行）
        speed_err = target - abs(self.actual_vx)
        speed_gain = max(0.0, min(1.0, self.kp_speed * speed_err)) if moving else 0.0

        # 进入/退出步态的平滑过渡
        ramp_target = 1.0 if moving else 0.0
        step = dt / max(self.ramp_time, 1e-3)
        if self.gait_factor < ramp_target:
            self.gait_factor = min(ramp_target, self.gait_factor + step)
        else:
            self.gait_factor = max(ramp_target, self.gait_factor - step)

        gain = self.gait_factor * speed_gain

        cmd = Float64MultiArray()
        cmd.data = [0.0] * len(JOINT_ORDER)

        if gain < 1e-3:
            # 静止/已达目标速度：零位（四腿竖直、足底着地）
            self.joint_pub.publish(cmd)
            return

        freq = max(0.4, self.freq_per_ms * target)
        self.t += dt

        for i, name in enumerate(JOINT_ORDER):
            leg = name.split('_')[0]
            phase = 2.0 * math.pi * freq * self.t + PHASE_OFFSET[leg]

            hip = self.hip_amp * math.sin(phase)
            knee = self.stance_knee + self.knee_lift * max(0.0, math.sin(phase))

            # 转弯：左右腿髋产生差动偏置（按目标角速度方向）
            side = 1.0 if leg[1] == 'l' else -1.0
            hip += side * self.turn_bias * (wz_cmd / self.max_ang)

            if name.endswith('_hip'):
                cmd.data[i] = gain * hip
            else:
                cmd.data[i] = gain * knee

        self.joint_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ScoutGait()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
