# Phase 0 环境搭建与运行说明

本文记录 Phase 0（环境 + 机器人 + 训练接口就绪）的完整搭建步骤、关键修复与已知问题。

## 1. 工具链

- ROS2 Humble + Gazebo classic 11.10.2
- Python 3.10、colcon、xacro、ros2_control（diff_drive_controller / joint_state_broadcaster / position_controllers）
- `gymnasium`（装入工作空间 `pylib/`，因系统 `~/.local` 只读无法 `--user` 安装）

## 2. 环境变量（每次运行前必须设置）

```bash
source /opt/ros/humble/setup.bash
source /home/admina/MAS_ws/install/setup.bash

# 系统 ~/.ros 只读，ROS 日志必须指向工作空间
export ROS_HOME=/home/admina/MAS_ws/.ros
export ROS_LOG_DIR=/home/admina/MAS_ws/.ros/log
mkdir -p $ROS_HOME $ROS_LOG_DIR

# gymnasium 所在目录
export PYTHONPATH=/home/admina/MAS_ws/pylib:$PYTHONPATH
```

> 无头环境需 Mesa 软件渲染，launch 内已通过 `SetEnvironmentVariable` 设置
> `__GLX_VENDOR_LIBRARY_NAME=mesa` 与 `LIBGL_ALWAYS_SOFTWARE=1`（否则 gzserver 因
> NVIDIA GLX 上下文创建失败而卡死、无法加载世界）。

## 3. 构建

```bash
cd /home/admina/MAS_ws
source /opt/ros/humble/setup.bash
colcon build
```

> 关键：工作空间里 `gz2c_fixed/` 是从源码构建的 **gazebo_ros2_control（含官方
> PR #398 修复）**，用于 overlay 系统自带的 0.4.10。系统版本在“多实例命名空间”上存在
> bug（第二台起的插件会读到上一台的 namespace，导致异构多机器人控制器串扰）。
> 修复内容：`gazebo_ros2_control_plugin.cpp` 移除 `__ns:=` 冗余命名空间注入。

## 4. 启动（无头）

```bash
ros2 launch lunar_bringup lunar_bringup.launch.py gui:=false
```

启动顺序为**完全串行**：`spawn(scout) → spawner(scout) → spawn(excavator) →
spawner(excavator) → spawn(hauler) → spawner(hauler)`，避免 gazebo_ros2_control 插件
共享 rcl 全局参数（`--params-file` 路径）被下一台覆盖的竞态。

## 5. 冒烟测试

```bash
# 快速 5 步验证 reset/step
python3 -c "
import rclpy
from lunar_env.env import LunarEnv
rclpy.init()
env = LunarEnv(scenario_file='/home/admina/MAS_ws/install/lunar_bringup/share/lunar_bringup/config/scenario.yaml')
obs, info = env.reset()
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print('PASS')
env.close(); rclpy.shutdown()
"

# 完整随机策略（每 step 0.5s，较慢）
ros2 run lunar_env random_policy /home/admina/MAS_ws/install/lunar_bringup/share/lunar_bringup/config/scenario.yaml
```

## 6. 验证机器人可动

```bash
ros2 topic pub -r 10 /excavator/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
gz model -m excavator -p   # 观察位姿变化
```

## 6. 验证机器人可动

```bash
# 注意：diff_drive_controller 实际订阅的话题是 cmd_vel_unstamped（use_stamped_vel: false）
ros2 topic pub -r 10 /excavator/drive_controller/cmd_vel_unstamped geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
gz model -m excavator -p   # 观察位姿变化
```

## 7. 当前状态

- ✅ 9 个业务包 + gazebo_ros2_control overlay 全部 `colcon build` 通过
- ✅ 一条 launch 无头拉起通用世界 + 3 台自建机器人 + 控制器 + 资源管理 + 可视化
- ✅ 控制器全部激活：Scout(legs)、Excavator(drive+arm)、Hauler(drive+dump)
- ✅ `lunar_env` 可 reset/step，随机策略不崩溃，返回合法 obs/action/reward
- ✅ 已修复：车轮方向与轴向（关节轴 `0 0 -1`，此前 `0 1 0`/`0 -1 0` 因 rpy 旋转映射到竖直轴，
  导致车轮像陀螺一样绕竖轴自转而不滚动——这是「打滑/侧漂」的根因）、cmd_vel 话题名
  （数据驱动 `cmd_vel_topic`）、`/gazebo/link_states` 缺失（加载 `libgazebo_ros_state.so`）
- ✅ 运动自检：轮式车前进 1.2~1.4 m/4s（直线无侧漂）；Scout 前进 0.14~0.16 m/4s
- ✅ **规则基线闭环跑通**：`rule_policy.py`（就近拍卖 + 会合/装卸状态机）+ `run_baseline.py`，
  240 步交付 **16.0 单位**（reward=14.80），作为后续 MARL 的对照下界与示范来源

## 8. 已知问题

1. **沙箱跨进程 DDS 投递受限**：本工作空间的沙箱每次工具调用可能处于不同网络命名空间，
   导致「后台 gzserver 的 ROS 话题 → 前台测试进程」的 DDS 数据投递失败（单进程内 DDS 正常）。
   已在 launch 内加 `motion_self_test` / `run_baseline`（与 gzserver 同进程组）绕开此限制。
   正常单命名空间环境不受影响。
2. **Scout 步态较慢**：对角小跑关节空间近似，能前进（~0.035 m/s）但不快，后续可调参
   或改三足爬行/IK 提速。
3. `gz model -i` 等 CLI 在控制器运行时偶发超时，与仿真负载有关，不影响 ROS 接口。
