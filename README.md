# 月球多智能体协同采矿系统（SCRP2 简化版）

基于 NASA Space Robotics Challenge Phase 2（SCRP2）简化规则的异构多机器人协同采矿项目。
由腿式侦察车（Scout）、挖掘车（Excavator）、运输车（Hauler）组成异构团队，自主完成
「资源确认 → 挖掘车调度 → 运载车调度 → 挖掘装载 → 运输卸载」闭环。

- **技术栈**：ROS2 Humble + Gazebo classic 11（`gazebo_ros` / `gazebo_ros2_control`）+ Python 3.10 + `colcon`
- **方法创新点**：角色感知的异构多智能体强化学习（HAPPO / HASAC）
- **初期范围（Phase 0）**：通用环境（非月面）+ 通行代价已知；自建三台简化机器人；
  交付可 `reset/step` 的 Gymnasium 训练接口。

## 包结构

| 包 | 职责 |
| --- | --- |
| `lunar_msgs` | 自定义 msg / action 接口 |
| `lunar_descriptions` | 三台机器人 URDF（xacro）+ 传感器挂载点 |
| `lunar_gazebo` | 通用世界、资源区 / 卸载点模型、spawn 脚本 |
| `lunar_control` | `ros2_control` 配置与驱动 |
| `lunar_perception` | Scout 资源探测等感知节点（Stage1 简化） |
| `lunar_task` | 任务分配 / 调度 / 会合状态机（规则基线） |
| `lunar_env` | Gymnasium 环境封装 ROS2 接口 + 基线策略 |
| `lunar_bringup` | 顶层 launch / 参数 |
| `lunar_utils` | 指标记录等通用工具 |

## 构建

```bash
cd /home/admina/MAS_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 启动（无头）

```bash
ros2 launch lunar_bringup lunar_bringup.launch.py
```

> 完整环境要求、关键修复与运行细节见 [`docs/SETUP.md`](docs/SETUP.md)。
> 无头环境需 Mesa 软件渲染（launch 已内置）；`~/.ros` 只读，需设置 `ROS_HOME`；
> `gymnasium` 装于工作空间 `pylib/`；工作空间内含源码构建的 gazebo_ros2_control overlay
> （修复官方 PR #398 的多实例命名空间 bug）。

详见 `docs/` 与各包内说明。
