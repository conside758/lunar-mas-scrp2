# 新对话提示词：把当前最优策略部署到 Gazebo

> 复制以下内容作为新对话的开场提示，即可在无需本对话上下文的情况下继续工作。

---

## 任务

把「月球多智能体采矿」的当前最优学习策略**部署到 Gazebo 并验证**。

## 项目背景（简述）

- 月球采矿多智能体：Scout（探测资源）/ Excavator（挖掘）/ Hauler（运输+卸载），三角色功能固定互斥。
- 技术栈：ROS2 Humble + Gazebo classic 11 + colcon + ament_python；训练用纯 Python 替身环境（`lunar_rl`），Gazebo 侧是 `lunar_env` + `lunar_bringup`。
- 规则基线：拍卖式任务分配（bid = value/(1+distance)）。

## 当前状态（务必先读，这是已确定的结论）

- **训练已收敛，当前最优 = 角色感知 BC（行为克隆）**，6 机器人场景 delivered_value = **143**（规则基线 191，总价值 233）。BC 已接受为最终策略。
- **MARL 精修全部放弃**：HAPPO 反事实、AWAC、奖励改进（压 Q 噪声）都已尝试且**无法超过 BC**（详见 `docs/9.16-9.21.md`）。**不要再重新训练或精修 MARL**。
- 6 机器人场景已就绪：`src/lunar_bringup/config/scenario_6r.yaml`（scout×2 / excavator×2 / hauler×2 + 2 depot + 8 异构资源，value 1/2/3）。
- checkpoint：`/home/admina/MAS_ws/.tmp/6r_bc_var.pt`（**角色感知 BC**，6 个 `VarRoleAwareActor`，机器人名 scout_0/scout_1/excavator_0/excavator_1/hauler_0/hauler_1）。

## 下一步任务（本次要做的事）

把 `6r_bc_var.pt` 部署到 Gazebo 并验证交付：

1. **Gazebo 里 spawn 6 机器人**（每角色 2 台，namespace 如上 6 个名字）。
2. **让 `lunar_env` 产出与替身一致的观测**（关键，见下「关键事实」），否则策略输入 OOD。
3. **在 Gazebo 里用 checkpoint 跑策略**（`eval_mode:=policy` 加载新 checkpoint），对照规则基线（`eval_mode:=baseline`）看交付。

## 关键事实（避免踩坑，务必核对）

- **替身观测格式（`src/lunar_rl/lunar_rl/obs.py` 的 `encode_obs_vars`）**：每车返回
  `fixed(13) + res_ent(8×5=[dx,dy,amount,value,detected]) + depot_ent(2×2=[dx,dy]) + team_ent(5×2=[dx,dy]) + 各 mask`。
  **不是旧的 32 维固定观测**。
- **Actor 结构（`networks.py` 的 `VarRoleAwareActor`）**：`fixed(13)→MLP→对资源/depot/队友三组实体做掩码注意力→drive(2)+tool_logits(3)`。
- **机器人名/角色映射**在 `src/lunar_rl/lunar_rl/registry.py`（`ROBOT_NAMES`、`ROBOT_ROLE`）。
- **`lunar_env` 目前还是 3 机器人旧版**（`env.py` 里 obs 键还是 scout/excavator/hauler、resources 是 (n,3)、单 depot）——需要对齐上面的 6 机器人 + 新观测格式。`lunar_bringup.launch.py` 的 `eval_mode:=policy` 目前还加载旧的 `stage2_gz_dagger2.pt`，要改成 `6r_bc_var.pt`。
- **规则策略已是 6 机器人拍卖版**（`src/lunar_rl/lunar_rl/rule_policy.py` 的 `rule_action`，目标点取 `obs["goals"]`），可作为 Gazebo 规则基线对照。
- **Gazebo 无头环境约束**：`export ROS_HOME=/home/admina/MAS_ws/.ros`；软件渲染 `__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1`；跨进程 DDS 有问题 → 评估节点要跑在 launch 进程组内（用 `eval_mode` 参数）。
- **改了 `lunar_bringup`/`lunar_env` 后必须 `colcon build` 重建**，否则用旧安装。

## 环境命令

```bash
cd /home/admina/MAS_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_HOME=/home/admina/MAS_ws/.ros
ros2 launch lunar_bringup lunar_bringup.launch.py gui:=false eval_mode:=policy
```

## 参考文档

- `docs/9.16-9.21.md`（6 机器人场景 + 结果 + 问题澄清）
- `docs/模型训练方案.md` §9.3（6 机器人结果）
- `docs/9.9-9.15.md`（更早的 HAPPO/HASAC/变规模/平滑奖励历史）

## 不要重复做的事

- 不要再训练/精修 MARL（HAPPO 反事实 / AWAC / 奖励改进）——都已试过且无法超过 BC。直接用 `6r_bc_var.pt` 部署。
