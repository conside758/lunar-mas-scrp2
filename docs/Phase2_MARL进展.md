# Phase 2 阶段进展 —— MARL 训练基础设施与基线实验

> 本文档记录 Phase 2（角色感知异构 MARL）第一阶段的进展：快速替身环境、观测/奖励改造、
> 自实现 MAPPO 基线 + BC 预热、当前实验结果与问题定位、以及下一步计划。
> 相关文档：`docs/训练方案设想.md`、`docs/训练接口与模型输入输出.md`、`docs/下一步计划.md`。

---

## 1. 总体结论

Phase 2 的**训练基础设施已全部搭好并跑通**：从「Gazebo 墙钟驱动」切换到「纯 Python 快速替身环境」，
实现了一套完整、可离线训练的 MAPPO 基线（CTDE + 集中 Critic + GAE + PPO + BC 预热）。
当前处于**调参/课程设计阶段**：规则基线在替身环境交付 48.0，BC 预热交付 20.0，
但 RL 从头训练与 BC+RL 微调均退化到 0，尚未达到「MAPPO ≥ 规则基线」的验收线。

| 里程碑 | 目标 | 状态 |
| --- | --- | --- |
| 快速替身环境 SurrogateEnv | 与 lunar_env 同接口、纯 Python、可离线训练 | ✅ 完成 |
| 观测/奖励改造 | 角色 one-hot + 相对编码 + 最近资源 + 密集奖励 | ✅ 完成 |
| 自实现 MAPPO 基线 | CTDE + GAE + PPO（连续 drive + 离散 tool 头） | ✅ 完成 |
| BC 规则策略预热 | 规则基线 rollout → 监督模仿 | ✅ 完成（交付 20.0） |
| **MAPPO ≥ 规则基线（48 替身 / 16 Gazebo）** | 交付量达标 | ❌ 未达标（当前 0） |
| 角色感知 HAPPO/HASAC | 异构 actor + 角色嵌入 + 注意力 | ⏳ 未开始 |
| Gazebo 验证 + 消融报告 | 四组对比 + 统计显著 | ⏳ 未开始 |

---

## 2. 已完成的工作（交付物）

新增纯 Python 包 `src/lunar_rl/`（ament_python，`colcon build` 通过），
所有训练在替身环境上进行，策略接口与 `lunar_env` 一致，后续可无缝迁移到 Gazebo。

### 2.1 快速替身环境 `lunar_rl/surrogate.py`

- `SurrogateEnv`：与 `lunar_env.LunarEnv` 同接口（`reset/step/_obs/action/reward`），
  纯 Python 运动学模型（`x += vx·cos(yaw)·dt` 等，无碰撞/打滑）。
- 任务逻辑与 `lunar_env._update_task` 逐条一致：dig/load/dump 按半径 + 速率判定。
- 关键收益：单回合毫秒级完成（规则基线 0.04 s/回合），可进行数十万步离线训练。

### 2.2 观测编码 `lunar_rl/obs.py`

每车定长向量（n_resources=3 时 **27 维**）：

```
自身[x,y,yaw,vx,wz,cargo] (6) + 角色 one-hot (3) + 相对 depot (2)
+ 相对最近有量资源 [dx,dy,amount] (3) + 相对所有资源展平 (9) + 相对所有队友展平 (4)
```

- 相对编码（depot/资源/队友）保证平移不变性；显式保留「最近资源」特征，避免池化后丢失
  「去哪挖」的关键信息。
- 集中 Critic 输入 = 三车编码拼接（27×3=81 维）。
- 后续支持变规模时改用注意力池化 / top-K 排序。

### 2.3 奖励塑形 `lunar_rl/rewards.py`

- 交付 `deliver` + 挖掘 `dig` + 装载 `load`（过程密集奖励）+ 吸引 `app`（势函数近似
  `prev_dist - dist`，各车逼近当前子目标）− 步进 `step`。
- 默认权重 `{"deliver":1.0,"dig":0.5,"load":0.5,"app":0.2,"step":0.02}`，可分阶段重加权。
- `goal_for()` 给出各车当前子目标（Excavator 挖满→去装载、Hauler 有货→去 depot 等）。

### 2.4 网络 `lunar_rl/networks.py`

- 准共享 `Actor`（输入含角色 one-hot）：共享主干 + 三个头
  - `drive_mean`（连续驱动 2 维）、`drive_logstd`（可学习参数，初始 −0.7 → std≈0.5）、
    `tool_logits`（离散工具，excavator 3 类 / hauler 2 类 / scout 忽略）。
- 集中 `Critic`（全局 81 维 → 值）。
- `sample_actions`/`eval_actions`/`actions_to_dict`：分角色采样、重算 logprob、转成 env dict action。

### 2.5 MAPPO `lunar_rl/mappo.py`

- 准共享 MAPPO：三车共用一个 Actor（角色 one-hot 区分）+ 集中 Critic，共享回报。
- rollout（收集 obs/动作/logprob/回报/值/done）→ GAE 优势（`gamma=0.99, lambda=0.95`）
  → 多 epoch PPO 更新（policy clip 0.2 + value loss + entropy coef 0.001）。
- 三车**联合** ratio 做策略梯度（joint policy），critic 输出三车共享回报的值。
- `act()` 确定性评估（drive 取均值 + tool 取 argmax）。

### 2.6 BC 预热 `lunar_rl/bc.py` + 平滑规则策略 `lunar_rl/rule_policy.py`

- `rule_policy.py`：规则基线（就近拍卖 + 会合/装卸状态机），关键改动是**平滑 go_to**
  （`vx = clip(1.0·dist,0,1)·max(0,cos(err)); wz = clip(1.2·err,-1,1)`，无硬阈值），
  便于 BC 学习连续控制。
- `bc.py`：规则策略 rollout → `(obs, drive, tool)` 数据集 → 监督模仿
  （drive 用 MSE，tool 用交叉熵）。

### 2.7 训练入口 `lunar_rl/train.py`

```
python3 -m lunar_rl.train <scenario> --seed 1 --iters 120 --steps 500 \
    --eval-every 15 --bc-episodes 20 --bc-epochs 30 --lr 1e-4
```

- 可选 BC 预热 → MAPPO 训练循环 → 定期确定性评估交付量 → 保存 `.tmp/mappo_baseline.pt`。

---

## 3. 当前实验结果

| 配置 | 交付量（确定性评估） | 说明 |
| --- | --- | --- |
| 规则基线（替身环境） | **48.0** | 0.04 s/回合，作为对照上界（替身） |
| BC 预热（seed 1，20 回合，30 epoch） | **20.0** | 规则模仿有损耗，但已形成部分闭环 |
| RL 从头训练（150 iter，75k 步） | 0.0 | 稀疏回报下难以自发形成协作 |
| BC + RL 微调（lr 1e-4，entropy 0.001） | 20 → 0 | ~15 iter 内退化回 0 |

> 注：Gazebo 规则基线为 **16.0 / 240 步**（见 `docs/工作总结.md`）；替身环境无碰撞/打滑、
> Scout 步态忽略，故规则基线更高（48.0）。验收以 Gazebo 16.0 为下界，替身 48.0 为对照。

---

## 4. 问题分析（为什么 RL 还没学好）

1. **协作任务的多阶段稀疏本质**：交付依赖「挖掘→会合→装载→运输→卸载」严格顺序，
   任一环断裂即 0 交付；从随机初始化的探索几乎不可能自发串起整条链。
2. **BC 微调被 PPO 破坏**：虽然已降低 `drive_logstd` 初始值（−0.7）、`entropy_coef`（0.001）、
   `lr`（1e-4），PPO 的探索噪声仍会快速洗掉 BC 学到的确定性子任务，且早期劣势估计噪声大，
   好的 BC 策略反而被「负优势」推离。
3. **奖励尺度/结构仍可优化**：`deliver` 奖励与过程奖励量级差较大；吸引奖励的势函数在
   多车互相影响时可能给出矛盾的梯度。
4. **单阶段全任务端到端训练难度过高**：未做课程（C0→C3），把「卸载/挖掘/会合/调度」一次性丢给模型。

---

## 5. 下一步计划

按「先跑通、再增强」的顺序：

1. **课程训练（当前优先级最高）**——把一次性全任务拆成递增子任务：
   - **C0**：单 Hauler「车斗预装货 → 去 depot 卸载」（无挖掘/会合），先验证 PPO 能学会单一 go-to-dump 闭环；
   - **C1**：+ Excavator 挖掘（单资源，无会合复杂交互）；
   - **C2**：+ 挖掘→装载→会合 完整链；
   - **C3**：+ Scout 巡视 / 多资源调度 / 动态重规划。
   - 每级都用上一级策略 **BC 预热 + 低 lr 微调**，避免从零探索。
2. **BC 微调防退化**：冻结早期 epochs 或用「BC 蒸馏损失」正则项、更低的 RL lr、
   关闭 tool 头噪声（离散头用更低温度/greedy），保住已学闭环。
3. **奖励重标定**：统一 `deliver/dig/load` 与过程项的量级，必要时对 `app` 势函数做归一化。
4. **角色感知 HAPPO/HASAC**：达标后引入异构 actor + 角色嵌入 + 队友注意力，
   对照「无角色 MAPPO」验证角色感知增益。
5. **Gazebo 迁移验证 + 消融报告**：四组对比（规则 / 无角色 MAPPO / 角色感知 MAPPO / HAPPO+HASAC），
   多种子 + 变规模 + 动态重规划，做统计显著消融。

---

## 6. 代码卫生备注（非阻塞）

- `networks.py` 中 `Actor`/`Critic` 构造函数默认值（21/63）与 `obs.py.obs_dim()`=27、
  `train.py` 传入的 27/81 不一致：实际由 `train.py` 显式传参覆盖，不影响运行，
  但建议后续统一默认值或移除魔术数字，避免误用。
