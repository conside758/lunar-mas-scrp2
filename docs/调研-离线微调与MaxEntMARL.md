# 调研：有助于当前情况的方法（离线→在线微调 + 最大熵 MARL）

> 日期：2026-09-14。针对当前两个卡点做文献调研：
> ① RL 精修（BC 预热后）不稳定、洗掉 BC（固定/变规模场景一致）；
> ② 角色感知 HAPPO「冻结」在 BC 水平无法向上（次优收敛），HASAC 实现不稳定。

---

## 一、直接对症的四组方法

### 1. RLPD —— 把离线（BC）数据混进在线缓冲，防坍塌【最高优先，改动最小】

- 论文：[Efficient Online Reinforcement Learning with Offline Data](https://arxiv.org/abs/2302.02948)（Ball et al., ICML 2023），代码 [github.com/ikostrikov/rlpd](https://github.com/ikostrikov/rlpd)。
- 核心：**在线更新时把离线（专家/BC）数据与在线数据 50/50 混合**回放，配合高 UTD（update-to-data 比）与 LayerNorm。
- 结论：几乎零改动、零额外开销，报告 **2.5×** 提升。**正好治「RL 精修洗掉 BC」**。
- 落地：在 HAPPO/HASAC 的回放缓冲里**始终保留 BC 示范样本**（不退场），在线 rollout 只占一半。对 on-policy MAPPO，可改成「rehearsal 缓冲」（每次 PPO 更新同时回放 BC 样本）。

### 2. Cal-QL / CQL —— 保守 Q，防 Q 高估导致的漂移

- 论文：[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/abs/2303.05479)（Nakamoto et al., NeurIPS 2023），项目页 [nakamotoo.github.io/Cal-QL](https://nakamotoo.github.io/Cal-QL)。
- 核心：学习**保守（低估）且校准**的 Q 值初始化，使离线策略在在线微调时**不塌**且能继续改进（Q 是「策略真值的下界、行为策略的上界」）。CQL 上一行代码即可加。
- 落地：给 HAPPO 的 `QCritic` 加保守项（对 OOD 动作 Q 加惩罚），或直接换成 Cal-QL 式 Q 更新——治「QCritic 噪声大 → 反事实优势漂移策略」。

### 3. AWAC —— 优势加权回归（隐式 BC 约束）

- 论文：[AWAC: Accelerating Online RL with Offline Datasets](https://arxiv.org/abs/2006.09359)（Nair et al., 2020）。
- 核心：策略更新用 `exp(A/λ)·log π`（优势加权），**隐式约束策略不离开离线数据分布**，天然适配「BC 预热 → 在线微调」。
- 落地：把 HAPPO 的策略更新换成优势加权回归（比「KL 锚 + 硬 clip」更稳），或作为额外的正则项。

### 4. TD3+BC（极简基线）

- 论文：[A Minimalist Approach to Offline RL](https://arxiv.org/abs/2106.06860)（Fujimoto & Gu, 2021）。
- 核心：确定性策略梯度 + 一个 BC 正则项 `λ·Q(s,π(s)) − ‖π(s)−a‖²`。
- 落地：给确定性 Actor 加 BC 拉力，一行改动，可作最简对照。

---

## 二、针对「HAPPO 冻结 / 次优收敛」的最大熵 MARL

### 5. HASAC（真实现，非我之前的朴素 SAC）【关键，直接命中原项目】

- 论文：[Maximum Entropy Heterogeneous-Agent Reinforcement Learning](https://arxiv.org/abs/2306.10715)（Liu, Zhong et al., ICLR 2024 Spotlight），主页 [sites.google.com/view/meharl](https://sites.google.com/view/meharl)。
- 核心机制（我此前实现**漏掉的**）：
  1. **最大熵（MaxEnt）MARL 目标**：从概率图模型导出，加熵 → **保留探索**，避免「标准目标把策略逼成确定性次优 NE」。
  2. **Heterogeneous-Agent Soft Policy Iteration（HASPI）**：证明**单调改进 + 收敛到 QRE（量化响应均衡）**。
  3. **建立在 HAML（异构智能体镜像学习）之上**：镜像学习 = 通用 trust-region 模板（PPO clip / KL 是特例），**给逐智能体单调改进保证**。
- **直接解释了我们两个现象**：
  - 论文 §3.2 的矩阵博弈分析：MAPPO/HAPPO 在标准目标下**会快速收敛到次优 NE**（对应我们「D=36 冻结、无法改进」）；
  - 论文明说「off-policy 算法存在训练不稳定、超参敏感」（对应我们「HASAC 不稳定」），而解法就是**镜像学习 trust region + MaxEnt**。
- **我的实现缺陷定位**：`hasac.py` 只是「逐 Agent 软 Q + 熵」的朴素 SAC，**没有镜像学习 trust region**（没有 per-agent 的 clip/KL 单调改进保证），所以不稳定。要重做应补：软策略迭代 + 镜像学习模板（HAML）+ 自动温度调节。

### 6. HAML / HATRPO / HAPPO（信任域理论底座）

- 论文：[Trust Region Policy Optimisation in Multi-Agent RL](https://arxiv.org/abs/2109.11251)（Kuba et al., ICLR 2022，HATRPO/HAPPO）+ [Heterogeneous-Agent Mirror Learning](https://arxiv.org/abs/2404.03503)（HAML）。
- 价值：**逐智能体单调改进 / NE 收敛的理论模板**——HAPPO/HASAC 的稳定性都源自这一族。我们的 HAPPO 已有顺序更新 + clip，但**QCritic 的高方差**破坏了单调改进，是下一步要收口的地方。

---

## 三、与当前情况的映射 + 推荐优先级

| 当前现象 | 文献对应 | 推荐做法 | 改动量 |
| --- | --- | --- | --- |
| RL 精修洗掉 BC（核心瓶颈） | RLPD / Cal-QL / AWAC / TD3+BC | ① 回放保留 BC 数据；② QCritic 加保守/校准；③ 优势加权回归 | 小～中 |
| HAPPO 冻结无法改进（次优收敛） | HASAC（MaxEnt + 镜像学习） | 重做 HASAC：软策略迭代 + HAML trust region + 自动温度 | 中～大 |
| 我实现的 HASAC 不稳定 | 同上（缺 trust region） | 见上，补镜像学习 | 中 |
| 注意力对 RL 漂移更敏感 | HAML trust region / 图注意力 | 对注意力 Actor 用逐 Agent trust region，或降低注意力层 lr | 小～中 |
| 交付量不连续 → Q 噪声大 | Cal-QL 校准 / 优势塑形 | 平滑势函数塑形 + 保守 Q | 小 |

**最短路径建议（按性价比）：**
1. **先做 RLPD 式「BC 数据保留」**：把 DAgger-BC 样本常驻进回放/缓冲，与在线数据 1:1 混合——大概率直接压住「洗掉 BC」。
2. **再做 Cal-QL 式保守 Q**：给 HAPPO 反事实的 QCritic 加 OOD 惩罚，降低反事实优势方差。
3. **重做 HASAC**：按论文补镜像学习 trust region + MaxEnt 自动温度，替换掉现在的朴素 SAC。
4. 若仍「冻结」，用 **HASAC 的熵项**（而非 PPO 的确定性）去打破次优 NE。

---

## 四、关键出处速查

- HASAC：[arXiv:2306.10715](https://arxiv.org/abs/2306.10715)（ICLR 2024 Spotlight）
- RLPD：[arXiv:2302.02948](https://arxiv.org/abs/2302.02948)（ICML 2023，代码 rlpd）
- Cal-QL：[arXiv:2303.05479](https://arxiv.org/abs/2303.05479)（NeurIPS 2023）
- AWAC：[arXiv:2006.09359](https://arxiv.org/abs/2006.09359)
- TD3+BC：[arXiv:2106.06860](https://arxiv.org/abs/2106.06860)
- HAPPO/HATRPO：[arXiv:2109.11251](https://arxiv.org/abs/2109.11251)
- HAML（异构镜像学习）：[arXiv:2404.03503](https://arxiv.org/abs/2404.03503)
- Heterogeneous-Agent RL 综述：[JMLR 2024](https://mlanthology.org/jmlr/2024/zhong2024jmlr-heterogeneousagent/)

---

## 五、落地结果（2026-09-14，按优先级已实施 ①②）

### ① RLPD 式 BC 数据保留 → 对 HASAC 无明显帮助

- 实现：`hasac.py` 加 `prefill_rule`（规则专家 transitions 预填回放缓冲），`train_hasac.py` 加 `--rlpd-steps`。
- 结果（固定场景 F=角色感知 HASAC，seed 3）：预填 2 万规则样本后交付 **28.0**，基线（无 RLPD）**32.0** → 无改善。
- 判断：HASAC 的不稳定主因不是「缺离线数据」，而是「缺信任域/镜像学习」（见 ③）。

### ② Cal-QL 式保守 Q → 对 HAPPO 反事实「部分降退化、但种子相关」

- 实现：`mappo.py`/`mappo_vars.py` 的 QCritic 损失加 OOD 惩罚 `+ cql_coef · Q(s, a_mean)`，`--cql-coef` 开关。
- 结果（变规模 G=角色感知 HAPPO 反事实，seed 0 基线 19.9）：

  | cql_coef | seed0 | seed2 | 说明 |
  | --- | --- | --- | --- |
  | 0（基线） | 19.9 | 26.8 | — |
  | 0.5 | 33.0 | 22.2 | seed0 改善、seed2 略降 |
  | 1.0 | **35.8** | 24.0 | seed0 最佳、seed2 略降 |
  | 2.0 | 23.5 | — | 过强、回落 |

- 结论：**保守 Q 能显著缓解「seed0 式」的退化（19.9→35.8），但种子相关、对已崩的 seed1（BC 本身仅 19.4）无效**。
  不是稳健解法，仅作「降方差」的辅助手段。**根本瓶颈仍在变规模场景的 Q 噪声 + 注意力对漂移敏感**。

### ③ 镜像学习 trust region（soft PPO clip）→ 固定场景无改善

- 实现：`hasac.py` 加 `clip_eps`（>0 时用 soft PPO clip ratio 作用于 buffer 实际动作，替代重参数化 SAC），
  缓冲新增 `logprob`（在线采样 + 规则预填均存 old_logprob），`train_hasac.py` 加 `--clip-eps`。
- 结果（固定场景 F=角色感知 HASAC，seed 3，clip_eps=0.2）：最终 **20.0**，基线（无信任域）**32.0**，且
  24~40 振荡 → 无改善。
- **判断**：①②③ 三种增量稳定化手段（RLPD / 保守 Q / 信任域）都**未能稳健改善「RL 洗掉 BC」**。
  根本瓶颈是「交付量指标不连续 + 变规模 Q 噪声 + 注意力对漂移敏感」，非单一机制可解。

### 总结论（按优先级已全部实施 + 追加方向 B）

- **RL 精修的稳定性瓶颈是结构性的**：RLPD（离线数据）、Cal-QL（保守 Q）、镜像学习信任域（clip）都无法
  稳健压住退化，仅 Cal-QL 保守 Q 有「种子相关」的部分降退化（seed0 19.9→35.8）。
- **方向 B（平滑奖励）已验证有效 ✅**：把离散「一次性倾倒」的 deliver 奖励改为**连续卸载**
  （`dump_rate·dt`，`surrogate.py`），Q 噪声显著下降——value_loss 从 0.7~1.7 降到 **0.5~0.9**，且
  HAPPO 反事实精修不再退化（固定场景稳定 hold 住 BC：seed0 BC=44 → 精修后 44 全程稳定；seed1 BC=36 →
  精修后 ~36）。BC 本身仍有种子方差（44/36/32），但「低 Q 噪声 + 不退化」是稳健的。见 §六。
