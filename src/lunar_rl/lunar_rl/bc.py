"""行为克隆（BC）+ DAgger：用规则基线 rollout 预热各角色 actor，缓解稀疏回报与复合误差。

- collect_rule_data：用 rule_policy 跑若干回合，收集 (obs_batch, drive, tools)。
- collect_policy_data：用当前 actors 确定性 rollout，但把访问到的状态按 rule_policy 重标注（DAgger）。
- dagger：迭代「rollout 当前策略 → 专家重标注 → 增广数据集 → 再训练」，修复 BC 复合误差。
- train_bc：逐 Agent 监督学习（每个 actor 只用自己的角色数据切片，drive 用 MSE，tool 用交叉熵）。
"""
import numpy as np
import torch
import torch.nn as nn

from lunar_rl.networks import ROBOT_NAMES
from lunar_rl.obs import encode_obs
from lunar_rl.rule_policy import rule_action

TOOL_DIM = {"excavator": 3, "hauler": 2, "scout": 0}


def _rule_labels(obs):
    """把 obs 转成 BC 标签 (obs_batch, drive, tools)。"""
    action = rule_action(obs)
    enc = encode_obs(obs)
    obs_batch = np.stack([enc[n] for n in ROBOT_NAMES]).astype(np.float32)
    drive = np.array([action[n]["drive"] for n in ROBOT_NAMES], dtype=np.float32)
    tools = [action[n].get("tool", None) for n in ROBOT_NAMES]
    return obs_batch, drive, tools


def _det_action(actors, obs):
    """actors 字典的确定性动作 dict（逐 Agent 取 drive 均值 + tool argmax）。"""
    enc = encode_obs(obs)
    ob = torch.tensor(np.stack([enc[n] for n in ROBOT_NAMES]),
                      dtype=torch.float32)
    action = {}
    with torch.no_grad():
        for i, name in enumerate(ROBOT_NAMES):
            dm, _, tl = actors[name](ob[i:i + 1])
            sub = {"drive": torch.clamp(dm[0], -1.0, 1.0).numpy().astype(np.float32)}
            k = TOOL_DIM[name]
            if k > 0:
                sub["tool"] = int(torch.argmax(tl[0, :k]).item())
            action[name] = sub
    return action


def collect_rule_data(env, episodes):
    data = []
    for _ in range(episodes):
        obs, _ = env.reset()
        while True:
            data.append(_rule_labels(obs))
            obs, _, term, trunc, _ = env.step(rule_action(obs))
            if term or trunc:
                break
    return data


def collect_policy_data(env, actors, episodes):
    """DAgger 采集：用当前 actors 滚动，但把访问状态按 rule_policy 重标注。"""
    data = []
    for _ in range(episodes):
        obs, _ = env.reset()
        while True:
            data.append(_rule_labels(obs))
            obs, _, term, trunc, _ = env.step(_det_action(actors, obs))
            if term or trunc:
                break
    return data


def dagger(actors, env, rounds=3, episodes=10, epochs=20, lr=1e-3,
           init_episodes=20):
    """DAgger 主循环：初始规则数据 → 迭代增广 → 每轮重训。"""
    data = collect_rule_data(env, init_episodes)
    train_bc(actors, data, epochs=epochs, lr=lr)
    for r in range(rounds):
        d = collect_policy_data(env, actors, episodes)
        data += d
        print(f"[dagger] round {r+1}: +{len(d)} samples (total {len(data)})",
              flush=True)
        train_bc(actors, data, epochs=epochs, lr=lr)
    return actors


def train_bc(actors, data, epochs=20, lr=1e-3, batch=256, device="cpu"):
    """逐 Agent 监督学习：每个 actor 只用自己角色的数据切片训练。"""
    params = [p for a in actors.values() for p in a.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    n = len(data)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            obs_batch = torch.tensor(np.stack([data[j][0] for j in b]),
                                     dtype=torch.float32, device=device)  # (B,3,obs_dim)
            drive_t = torch.tensor(np.stack([data[j][1] for j in b]),
                                   dtype=torch.float32, device=device)  # (B,3,2)
            loss = 0.0
            for k, role in enumerate(ROBOT_NAMES):
                obs_k = obs_batch[:, k, :]          # (B, obs_dim)
                drive_mean, _, tool_logits = actors[role](obs_k)
                loss = loss + mse(drive_mean, drive_t[:, k, :])
                n_tool = TOOL_DIM[role]
                if n_tool > 0:
                    tools_k = torch.tensor([data[j][2][k] for j in b],
                                           dtype=torch.long, device=device)
                    loss = loss + ce(tool_logits[:, :n_tool], tools_k)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"[bc] epoch={ep+1} loss={total/max(1, n//batch):.4f}", flush=True)
    return actors
