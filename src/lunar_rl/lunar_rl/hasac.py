"""HASAC：异构角色 Soft Actor-Critic（离线/回放缓冲）。

每个 agent 一个 Actor（复用 `Actor` / `RoleAwareActor`）+ 一个**逐 Agent 软 Q Critic** Q_i(s, a_i)：
  - Q_i 输入 = 全局状态 s（96 维）+ 该 agent 自身动作 a_i（drive 2 + tool one-hot k）。
  - 熵正则软 Bellman 备份：
        y_i = r_i + γ(1-done)·( Q_targ_i(s', a'_i) − α·log π_i(a'_i|o'_i) )
  - 策略目标（最小化）：J_i = E[ α·log π_i(a|o) − Q_i(s,a) ]；
        drive 用重参数化（reparameterization），tool 用离散精确期望（Σ_t p_t·Q）。
  - 逐角色奖励 shaping：r_i 来自 env.info["per_agent_reward"]（见 surrogate.py）。

与 PPO/HAPPO 的关键区别：① 3 个逐 Agent Critic（非 1 个集中 V/Q）；② 熵正则价值目标（软 Q）；
③ 逐角色奖励 shaping。
"""
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from lunar_rl.networks import (AGENT_ACTION_DIM, ROBOT_NAMES, ROLE_TOOL_DIM,
                               QCritic, actions_to_dict, encode_agent_action)
from lunar_rl.obs import encode_global, encode_obs


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, t):
        self.buf.append(t)

    def sample(self, batch):
        idx = np.random.randint(0, len(self.buf), size=batch)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


class HASAC:
    def __init__(self, actors, qcritics, lr=3e-4, critic_lr=None, gamma=0.99, tau=0.005,
                 alpha=0.2, device="cpu", ref_actors=None, kl_coef=0.0,
                 buffer_capacity=100000, global_dim=96, clip_eps=0.0):
        self.actors = {k: v.to(device) for k, v in actors.items()}
        self.qcritics = {k: v.to(device) for k, v in qcritics.items()}
        self.q_targets = {k: copy.deepcopy(v).to(device) for k, v in self.qcritics.items()}
        for t in self.q_targets.values():
            for p in t.parameters():
                p.requires_grad_(False)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.clip_eps = clip_eps  # >0 时用镜像学习信任域（soft PPO clip ratio），否则用重参数化 SAC
        critic_lr = lr if critic_lr is None else critic_lr
        self.actor_opts = {k: torch.optim.Adam(v.parameters(), lr=lr)
                           for k, v in self.actors.items()}
        self.critic_opts = {k: torch.optim.Adam(v.parameters(), lr=critic_lr)
                            for k, v in self.qcritics.items()}
        # KL/L2 防漂移参考策略（BC 预热后的 actors 深拷贝），>0 时在策略损失里加 KL 正则
        self.ref_actors = ref_actors
        if self.ref_actors is not None:
            for a in self.ref_actors.values():
                for p in a.parameters():
                    p.requires_grad_(False)
        self.kl_coef = kl_coef
        self.buffer = ReplayBuffer(buffer_capacity)

    # ---------- 采样 / 评估 ----------
    def sample(self, obs):
        """从当前策略采样（drive 重参数化 + tool 采样）。返回 (action_dict, drive, tools, logprob)。"""
        enc = encode_obs(obs)
        obs_batch = torch.tensor(np.stack([enc[n] for n in ROBOT_NAMES]),
                                 dtype=torch.float32, device=self.device)
        drives, tools, logprobs = [], [], []
        with torch.no_grad():
            for i, role in enumerate(ROBOT_NAMES):
                dm, ds, tl = self.actors[role](obs_batch[i:i + 1])
                dist = Normal(dm[0], ds)
                u = dist.sample()
                drive = torch.clamp(u, -1.0, 1.0)
                logprob = dist.log_prob(u).sum()
                tool = 0
                k = ROLE_TOOL_DIM[role]
                if k > 0:
                    d = Categorical(logits=tl[0, :k])
                    tool = d.sample().item()
                    logprob = logprob + d.log_prob(torch.tensor(tool, device=self.device))
                drives.append(drive)
                tools.append(tool)
                logprobs.append(logprob)
        drive_t = torch.stack(drives)
        return actions_to_dict(drive_t, tools), drive_t, tools, torch.stack(logprobs)

    def act(self, obs):
        """确定性动作（评估用，逐 Agent 取 drive 均值 + tool argmax）。"""
        enc = encode_obs(obs)
        obs_batch = torch.tensor(np.stack([enc[n] for n in ROBOT_NAMES]),
                                 dtype=torch.float32, device=self.device)
        drives, tools = [], []
        with torch.no_grad():
            for i, role in enumerate(ROBOT_NAMES):
                dm, _, tl = self.actors[role](obs_batch[i:i + 1])
                drives.append(torch.clamp(dm[0], -1.0, 1.0))
                k = ROLE_TOOL_DIM[role]
                tools.append(int(torch.argmax(tl[0, :k]).item()) if k > 0 else 0)
        return actions_to_dict(torch.stack(drives), tools)

    def rollout(self, env, steps):
        """与环境交互 steps 步，把 (s, a, r_i, s', done) 存入回放缓冲。"""
        obs, _ = env.reset()
        n = 0
        for _ in range(steps):
            enc = encode_obs(obs)
            g = encode_global(enc)
            action, drive, tools, logprob = self.sample(obs)
            obs2, reward, term, trunc, info = env.step(action)
            enc2 = encode_obs(obs2)
            ng = encode_global(enc2)
            r = np.array([info["per_agent_reward"][n] for n in ROBOT_NAMES],
                         dtype=np.float32)
            self.buffer.push({
                "g": g.astype(np.float32),
                "obs": np.stack([enc[n] for n in ROBOT_NAMES]).astype(np.float32),
                "drive": drive.detach().cpu().numpy().astype(np.float32),
                "tools": tools,
                "logprob": logprob.detach().cpu().numpy().astype(np.float32),
                "r": r,
                "ng": ng.astype(np.float32),
                "nobs": np.stack([enc2[n] for n in ROBOT_NAMES]).astype(np.float32),
                "done": bool(term or trunc),
            })
            n += 1
            obs = obs2
            if term or trunc:
                obs, _ = env.reset()
        return n

    def prefill_rule(self, env, steps):
        """RLPD 式预填：把规则专家（或 BC）示范 transitions 存进回放缓冲，供在线更新 50/50 混合。

        用规则策略收集 (s, a, r_i, s', done)，使离线示范数据常驻缓冲，防止在线精修把策略洗掉。
        """
        from lunar_rl.rule_policy import rule_action
        obs, _ = env.reset()
        n = 0
        for _ in range(steps):
            enc = encode_obs(obs)
            g = encode_global(enc)
            action = rule_action(obs)
            drive = np.array([action[n]["drive"] for n in ROBOT_NAMES],
                             dtype=np.float32)
            tools = [action[n].get("tool", 0) for n in ROBOT_NAMES]
            # 当前策略对规则动作的 logprob（信任域 old_logprob 用；预填时策略≈BC）
            with torch.no_grad():
                ob = torch.tensor(np.stack([enc[n] for n in ROBOT_NAMES]),
                                  dtype=torch.float32, device=self.device)
                lp = []
                for j, role in enumerate(ROBOT_NAMES):
                    dm, ds, tl = self.actors[role](ob[j:j + 1])
                    dist = Normal(dm[0], ds)
                    lp_j = dist.log_prob(torch.tensor(drive[j], dtype=torch.float32,
                                                      device=self.device)).sum()
                    k = ROLE_TOOL_DIM[role]
                    if k > 0:
                        d = Categorical(logits=tl[0, :k])
                        lp_j = lp_j + d.log_prob(torch.tensor(tools[j], dtype=torch.long,
                                                              device=self.device))
                    lp.append(lp_j.item())
                logprob = np.array(lp, dtype=np.float32)
            obs2, reward, term, trunc, info = env.step(action)
            enc2 = encode_obs(obs2)
            ng = encode_global(enc2)
            r = np.array([info["per_agent_reward"][n] for n in ROBOT_NAMES],
                         dtype=np.float32)
            self.buffer.push({
                "g": g.astype(np.float32),
                "obs": np.stack([enc[n] for n in ROBOT_NAMES]).astype(np.float32),
                "drive": drive,
                "tools": tools,
                "logprob": logprob,
                "r": r,
                "ng": ng.astype(np.float32),
                "nobs": np.stack([enc2[n] for n in ROBOT_NAMES]).astype(np.float32),
                "done": bool(term or trunc),
            })
            n += 1
            obs = obs2
            if term or trunc:
                obs, _ = env.reset()
        return n

    # ---------- SAC 更新 ----------
    def update(self, batch_size=256, updates=1):
        stats = {"q_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0}
        if len(self.buffer) < batch_size:
            return stats
        for _ in range(updates):
            batch = self.buffer.sample(batch_size)
            B = len(batch)
            g = torch.tensor(np.stack([t["g"] for t in batch]),
                             dtype=torch.float32, device=self.device)
            ng = torch.tensor(np.stack([t["ng"] for t in batch]),
                              dtype=torch.float32, device=self.device)
            obs = torch.tensor(np.stack([t["obs"] for t in batch]),
                               dtype=torch.float32, device=self.device)   # (B,3,32)
            nobs = torch.tensor(np.stack([t["nobs"] for t in batch]),
                                dtype=torch.float32, device=self.device)
            drive = torch.tensor(np.stack([t["drive"] for t in batch]),
                                 dtype=torch.float32, device=self.device)  # (B,3,2)
            r = torch.tensor(np.stack([t["r"] for t in batch]),
                             dtype=torch.float32, device=self.device)      # (B,3)
            old_logprob = torch.tensor(np.stack([t["logprob"] for t in batch]),
                                       dtype=torch.float32, device=self.device)  # (B,3)
            done = torch.tensor([1.0 if t["done"] else 0.0 for t in batch],
                                dtype=torch.float32, device=self.device)

            q_loss_sum = 0.0
            pi_loss_sum = 0.0
            ent_sum = 0.0
            for j, role in enumerate(ROBOT_NAMES):
                k = ROLE_TOOL_DIM[role]
                # ---- 实际动作 a_j 编码 ----
                tools_j = torch.tensor([int(t["tools"][j]) for t in batch],
                                       dtype=torch.long, device=self.device)
                if k > 0:
                    oh_a = torch.zeros(B, k, device=self.device)
                    oh_a[torch.arange(B), tools_j] = 1.0
                    a_j = torch.cat([drive[:, j, :], oh_a], dim=-1)
                else:
                    a_j = drive[:, j, :]

                # ---- 软 Bellman 目标 y_i ----
                with torch.no_grad():
                    dm_n, ds_n, tl_n = self.actors[role](nobs[:, j, :])
                    dist_n = Normal(dm_n, ds_n)
                    u_n = dist_n.sample()
                    drive_n = torch.clamp(u_n, -1.0, 1.0)
                    logp_n = dist_n.log_prob(u_n).sum(-1)
                    if k > 0:
                        cat_n = Categorical(logits=tl_n[:, :k])
                        tool_n = cat_n.sample()
                        logp_n = logp_n + cat_n.log_prob(tool_n)
                        oh_n = torch.zeros(B, k, device=self.device)
                        oh_n[torch.arange(B), tool_n] = 1.0
                        a_n = torch.cat([drive_n, oh_n], dim=-1)
                    else:
                        a_n = drive_n
                    q_next = self.q_targets[role](ng, a_n)
                    y = r[:, j] + self.gamma * (1.0 - done) * (q_next - self.alpha * logp_n)

                # ---- Critic 更新（先更新 critic，再算策略损失，避免 inplace 冲突）----
                q_cur = self.qcritics[role](g, a_j)
                q_loss = ((q_cur - y.detach()) ** 2).mean()
                self.critic_opts[role].zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(self.qcritics[role].parameters(), 0.5)
                self.critic_opts[role].step()

                # ---- 策略更新（critic 更新后的 fresh forward）----
                dm, ds, tl = self.actors[role](obs[:, j, :])
                dist = Normal(dm, ds)
                if self.clip_eps > 0.0:
                    # 镜像学习信任域：soft PPO clip ratio 作用于 buffer 里的实际动作 a_j
                    new_lp = dist.log_prob(drive[:, j]).sum(-1)
                    if k > 0:
                        d = Categorical(logits=tl[:, :k])
                        new_lp = new_lp + d.log_prob(tools_j)
                    ratio = torch.exp(new_lp - old_logprob[:, j])
                    q_val = self.qcritics[role](g, a_j)          # Q(s, a_actual)
                    soft_adv = q_val - self.alpha * new_lp       # 软优势（最大化）
                    surr1 = ratio * soft_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                        1 + self.clip_eps) * soft_adv
                    pi_loss = -torch.min(surr1, surr2).mean()
                    ent_j = dist.entropy().sum(-1)
                    if k > 0:
                        ent_j = ent_j + Categorical(logits=tl[:, :k]).entropy()
                else:
                    # 重参数化 SAC
                    u = dist.rsample()
                    drive_s = torch.clamp(u, -1.0, 1.0)
                    logp_drive = dist.log_prob(u).sum(-1)
                    if k == 0:
                        q_exp = self.qcritics[role](g, drive_s)
                        h_tool = torch.zeros(B, device=self.device)
                    else:
                        p = torch.softmax(tl[:, :k], dim=-1)          # (B,k)
                        qs = []
                        for t_idx in range(k):
                            oh = torch.zeros(B, k, device=self.device)
                            oh[:, t_idx] = 1.0
                            a_s = torch.cat([drive_s, oh], dim=-1)
                            qs.append(self.qcritics[role](g, a_s))
                        q_tool = torch.stack(qs, dim=-1)              # (B,k)
                        q_exp = (p * q_tool).sum(-1)                  # Σ_t p_t Q
                        h_tool = -(p * (p + 1e-8).log()).sum(-1)      # 离散熵
                    pi_loss = (self.alpha * logp_drive - self.alpha * h_tool - q_exp).mean()
                    ent_j = h_tool - logp_drive

                if self.ref_actors is not None:
                    dm_ref, _, tl_ref = self.ref_actors[role](obs[:, j, :])
                    kl = ((dm - dm_ref) ** 2).mean(-1)
                    if k > 0:
                        p = torch.softmax(tl[:, :k], dim=-1)
                        q = torch.softmax(tl_ref[:, :k], dim=-1)
                        kl = kl + (p * (p.log() - q.log())).sum(-1)
                    pi_loss = pi_loss + self.kl_coef * kl.mean()

                self.actor_opts[role].zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[role].parameters(), 0.5)
                self.actor_opts[role].step()

                q_loss_sum += q_loss.item()
                pi_loss_sum += pi_loss.item()
                ent_sum += ent_j.mean().item()

            self._soft_update()

            stats["q_loss"] += q_loss_sum / 3
            stats["policy_loss"] += pi_loss_sum / 3
            stats["entropy"] += ent_sum / 3
        for k in stats:
            stats[k] /= updates
        return stats

    def _soft_update(self):
        with torch.no_grad():
            for role in ROBOT_NAMES:
                for p_t, p in zip(self.q_targets[role].parameters(),
                                  self.qcritics[role].parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
