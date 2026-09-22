"""异构 MAPPO：每车一个独立 Actor + 一个集中 Critic，共享回报，GAE + PPO。

- rollout：与环境交互，收集 (obs, 动作, logprob, 回报, 值, done)。
- update：GAE 计算优势 → 多 epoch PPO 更新（逐 Agent ratio × 共享优势 + value loss + entropy）。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from lunar_rl.networks import (ROBOT_NAMES, ROLE_TOOL_DIM, actions_to_dict,
                               encode_joint_action, eval_actions, sample_actions)
from lunar_rl.obs import encode_global, encode_obs


class MAPPO:
    def __init__(self, actors, critic, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.001, value_coef=0.5, epochs=4,
                 device="cpu", ref_actors=None, kl_coef=0.0, hap=False,
                 qcritic=None, critic_lr=None, cql_coef=0.0):
        self.actors = {k: v.to(device) for k, v in actors.items()}
        self.critic = critic.to(device)
        self.device = device
        self.gamma = gamma
        self.cql_coef = cql_coef  # Cal-QL/CQL 保守 Q 惩罚系数（对 OOD 均值动作的 Q 加惩罚）
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        # Critic/QCritic 用独立学习率（默认与 actor 相同）；反事实 Q 拟合需要更大 lr 才能跟住回报
        critic_lr = lr if critic_lr is None else critic_lr
        # 参考策略（通常是 BC/DAgger 预热后的 actors），用于 KL/L2 正则，防止 RL 把好策略洗掉
        self.ref_actors = ref_actors
        if self.ref_actors is not None:
            for a in self.ref_actors.values():
                for p in a.parameters():
                    p.requires_grad_(False)
        self.kl_coef = kl_coef
        self.hap = hap  # True=HAPPO 顺序更新（逐 Agent 独立优化器），False=MAPPO 同时更新
        # Q 函数（反事实基线）：仅 HAPPO 用；为 None 时退化为共享 V(s) 优势
        self.qcritic = qcritic.to(device) if qcritic is not None else None
        if hap:
            self.actor_opts = {k: torch.optim.Adam(v.parameters(), lr=lr)
                               for k, v in self.actors.items()}
            self.critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
            if self.qcritic is not None:
                self.qcritic_opt = torch.optim.Adam(self.qcritic.parameters(), lr=critic_lr)
        else:
            all_params = [p for a in self.actors.values() for p in a.parameters()]
            all_params += list(critic.parameters())
            if self.qcritic is not None:
                all_params += list(self.qcritic.parameters())
            self.opt = torch.optim.Adam(all_params, lr=lr)

    def warmup_critic(self, env, episodes=20, epochs=30, lr=1e-3):
        """用当前策略的（与 rollout 一致的随机）采样 rollout 做蒙特卡洛回报，预拟合集中 Critic。

        避免随机初始化的 critic 在早期产生噪声优势、把 BC 学到的策略洗掉。
        """
        opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        qopt = torch.optim.Adam(self.qcritic.parameters(), lr=lr) \
            if self.qcritic is not None else None
        mse = nn.MSELoss()
        xs, ys, jas = [], [], []
        for _ in range(episodes):
            obs, _ = env.reset()
            traj = []
            while True:
                enc = encode_obs(obs)
                g = encode_global(enc)
                ob = torch.tensor(
                    np.stack([enc[n] for n in ROBOT_NAMES]), dtype=torch.float32,
                    device=self.device)
                drive, tools, _ = sample_actions(self.actors, ob, ROBOT_NAMES)
                ja = encode_joint_action(drive, tools).detach().cpu()
                action = actions_to_dict(drive, tools)
                obs, reward, term, trunc, _ = env.step(action)
                traj.append((g, ja, float(reward), bool(term or trunc)))
                if term or trunc:
                    break
            G = 0.0
            for g, ja, r, done in reversed(traj):
                G = r + self.gamma * G * (1 - int(done))
                xs.append(g)
                ys.append(G)
                jas.append(ja)
        xs = torch.tensor(np.array(xs, dtype=np.float32), device=self.device)
        ys = torch.tensor(np.array(ys, dtype=np.float32), device=self.device)
        jas = torch.stack(jas).to(self.device)
        n = len(xs)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            total = 0.0
            qtotal = 0.0
            for i in range(0, n, 256):
                b = idx[i:i + 256]
                pred = self.critic(xs[b])
                loss = mse(pred, ys[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                if qopt is not None:
                    qpred = self.qcritic(xs[b], jas[b])
                    qloss = mse(qpred, ys[b])
                    qopt.zero_grad()
                    qloss.backward()
                    qopt.step()
                    qtotal += qloss.item()
            if (ep + 1) % 10 == 0:
                msg = f"[critic-warmup] epoch={ep+1} loss={total/max(1, n//256):.4f}"
                if qopt is not None:
                    msg += f" qloss={qtotal/max(1, n//256):.4f}"
                print(msg, flush=True)
        return self

    def rollout(self, env, steps):
        """收集 steps 步轨迹。"""
        obs, _ = env.reset()
        buf = []
        for _ in range(steps):
            enc = encode_obs(obs)
            obs_batch = torch.tensor(
                np.stack([enc[n] for n in ROBOT_NAMES]), dtype=torch.float32,
                device=self.device)
            g = torch.tensor(encode_global(enc), dtype=torch.float32,
                             device=self.device).unsqueeze(0)

            drive, tools, logprob = sample_actions(self.actors, obs_batch, ROBOT_NAMES)
            joint_action = encode_joint_action(drive, tools).detach().cpu()
            if self.qcritic is not None:
                ja = encode_joint_action(drive, tools).unsqueeze(0)
                value = self.qcritic(g, ja)
            else:
                value = self.critic(g)

            action = actions_to_dict(drive, tools)
            obs2, reward, term, trunc, info = env.step(action)

            buf.append({
                "obs_batch": obs_batch.detach().cpu(),
                "global_obs": g.detach().cpu(),
                "drive": drive.detach().cpu(),
                "tools": tools,
                "joint_action": joint_action,
                "logprob": logprob.detach().cpu(),
                "reward": float(reward),
                "value": value.detach().cpu(),
                "done": bool(term or trunc),
            })
            obs = obs2
            if term or trunc:
                obs, _ = env.reset()
        return buf

    def _gae(self, buf):
        rewards = [b["reward"] for b in buf]
        values = [b["value"].item() for b in buf]
        dones = [b["done"] for b in buf]
        advs = []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(buf))):
            delta = rewards[i] + self.gamma * next_value * (1 - int(dones[i])) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[i])) * gae
            advs.insert(0, gae)
            next_value = values[i]
        advs = torch.tensor(advs, dtype=torch.float32)
        returns = advs + torch.tensor(values, dtype=torch.float32)
        # 优势标准化
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs, returns

    def update(self, buf):
        """PPO 更新（逐 Agent ratio × 共享优势），返回损失统计。"""
        advs, returns = self._gae(buf)
        advs = advs.to(self.device)
        returns = returns.to(self.device)

        n = len(buf)
        obs_batch = torch.stack([b["obs_batch"] for b in buf]).to(self.device)  # (n,3,obs_dim)
        g_obs = torch.cat([b["global_obs"] for b in buf]).to(self.device)       # (n,93)
        old_drive = torch.stack([b["drive"] for b in buf]).to(self.device)      # (n,3,2)
        old_logprob = torch.stack([b["logprob"] for b in buf]).to(self.device)  # (n,3)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.epochs):
            total_ratio = []                                  # (n,3) 逐 Agent ratio
            ent_terms = {r: [] for r in ROBOT_NAMES}
            kl_terms = {r: [] for r in ROBOT_NAMES}
            for i in range(n):
                ob = obs_batch[i]  # (3, obs_dim)
                new_logprob = eval_actions(self.actors, ob, ROBOT_NAMES,
                                           old_drive[i], buf[i]["tools"])  # (3,)
                ratio = torch.exp(new_logprob - old_logprob[i])            # (3,)
                total_ratio.append(ratio)
                for j, role in enumerate(ROBOT_NAMES):
                    dm, ds, tl = self.actors[role](ob[j:j + 1])
                    ent_j = Normal(dm[0], ds).entropy().sum()
                    k = ROLE_TOOL_DIM[role]
                    if k > 0:
                        ent_j = ent_j + Categorical(logits=tl[0, :k]).entropy()
                    ent_terms[role].append(ent_j)
                    if self.ref_actors is not None:
                        with torch.no_grad():
                            dm_ref, _, tl_ref = self.ref_actors[role](ob[j:j + 1])
                        kl_j = ((dm - dm_ref) ** 2).mean()
                        if k > 0:
                            p = torch.softmax(tl[0, :k], dim=-1)
                            q = torch.softmax(tl_ref[0, :k], dim=-1)
                            kl_j = kl_j + (p * (p.log() - q.log())).sum()
                        kl_terms[role].append(kl_j)

            ratio = torch.stack(total_ratio)                # (n, 3)
            ent_per = {r: torch.stack(ent_terms[r]).mean() for r in ROBOT_NAMES}
            entropy = sum(ent_per[r] for r in ROBOT_NAMES)  # 各车熵之和的步均值（与旧版一致）

            if self.hap:
                # HAPPO：顺序更新 + 反事实优势 A_i = Q(s,a) − Q(s,a_cf_i)
                plosses = []
                ent_sum = 0.0
                # 预计算 Q(s, 实际动作) + 拟合 QCritic（可加 Cal-QL/CQL 保守 OOD 惩罚）
                if self.qcritic is not None:
                    joint_actions = torch.stack([b["joint_action"] for b in buf]).to(self.device)
                    a_mean = None
                    with torch.no_grad():
                        q_actual = self.qcritic(g_obs, joint_actions)   # (n,)
                        if self.cql_coef > 0.0:
                            # 确定性均值联合动作（OOD：各车均值 drive + argmax tool）
                            drive_means = []
                            for jj in range(3):
                                dm, _, _ = self.actors[ROBOT_NAMES[jj]](obs_batch[:, jj])
                                drive_means.append(dm)
                            drive_mean_all = torch.cat(drive_means, dim=-1)   # (n,6)
                            _, _, tl_ex = self.actors["excavator"](obs_batch[:, 1])
                            oh_ex = torch.zeros(n, 3, device=self.device)
                            oh_ex[torch.arange(n), torch.argmax(tl_ex[:, :3], dim=-1)] = 1.0
                            _, _, tl_ha = self.actors["hauler"](obs_batch[:, 2])
                            oh_ha = torch.zeros(n, 2, device=self.device)
                            oh_ha[torch.arange(n), torch.argmax(tl_ha[:, :2], dim=-1)] = 1.0
                            a_mean = torch.cat([drive_mean_all, oh_ex, oh_ha], dim=-1)  # (n,11)
                    q_loss = ((self.qcritic(g_obs, joint_actions) - returns) ** 2).mean()
                    if self.cql_coef > 0.0 and a_mean is not None:
                        q_loss = q_loss + self.cql_coef * self.qcritic(g_obs, a_mean).mean()
                    self.qcritic_opt.zero_grad()
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(self.qcritic.parameters(), 0.5)
                    self.qcritic_opt.step()
                    value_loss = q_loss.detach()
                for j, role in enumerate(ROBOT_NAMES):
                    ratios, ents, kls, advs_j = [], [], [], []
                    for i in range(n):
                        ob = obs_batch[i]
                        dm, ds, tl = self.actors[role](ob[j:j + 1])
                        dist = Normal(dm[0], ds)
                        new_lp = dist.log_prob(old_drive[i][j]).sum()
                        k = ROLE_TOOL_DIM[role]
                        if k > 0 and buf[i]["tools"][j] is not None:
                            d = Categorical(logits=tl[0, :k])
                            new_lp = new_lp + d.log_prob(buf[i]["tools"][j])
                        ratios.append(torch.exp(new_lp - old_logprob[i][j]))
                        ent_j = dist.entropy().sum()
                        if k > 0:
                            ent_j = ent_j + Categorical(logits=tl[0, :k]).entropy()
                        ents.append(ent_j)
                        # 反事实动作（mean drive + argmax tool）→ A_i
                        if self.qcritic is not None:
                            a_cf = buf[i]["joint_action"].clone().to(self.device)
                            a_cf[2 * j:2 * j + 2] = dm[0].detach()
                            if role == "excavator":
                                a_cf[6:9] = 0.0
                                a_cf[6 + int(torch.argmax(tl[0, :3]))] = 1.0
                            elif role == "hauler":
                                a_cf[9:11] = 0.0
                                a_cf[9 + int(torch.argmax(tl[0, :2]))] = 1.0
                            with torch.no_grad():
                                q_cf = self.qcritic(g_obs[i:i + 1], a_cf.unsqueeze(0))
                            advs_j.append((q_actual[i] - q_cf).detach())
                        if self.ref_actors is not None:
                            with torch.no_grad():
                                dm_ref, _, tl_ref = self.ref_actors[role](ob[j:j + 1])
                            kl_j = ((dm - dm_ref) ** 2).mean()
                            if k > 0:
                                p = torch.softmax(tl[0, :k], dim=-1)
                                q = torch.softmax(tl_ref[0, :k], dim=-1)
                                kl_j = kl_j + (p * (p.log() - q.log())).sum()
                            kls.append(kl_j)
                    ratio_j = torch.stack(ratios)          # (n,)
                    ent_j = torch.stack(ents).mean()
                    if self.qcritic is not None:
                        adv_j = torch.stack(advs_j)         # (n,) 反事实优势
                        adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    else:
                        adv_j = advs                        # 共享 GAE 优势
                    surr1 = ratio_j * adv_j
                    surr2 = torch.clamp(ratio_j, 1 - self.clip_eps, 1 + self.clip_eps) * adv_j
                    ploss_j = -torch.min(surr1, surr2).mean()
                    loss_j = ploss_j - self.entropy_coef * ent_j
                    if self.ref_actors is not None:
                        loss_j = loss_j + self.kl_coef * torch.stack(kls).mean()
                    self.actor_opts[role].zero_grad()
                    loss_j.backward()
                    nn.utils.clip_grad_norm_(self.actors[role].parameters(), 0.5)
                    self.actor_opts[role].step()
                    plosses.append(ploss_j.detach())
                    ent_sum = ent_sum + ent_j.detach()
                policy_loss = torch.stack(plosses).mean()
                entropy = ent_sum
                # Critic 单独更新（仅当未用 QCritic 时）
                if self.qcritic is None:
                    values = self.critic(g_obs)
                    value_loss = ((values - returns) ** 2).mean()
                    self.critic_opt.zero_grad()
                    (self.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_opt.step()
            else:
                # MAPPO：同时更新（共享优势广播到 3 个 Agent）
                advs_ = advs.unsqueeze(-1)                  # (n,1)
                surr1 = ratio * advs_
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.critic(g_obs)
                value_loss = ((values - returns) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if self.ref_actors is not None:
                    kl_all = sum(torch.stack(kl_terms[r]).mean() for r in ROBOT_NAMES)
                    loss = loss + self.kl_coef * kl_all
                self.opt.zero_grad()
                loss.backward()
                for a in self.actors.values():
                    nn.utils.clip_grad_norm_(a.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()

        for k in stats:
            stats[k] /= self.epochs
        return stats

    def act(self, obs):
        """确定性动作（评估用，逐 Agent 取 drive 均值 + tool argmax）。"""
        enc = encode_obs(obs)
        obs_batch = torch.tensor(
            np.stack([enc[n] for n in ROBOT_NAMES]), dtype=torch.float32,
            device=self.device)
        drives, tools = [], []
        with torch.no_grad():
            for i, role in enumerate(ROBOT_NAMES):
                dm, _, tl = self.actors[role](obs_batch[i:i + 1])
                drives.append(torch.clamp(dm[0], -1.0, 1.0))
                k = ROLE_TOOL_DIM[role]
                tools.append(torch.argmax(tl[0, :k]) if k > 0 else None)
        return actions_to_dict(torch.stack(drives), tools)
