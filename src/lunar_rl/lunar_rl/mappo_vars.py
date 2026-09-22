"""变规模场景的 MAPPO / HAPPO（反事实优势）——把固定场景的 mappo.py 适配到可变实体观测。

支持任意数量机器人（同角色多实例）、多 depot、异构资源价值。
- Actor：`VarRoleAwareActor`（掩码注意力，var=True）或 `FlatActor`（= Actor + 展平定长，var=False）。
- 集中 Critic/QCritic 输入 = 3 车展平观测拼接（global_flat）+ 联合动作（dim = joint_action_dim()）。
- `hap=True` 用 HAPPO 顺序更新 + 反事实优势 A_i = Q(s,a) − Q(s,a_cf_i)。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from lunar_rl.networks import actions_to_dict, encode_joint_action
from lunar_rl.obs import encode_obs_vars, flat_vars, flat_vars_dim
from lunar_rl.registry import (ROBOT_NAMES, ROBOT_ROLE, ROLE_TOOL_DIM, joint_action_dim,
                               tool_offsets)

MAX_RES = 8
MAX_TEAM = 5      # 6 机器人 → 5 队友（默认 6 台场景）；3 台场景时多出的掩码为 0
MAX_DEPOT = 2
FLAT_DIM = flat_vars_dim(MAX_RES, MAX_TEAM, MAX_DEPOT)
GLOBAL_DIM = len(ROBOT_NAMES) * FLAT_DIM


def encode_agents(raw_obs):
    """把 raw obs 编码成 (fixed, res_ent, depot_ent, team_ent, 各 mask, flat, global_flat)。"""
    enc = encode_obs_vars(raw_obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
    fixed = np.stack([enc[n]["fixed"] for n in ROBOT_NAMES]).astype(np.float32)
    res_ent = np.stack([enc[n]["res_ent"] for n in ROBOT_NAMES]).astype(np.float32)
    depot_ent = np.stack([enc[n]["depot_ent"] for n in ROBOT_NAMES]).astype(np.float32)
    team_ent = np.stack([enc[n]["team_ent"] for n in ROBOT_NAMES]).astype(np.float32)
    res_mask = np.stack([enc[n]["res_mask"] for n in ROBOT_NAMES]).astype(np.float32)
    depot_mask = np.stack([enc[n]["depot_mask"] for n in ROBOT_NAMES]).astype(np.float32)
    team_mask = np.stack([enc[n]["team_mask"] for n in ROBOT_NAMES]).astype(np.float32)
    flat = np.stack([flat_vars(enc[n]) for n in ROBOT_NAMES]).astype(np.float32)
    return fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask, flat, flat.reshape(-1)


def _actor_forward(actor, fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask,
                   flat, var):
    if var:
        return actor(fixed, res_ent, team_ent, res_mask, team_mask, depot_ent, depot_mask)
    return actor(flat)


def sample_actions_vars(actors, fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask,
                        team_mask, flat, var):
    """逐 Agent 采样动作。返回 (drive (B,N,2), tools list, logprob (B,N))。"""
    B = fixed.shape[0]
    drives, tools, logprobs = [], [], []
    with torch.no_grad():
        for j, name in enumerate(ROBOT_NAMES):
            dm, ds, tl = _actor_forward(actors[name], fixed[:, j], res_ent[:, j], depot_ent[:, j],
                                        team_ent[:, j], res_mask[:, j], depot_mask[:, j],
                                        team_mask[:, j], flat[:, j], var)
            dist = Normal(dm, ds)
            u = dist.sample()
            drive = torch.clamp(u, -1.0, 1.0)
            logprob = dist.log_prob(u).sum(-1)
            tool = torch.zeros(B, dtype=torch.long)
            k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
            if k > 0:
                d = Categorical(logits=tl[:, :k])
                tool = d.sample()
                logprob = logprob + d.log_prob(tool)
            drives.append(drive)
            tools.append(tool)
            logprobs.append(logprob)
    return torch.stack(drives, dim=1), tools, torch.stack(logprobs, dim=1)


def eval_actions_vars(actors, fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask,
                      team_mask, flat, var, drive, tools):
    """给定动作重算 logprob。drive:(B,N,2)；tools:(B,N) long tensor。"""
    logprobs = []
    for j, name in enumerate(ROBOT_NAMES):
        dm, ds, tl = _actor_forward(actors[name], fixed[:, j], res_ent[:, j], depot_ent[:, j],
                                    team_ent[:, j], res_mask[:, j], depot_mask[:, j],
                                    team_mask[:, j], flat[:, j], var)
        dist = Normal(dm, ds)
        logprob = dist.log_prob(drive[:, j]).sum(-1)
        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
        if k > 0:
            d = Categorical(logits=tl[:, :k])
            logprob = logprob + d.log_prob(tools[:, j])
        logprobs.append(logprob)
    return torch.stack(logprobs, dim=1)


def _act_det(actors, raw_obs, var, device="cpu"):
    """确定性动作 dict（评估用）。"""
    fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask, flat, _ = encode_agents(raw_obs)
    with torch.no_grad():
        drives, tools = [], []
        for j, name in enumerate(ROBOT_NAMES):
            dm, _, tl = _actor_forward(
                actors[name],
                torch.tensor(fixed[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(res_ent[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(depot_ent[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(team_ent[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(res_mask[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(depot_mask[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(team_mask[j:j + 1], dtype=torch.float32, device=device),
                torch.tensor(flat[j:j + 1], dtype=torch.float32, device=device),
                var)
            drives.append(torch.clamp(dm[0], -1.0, 1.0))
            k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
            tools.append(int(torch.argmax(tl[0, :k]).item()) if k > 0 else 0)
    return actions_to_dict(torch.stack(drives), tools)


def _counterfactual(joint_action, j, dm_mean, tool_idx):
    """把第 j 个 agent 的动作换成反事实动作（均值 drive + argmax tool）。"""
    a_cf = joint_action.clone()
    a_cf[2 * j:2 * j + 2] = dm_mean
    name = ROBOT_NAMES[j]
    k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
    if k > 0:
        off = tool_offsets()[name]
        a_cf[off:off + k] = 0.0
        a_cf[off + tool_idx] = 1.0
    return a_cf


class MAPPOVars:
    def __init__(self, actors, critic, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.001, value_coef=0.5, epochs=4,
                 device="cpu", ref_actors=None, kl_coef=0.0, hap=False,
                 qcritic=None, critic_lr=None, var=True, cql_coef=0.0):
        self.actors = {k: v.to(device) for k, v in actors.items()}
        self.critic = critic.to(device)
        self.device = device
        self.gamma = gamma
        self.cql_coef = cql_coef
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.var = var
        critic_lr = lr if critic_lr is None else critic_lr
        self.ref_actors = ref_actors
        if self.ref_actors is not None:
            for a in self.ref_actors.values():
                for p in a.parameters():
                    p.requires_grad_(False)
        self.kl_coef = kl_coef
        self.hap = hap
        self.qcritic = qcritic.to(device) if qcritic is not None else None
        self.N = len(ROBOT_NAMES)
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

    # ---------- 动作前向（通用） ----------
    def _fwd(self, actor, fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask,
             flat):
        return _actor_forward(actor, fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask,
                              team_mask, flat, self.var)

    def warmup_critic(self, env, episodes=20, epochs=30, lr=1e-3):
        opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        qopt = torch.optim.Adam(self.qcritic.parameters(), lr=lr) \
            if self.qcritic is not None else None
        mse = nn.MSELoss()
        xs, ys, jas = [], [], []
        for _ in range(episodes):
            obs, _ = env.reset()
            traj = []
            while True:
                fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask, flat, g = \
                    encode_agents(obs)
                f = torch.tensor(fixed, dtype=torch.float32, device=self.device)
                re = torch.tensor(res_ent, dtype=torch.float32, device=self.device)
                de = torch.tensor(depot_ent, dtype=torch.float32, device=self.device)
                te = torch.tensor(team_ent, dtype=torch.float32, device=self.device)
                rm = torch.tensor(res_mask, dtype=torch.float32, device=self.device)
                dm = torch.tensor(depot_mask, dtype=torch.float32, device=self.device)
                tm = torch.tensor(team_mask, dtype=torch.float32, device=self.device)
                fl = torch.tensor(flat, dtype=torch.float32, device=self.device)
                drive, tools, _ = sample_actions_vars(self.actors, f[None], re[None], de[None],
                                                      te[None], rm[None], dm[None], tm[None],
                                                      fl[None], self.var)
                ja = encode_joint_action(drive[0], [t[0] for t in tools]).detach().cpu()
                action = actions_to_dict(drive[0], [t[0] for t in tools])
                obs, reward, term, trunc, _ = env.step(action)
                traj.append((g, ja, float(reward), bool(term or trunc)))
                if term or trunc:
                    break
            G = 0.0
            for g, ja, r, done in reversed(traj):
                G = r + self.gamma * G * (1 - int(done))
                xs.append(g); ys.append(G); jas.append(ja)
        xs = torch.tensor(np.array(xs, dtype=np.float32), device=self.device)
        ys = torch.tensor(np.array(ys, dtype=np.float32), device=self.device)
        jas = torch.stack(jas).to(self.device)
        n = len(xs)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            total = 0.0; qtotal = 0.0
            for i in range(0, n, 256):
                b = idx[i:i + 256]
                pred = self.critic(xs[b]); loss = mse(pred, ys[b])
                opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
                if qopt is not None:
                    qpred = self.qcritic(xs[b], jas[b]); qloss = mse(qpred, ys[b])
                    qopt.zero_grad(); qloss.backward(); qopt.step(); qtotal += qloss.item()
            if (ep + 1) % 10 == 0:
                msg = f"[critic-warmup] epoch={ep+1} loss={total/max(1, n//256):.4f}"
                if qopt is not None:
                    msg += f" qloss={qtotal/max(1, n//256):.4f}"
                print(msg, flush=True)
        return self

    def rollout(self, env, steps):
        obs, _ = env.reset()
        buf = []
        for _ in range(steps):
            fixed, res_ent, depot_ent, team_ent, res_mask, depot_mask, team_mask, flat, g = \
                encode_agents(obs)
            f = torch.tensor(fixed, dtype=torch.float32, device=self.device)
            re = torch.tensor(res_ent, dtype=torch.float32, device=self.device)
            de = torch.tensor(depot_ent, dtype=torch.float32, device=self.device)
            te = torch.tensor(team_ent, dtype=torch.float32, device=self.device)
            rm = torch.tensor(res_mask, dtype=torch.float32, device=self.device)
            dm = torch.tensor(depot_mask, dtype=torch.float32, device=self.device)
            tm = torch.tensor(team_mask, dtype=torch.float32, device=self.device)
            fl = torch.tensor(flat, dtype=torch.float32, device=self.device)
            g_ = torch.tensor(g, dtype=torch.float32, device=self.device).unsqueeze(0)

            drive, tools, logprob = sample_actions_vars(self.actors, f[None], re[None], de[None],
                                                        te[None], rm[None], dm[None], tm[None],
                                                        fl[None], self.var)
            joint_action = encode_joint_action(drive[0], [t[0] for t in tools]).detach().cpu()
            if self.qcritic is not None:
                ja = encode_joint_action(drive[0], [t[0] for t in tools]).unsqueeze(0)
                value = self.qcritic(g_, ja)
            else:
                value = self.critic(g_)

            action = actions_to_dict(drive[0], [t[0] for t in tools])
            obs2, reward, term, trunc, _ = env.step(action)

            buf.append({
                "fixed": fixed, "res_ent": res_ent, "depot_ent": depot_ent, "team_ent": team_ent,
                "res_mask": res_mask, "depot_mask": depot_mask, "team_mask": team_mask, "flat": flat,
                "global": g, "drive": drive[0].detach().cpu().numpy(),
                "tools": np.array([int(t[0]) for t in tools], dtype=np.int64),
                "logprob": logprob[0].detach().cpu().numpy(),
                "joint_action": joint_action, "reward": float(reward),
                "value": value.detach().cpu(), "done": bool(term or trunc),
            })
            obs = obs2
            if term or trunc:
                obs, _ = env.reset()
        return buf

    def _gae(self, buf):
        rewards = [b["reward"] for b in buf]
        values = [b["value"].item() for b in buf]
        dones = [b["done"] for b in buf]
        advs, gae = [], 0.0
        next_value = 0.0
        for i in reversed(range(len(buf))):
            delta = rewards[i] + self.gamma * next_value * (1 - int(dones[i])) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[i])) * gae
            advs.insert(0, gae)
            next_value = values[i]
        advs = torch.tensor(advs, dtype=torch.float32)
        returns = advs + torch.tensor(values, dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs, returns

    def update(self, buf):
        advs, returns = self._gae(buf)
        advs = advs.to(self.device); returns = returns.to(self.device)
        n = len(buf)
        fixed = torch.tensor(np.stack([b["fixed"] for b in buf]), dtype=torch.float32, device=self.device)
        res_ent = torch.tensor(np.stack([b["res_ent"] for b in buf]), dtype=torch.float32, device=self.device)
        depot_ent = torch.tensor(np.stack([b["depot_ent"] for b in buf]), dtype=torch.float32, device=self.device)
        team_ent = torch.tensor(np.stack([b["team_ent"] for b in buf]), dtype=torch.float32, device=self.device)
        res_mask = torch.tensor(np.stack([b["res_mask"] for b in buf]), dtype=torch.float32, device=self.device)
        depot_mask = torch.tensor(np.stack([b["depot_mask"] for b in buf]), dtype=torch.float32, device=self.device)
        team_mask = torch.tensor(np.stack([b["team_mask"] for b in buf]), dtype=torch.float32, device=self.device)
        flat = torch.tensor(np.stack([b["flat"] for b in buf]), dtype=torch.float32, device=self.device)
        g_obs = torch.tensor(np.stack([b["global"] for b in buf]), dtype=torch.float32, device=self.device)
        old_drive = torch.tensor(np.stack([b["drive"] for b in buf]), dtype=torch.float32, device=self.device)
        old_logprob = torch.tensor(np.stack([b["logprob"] for b in buf]), dtype=torch.float32, device=self.device)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.epochs):
            if self.hap:
                plosses, ent_sum = [], 0.0
                if self.qcritic is not None:
                    joint_actions = torch.stack([b["joint_action"] for b in buf]).to(self.device)
                    a_mean = None
                    with torch.no_grad():
                        q_actual = self.qcritic(g_obs, joint_actions)
                        if self.cql_coef > 0.0:
                            dms = []
                            for j, name in enumerate(ROBOT_NAMES):
                                dm, _, _ = self._fwd(self.actors[name], fixed[:, j], res_ent[:, j],
                                                     depot_ent[:, j], team_ent[:, j], res_mask[:, j],
                                                     depot_mask[:, j], team_mask[:, j], flat[:, j])
                                dms.append(dm)
                            a_mean = torch.cat(dms, dim=-1)  # (n, 2N)
                            for j, name in enumerate(ROBOT_NAMES):
                                k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
                                if k > 0:
                                    _, _, tl = self._fwd(self.actors[name], fixed[:, j], res_ent[:, j],
                                                         depot_ent[:, j], team_ent[:, j], res_mask[:, j],
                                                         depot_mask[:, j], team_mask[:, j], flat[:, j])
                                    oh = torch.zeros(n, k, device=self.device)
                                    oh[torch.arange(n), torch.argmax(tl[:, :k], dim=-1)] = 1.0
                                    a_mean = torch.cat([a_mean, oh], dim=-1)
                    q_loss = ((self.qcritic(g_obs, joint_actions) - returns) ** 2).mean()
                    if self.cql_coef > 0.0 and a_mean is not None:
                        q_loss = q_loss + self.cql_coef * self.qcritic(g_obs, a_mean).mean()
                    self.qcritic_opt.zero_grad(); q_loss.backward()
                    nn.utils.clip_grad_norm_(self.qcritic.parameters(), 0.5)
                    self.qcritic_opt.step()
                    value_loss = q_loss.detach()
                for j, name in enumerate(ROBOT_NAMES):
                    ratios, ents, kls, advs_j = [], [], [], []
                    for i in range(n):
                        dm, ds, tl = self._fwd(self.actors[name], fixed[i, j:j + 1], res_ent[i, j:j + 1],
                                               depot_ent[i, j:j + 1], team_ent[i, j:j + 1],
                                               res_mask[i, j:j + 1], depot_mask[i, j:j + 1],
                                               team_mask[i, j:j + 1], flat[i, j:j + 1])
                        dist = Normal(dm[0], ds)
                        new_lp = dist.log_prob(old_drive[i, j]).sum()
                        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
                        if k > 0:
                            d = Categorical(logits=tl[0, :k])
                            new_lp = new_lp + d.log_prob(torch.tensor(buf[i]["tools"][j], device=self.device))
                        ratios.append(torch.exp(new_lp - old_logprob[i, j]))
                        ent_j = dist.entropy().sum()
                        if k > 0:
                            ent_j = ent_j + Categorical(logits=tl[0, :k]).entropy()
                        ents.append(ent_j)
                        if self.qcritic is not None:
                            tool_idx = int(torch.argmax(tl[0, :k]).item()) if k > 0 else 0
                            a_cf = _counterfactual(buf[i]["joint_action"].to(self.device), j,
                                                   dm[0].detach(), tool_idx)
                            with torch.no_grad():
                                q_cf = self.qcritic(g_obs[i:i + 1], a_cf.unsqueeze(0))
                            advs_j.append((q_actual[i] - q_cf).detach())
                        if self.ref_actors is not None:
                            with torch.no_grad():
                                dm_ref, _, tl_ref = self._fwd(self.ref_actors[name],
                                                              fixed[i, j:j + 1], res_ent[i, j:j + 1],
                                                              depot_ent[i, j:j + 1], team_ent[i, j:j + 1],
                                                              res_mask[i, j:j + 1], depot_mask[i, j:j + 1],
                                                              team_mask[i, j:j + 1], flat[i, j:j + 1])
                            kl_j = ((dm - dm_ref) ** 2).mean()
                            if k > 0:
                                p = torch.softmax(tl[0, :k], dim=-1)
                                q = torch.softmax(tl_ref[0, :k], dim=-1)
                                kl_j = kl_j + (p * (p.log() - q.log())).sum()
                            kls.append(kl_j)
                    ratio_j = torch.stack(ratios)
                    ent_j = torch.stack(ents).mean()
                    if self.qcritic is not None:
                        adv_j = torch.stack(advs_j)
                        adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    else:
                        adv_j = advs
                    surr1 = ratio_j * adv_j
                    surr2 = torch.clamp(ratio_j, 1 - self.clip_eps, 1 + self.clip_eps) * adv_j
                    ploss_j = -torch.min(surr1, surr2).mean()
                    loss_j = ploss_j - self.entropy_coef * ent_j
                    if self.ref_actors is not None:
                        loss_j = loss_j + self.kl_coef * torch.stack(kls).mean()
                    self.actor_opts[name].zero_grad(); loss_j.backward()
                    nn.utils.clip_grad_norm_(self.actors[name].parameters(), 0.5)
                    self.actor_opts[name].step()
                    plosses.append(ploss_j.detach()); ent_sum = ent_sum + ent_j.detach()
                policy_loss = torch.stack(plosses).mean(); entropy = ent_sum
                if self.qcritic is None:
                    values = self.critic(g_obs)
                    value_loss = ((values - returns) ** 2).mean()
                    self.critic_opt.zero_grad()
                    (self.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_opt.step()
            else:
                old_tools = torch.tensor(np.stack([b["tools"] for b in buf]), dtype=torch.long,
                                         device=self.device)
                new_logprob = eval_actions_vars(self.actors, fixed, res_ent, depot_ent, team_ent,
                                                res_mask, depot_mask, team_mask, flat, self.var,
                                                old_drive, old_tools)
                ratio = torch.exp(new_logprob - old_logprob)
                ent_terms = []
                for j, name in enumerate(ROBOT_NAMES):
                    dm, ds, tl = self._fwd(self.actors[name], fixed[:, j], res_ent[:, j],
                                           depot_ent[:, j], team_ent[:, j], res_mask[:, j],
                                           depot_mask[:, j], team_mask[:, j], flat[:, j])
                    ent_j = Normal(dm, ds).entropy().sum(-1)
                    k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
                    if k > 0:
                        ent_j = ent_j + Categorical(logits=tl[:, :k]).entropy()
                    ent_terms.append(ent_j)
                entropy = sum(e.mean() for e in ent_terms)
                advs_ = advs.unsqueeze(-1)
                surr1 = ratio * advs_
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_
                policy_loss = -torch.min(surr1, surr2).mean()
                values = self.critic(g_obs)
                value_loss = ((values - returns) ** 2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if self.ref_actors is not None:
                    kl_all = 0.0
                    for j, name in enumerate(ROBOT_NAMES):
                        dm, _, tl = self._fwd(self.actors[name], fixed[:, j], res_ent[:, j],
                                              depot_ent[:, j], team_ent[:, j], res_mask[:, j],
                                              depot_mask[:, j], team_mask[:, j], flat[:, j])
                        with torch.no_grad():
                            dm_ref, _, tl_ref = self._fwd(self.ref_actors[name], fixed[:, j],
                                                          res_ent[:, j], depot_ent[:, j], team_ent[:, j],
                                                          res_mask[:, j], depot_mask[:, j],
                                                          team_mask[:, j], flat[:, j])
                        kl_j = ((dm - dm_ref) ** 2).mean(-1)
                        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
                        if k > 0:
                            p = torch.softmax(tl[:, :k], dim=-1)
                            q = torch.softmax(tl_ref[:, :k], dim=-1)
                            kl_j = kl_j + (p * (p.log() - q.log())).sum(-1)
                        kl_all = kl_all + kl_j.mean()
                    loss = loss + self.kl_coef * kl_all
                self.opt.zero_grad(); loss.backward()
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
        return _act_det(self.actors, obs, self.var, self.device)
