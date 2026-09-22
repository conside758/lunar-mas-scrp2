"""MAPPO/异构 MARL 的网络：每车独立 Actor + 集中 Critic。

Actor 输出三个头：
  - drive_mean: Linear(feat, 2)（连续驱动）
  - drive_logstd: 可学习参数 (2,)
  - tool_logits: Linear(feat, 3)（离散工具，excavator 用 3 类，hauler 用前 2 类，scout 忽略）

辅助函数 sample_actions / eval_actions 接收 actors 字典（{name: Actor}），逐 Agent 独立前向。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from lunar_rl.registry import (ROBOT_NAMES, ROBOT_ROLE, ROLE_ORDER, ROLE_TOOL_DIM,
                               robots_of_role)

# 逐 Agent 动作编码维度（drive 2 + tool one-hot）——HASAC 逐 Agent Q(s, a_i) 用
AGENT_ACTION_DIM = {r: 2 + ROLE_TOOL_DIM[r] for r in ROLE_ORDER}


def encode_agent_action(drive_i, tool_i, role):
    """把单个 agent 的动作编码成 [drive(2) + tool one-hot(k)] 向量（QCritic 输入）。

    drive_i: (2,) tensor；tool_i: int 或 None；role: 'scout'/'excavator'/'hauler'。
    """
    k = ROLE_TOOL_DIM[role]
    d = drive_i.reshape(-1)
    if k == 0:
        return d
    oh = torch.zeros(k, device=drive_i.device)
    t = int(tool_i) if tool_i is not None else 0
    oh[t] = 1.0
    return torch.cat([d, oh])


class Actor(nn.Module):
    def __init__(self, obs_dim=21, hidden=128, drive_logstd_init=-0.7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.drive_mean = nn.Linear(hidden, 2)
        # std = exp(logstd)；-0.7→0.50（探索），-2.0→0.14（细调微调）
        self.drive_logstd = nn.Parameter(torch.full((2,), drive_logstd_init))
        self.tool_logits = nn.Linear(hidden, 3)  # 最大 3 类（excavator）

    def forward(self, obs):
        f = self.net(obs)
        drive_mean = self.drive_mean(f)
        drive_std = torch.exp(self.drive_logstd).clamp(min=1e-3)
        tool_logits = self.tool_logits(f)
        return drive_mean, drive_std, tool_logits


class Critic(nn.Module):
    def __init__(self, global_dim=63, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs):
        return self.net(global_obs).squeeze(-1)


class QCritic(nn.Module):
    """集中 Q 函数 Q(s, a)：输入 = 全局观测 + 联合动作，输出标量 Q 值。

    用于 HAPPO 的 multi-agent advantage（反事实基线）：
      A_i = Q(s, a) − Q(s, a 把第 i 个 agent 的动作换成反事实动作)。
    联合动作编码（11 维）：drive(3×2=6) + excavator tool one-hot(3) + hauler tool one-hot(2)。
    """
    def __init__(self, global_dim=96, action_dim=11, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim + action_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs, action):
        return self.net(torch.cat([global_obs, action], dim=-1)).squeeze(-1)


def encode_joint_action(drive, tools):
    """把 (drive (N,2), tools list) 编码成 (2N + Σ tool_onehot) 维联合动作向量。

    drive 展平(2N) + 各 excavator tool one-hot(3) + 各 hauler tool one-hot(2)（按 ROBOT_NAMES 顺序）。
    tools: list 长度 N（与 ROBOT_NAMES 对齐），元素为 int 或 None。
    """
    parts = [drive.reshape(-1)]
    for i, name in enumerate(ROBOT_NAMES):
        k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
        if k > 0:
            oh = torch.zeros(k, device=drive.device)
            t = int(tools[i]) if tools[i] is not None else 0
            oh[t] = 1.0
            parts.append(oh)
    return torch.cat(parts)


def sample_actions(actors, obs_batch, roles):
    """逐 Agent 采样动作，返回 (drive (B,2), tools list, logprob (B,))。

    actors: dict[str, Actor]；obs_batch: (B, obs_dim)；roles: list[str] 长度 B。
    """
    drives, tools, logprobs = [], [], []
    for i, role in enumerate(roles):
        dm, ds, tl = actors[role](obs_batch[i:i + 1])   # (1, obs_dim)
        dist = Normal(dm[0], ds)
        drive = dist.sample()
        drive = torch.clamp(drive, -1.0, 1.0)
        logprob = dist.log_prob(drive).sum()
        tool = None
        n = ROLE_TOOL_DIM[role]
        if n > 0:
            d = Categorical(logits=tl[0, :n])
            tool = d.sample()
            logprob = logprob + d.log_prob(tool)
        drives.append(drive)
        tools.append(tool)
        logprobs.append(logprob)
    return torch.stack(drives), tools, torch.stack(logprobs)


def eval_actions(actors, obs_batch, roles, drive, tools):
    """给定动作，逐 Agent 重算 logprob（PPO 更新用）。

    drive: (B, 2)；tools: list 长度 B。
    """
    logprobs = []
    for i, role in enumerate(roles):
        dm, ds, tl = actors[role](obs_batch[i:i + 1])
        dist = Normal(dm[0], ds)
        logprob = dist.log_prob(drive[i]).sum()
        n = ROLE_TOOL_DIM[role]
        if n > 0 and tools[i] is not None:
            d = Categorical(logits=tl[0, :n])
            logprob = logprob + d.log_prob(tools[i])
        logprobs.append(logprob)
    return torch.stack(logprobs)


def actions_to_dict(drive, tools):
    """把张量动作转成 env 的 dict action（第 0 维对应 ROBOT_NAMES）。tools 元素可为 tensor/int/None。"""
    d = drive.detach().cpu().numpy()
    action = {}
    for i, name in enumerate(ROBOT_NAMES):
        sub = {"drive": d[i].astype(np.float32)}
        if ROLE_TOOL_DIM[ROBOT_ROLE[name]] > 0 and tools[i] is not None:
            t = tools[i]
            sub["tool"] = int(t.item()) if hasattr(t, "item") else int(t)
        action[name] = sub
    return action


class RoleAwareActor(nn.Module):
    """角色感知 Actor（HAPPO 第一步）：对资源/队友做可学习注意力池化，替代扁平拼接。

    观测布局（obs_dim=32，见 obs.py）：
      [0:4]   self [x,y,yaw,cargo]
      [4:7]   role one-hot
      [7:9]   rel_depot
      [9:12]  rel_nearest（已发现）
      [12:21] rel_res（3 资源 × [dx,dy,amount]）
      [21:24] det_mask（发现掩码）
      [24:28] team_rel（2 队友 × [dx,dy]）
      [28:32] rel_goal world + body

    把「资源实体(3×4=[dx,dy,amount,detected])」和「队友实体(2×2=[dx,dy])」用注意力加权求和，
    使每个角色的网络能按需聚焦「它这个角色该关注的实体」（Scout→未发现资源、Excavator→已发现资源+Hauler、
    Hauler→Excavator+depot）。
    """
    def __init__(self, obs_dim=32, hidden=128, attn_dim=64, drive_logstd_init=-0.7):
        super().__init__()
        assert obs_dim == 32, "RoleAwareActor 假设 obs_dim=32（Stage 2 观测）"
        self.attn_dim = attn_dim
        self.base_dim = 4 + 3 + 2 + 3 + 3 + 4  # self+role+depot+nearest+det_mask+goal = 19
        self.base = nn.Sequential(
            nn.Linear(self.base_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # 资源注意力（query=自身隐藏，key/value=资源实体 4 维）
        self.res_q = nn.Linear(hidden, attn_dim)
        self.res_k = nn.Linear(4, attn_dim)
        self.res_v = nn.Linear(4, attn_dim)
        # 队友注意力（key/value=队友实体 2 维）
        self.team_q = nn.Linear(hidden, attn_dim)
        self.team_k = nn.Linear(2, attn_dim)
        self.team_v = nn.Linear(2, attn_dim)
        # 输出头
        self.head = nn.Sequential(
            nn.Linear(hidden + 2 * attn_dim, hidden), nn.Tanh(),
        )
        self.drive_mean = nn.Linear(hidden, 2)
        self.drive_logstd = nn.Parameter(torch.full((2,), drive_logstd_init))
        self.tool_logits = nn.Linear(hidden, 3)

    def _attn(self, q, k, v):
        # q:(B,1,d) k,v:(B,n,d) → (B,d)
        score = (q * k).sum(-1) / (self.attn_dim ** 0.5)   # (B,n)
        w = torch.softmax(score, dim=-1)                    # (B,n)
        return (w.unsqueeze(-1) * v).sum(1)                 # (B,d)

    def forward(self, obs):
        B = obs.shape[0]
        self_feat = obs[:, 0:4]
        role = obs[:, 4:7]
        rel_depot = obs[:, 7:9]
        rel_nearest = obs[:, 9:12]
        rel_res = obs[:, 12:21].reshape(B, 3, 3)            # (B,3,3)
        det_mask = obs[:, 21:24].unsqueeze(-1)              # (B,3,1)
        team_ent = obs[:, 24:28].reshape(B, 2, 2)           # (B,2,2)
        rel_goal = obs[:, 28:32]

        res_ent = torch.cat([rel_res, det_mask], dim=-1)    # (B,3,4)

        base_in = torch.cat([self_feat, role, rel_depot, rel_nearest,
                             det_mask.squeeze(-1), rel_goal], dim=-1)
        h = self.base(base_in)                              # (B,hidden)

        attn_res = self._attn(self.res_q(h).unsqueeze(1),
                              self.res_k(res_ent), self.res_v(res_ent))
        attn_team = self._attn(self.team_q(h).unsqueeze(1),
                               self.team_k(team_ent), self.team_v(team_ent))

        out = self.head(torch.cat([h, attn_res, attn_team], dim=-1))
        drive_mean = self.drive_mean(out)
        drive_std = torch.exp(self.drive_logstd).clamp(min=1e-3)
        tool_logits = self.tool_logits(out)
        return drive_mean, drive_std, tool_logits


class VarRoleAwareActor(nn.Module):
    """变规模角色感知 Actor：对「可变数量的资源/depot/队友实体」做掩码注意力池化。

    输入（来自 `encode_obs_vars`）：
      fixed (B,13)           自身+角色+最近depot+子目标
      res_ent (B,max_res,5)  资源实体 [dx,dy,amount,value,detected]
      depot_ent (B,max_depot,2) depot 实体 [dx,dy]
      team_ent (B,max_team,2)  队友实体 [dx,dy]
      res_mask / depot_mask / team_mask  有效实体掩码
    注意力对实体数量不敏感（置换不变），是变规模 + 任务/目的地选择泛化的关键。
    """
    def __init__(self, fixed_dim=13, hidden=128, attn_dim=64, drive_logstd_init=-0.7):
        super().__init__()
        self.attn_dim = attn_dim
        self.base = nn.Sequential(
            nn.Linear(fixed_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.res_q = nn.Linear(hidden, attn_dim)
        self.res_k = nn.Linear(5, attn_dim)
        self.res_v = nn.Linear(5, attn_dim)
        self.depot_q = nn.Linear(hidden, attn_dim)
        self.depot_k = nn.Linear(2, attn_dim)
        self.depot_v = nn.Linear(2, attn_dim)
        self.team_q = nn.Linear(hidden, attn_dim)
        self.team_k = nn.Linear(2, attn_dim)
        self.team_v = nn.Linear(2, attn_dim)
        self.head = nn.Sequential(nn.Linear(hidden + 3 * attn_dim, hidden), nn.Tanh())
        self.drive_mean = nn.Linear(hidden, 2)
        self.drive_logstd = nn.Parameter(torch.full((2,), drive_logstd_init))
        self.tool_logits = nn.Linear(hidden, 3)

    def _masked_attn(self, q, k, v, mask):
        # q:(B,1,d) k/v:(B,n,d) mask:(B,n)
        score = (q * k).sum(-1) / (self.attn_dim ** 0.5)   # (B,n)
        score = score.masked_fill(mask < 0.5, -1e9)
        w = torch.softmax(score, dim=-1)                    # (B,n)
        return (w.unsqueeze(-1) * v).sum(1)                 # (B,d)

    def forward(self, fixed, res_ent, team_ent, res_mask, team_mask,
                depot_ent=None, depot_mask=None):
        h = self.base(fixed)
        attn_res = self._masked_attn(self.res_q(h).unsqueeze(1),
                                     self.res_k(res_ent), self.res_v(res_ent), res_mask)
        if depot_ent is not None and depot_mask is not None:
            attn_depot = self._masked_attn(self.depot_q(h).unsqueeze(1),
                                           self.depot_k(depot_ent), self.depot_v(depot_ent),
                                           depot_mask)
        else:
            attn_depot = torch.zeros(h.shape[0], self.attn_dim, device=h.device)
        attn_team = self._masked_attn(self.team_q(h).unsqueeze(1),
                                      self.team_k(team_ent), self.team_v(team_ent), team_mask)
        out = self.head(torch.cat([h, attn_res, attn_depot, attn_team], dim=-1))
        drive_mean = self.drive_mean(out)
        drive_std = torch.exp(self.drive_logstd).clamp(min=1e-3)
        tool_logits = self.tool_logits(out)
        return drive_mean, drive_std, tool_logits
