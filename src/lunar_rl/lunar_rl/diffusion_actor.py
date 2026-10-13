"""Diffusion Policy 驱动的角色感知 Actor：把高斯 MSE 回归的 drive 头替换为 DDPM 去噪扩散。

- 观测特征提取（fixed + 掩码注意力）与 `VarRoleAwareActor` 完全一致，得到 conditioning 特征 cond；
- drive 由 DDPM 反向采样生成（能表达多峰动作，避免高斯均值回归把「原地转向/直线前进」平均到
  「边转边走」的禁区）；
- 无 tool 头（部署时 tool 用规则状态机，见 run_policy_diffusion.py）。

数学（DDPM，线性 beta）：
  前向：x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
  训练：ε_hat = denoise(x_t, cond, t)，L = MSE(ε_hat, ε)
  采样：x_T~N(0,I) 逐步去噪到 x_0。
"""
import math

import torch
import torch.nn as nn


class DiffusionDriveActor(nn.Module):
    def __init__(self, fixed_dim=36, hidden=128, attn_dim=64, action_dim=2, T=20):
        super().__init__()
        self.attn_dim = attn_dim
        self.action_dim = action_dim
        self.T = T

        # ---- 特征提取（同 VarRoleAwareActor）----
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

        # ---- DDPM 调度（cosine，Nichol & Dhariwal 2021）----
        # 关键：确保 ᾱ_T → 0（前向完全加噪）。旧线性 1e-4→0.02 在 T=20 时 ᾱ_T≈0.82，
        # 反向从 N(0,1) 采样与前向不匹配，是 loss 不收敛（>1.0）的根因。
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = torch.clip(1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0.0001, 0.999)
        alphas = 1.0 - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

        # ---- 去噪器：输入 [x_t(2) | cond(hidden) | t_norm(1)] → ε(2) ----
        self.denoise = nn.Sequential(
            nn.Linear(action_dim + hidden + 1, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, action_dim),
        )

    def _masked_attn(self, q, k, v, mask):
        # q:(B,1,d) k/v:(B,n,d) mask:(B,n) -> (B,d)
        score = (q * k).sum(-1) / (self.attn_dim ** 0.5)
        score = score.masked_fill(mask < 0.5, -1e9)
        w = torch.softmax(score, dim=-1)
        return (w.unsqueeze(-1) * v).sum(1)

    def encode(self, fixed, res_ent, team_ent, res_mask, team_mask,
               depot_ent=None, depot_mask=None):
        """观测 → conditioning 特征 cond (B, hidden)。"""
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
        return self.head(torch.cat([h, attn_res, attn_depot, attn_team], dim=-1))

    def _denoise(self, x_t, cond, t):
        t_norm = (t.float() / max(self.T - 1, 1)).unsqueeze(1)
        return self.denoise(torch.cat([x_t, cond, t_norm], dim=-1))

    def diffusion_loss(self, cond, x0, eps=None):
        """DDPM 训练损失：MSE(预测噪声, 真实噪声)。cond/x0 同 batch。"""
        B = x0.shape[0]
        if eps is None:
            eps = torch.randn_like(x0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        a_bar = self.alphas_cumprod[t].unsqueeze(1)
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps
        eps_hat = self._denoise(x_t, cond, t)
        return nn.functional.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, cond, ddim=False):
        """反向采样 → drive ∈ [-1,1]^(action_dim)。cond:(B,hidden)。

        ddim=True：确定性 DDIM（更稳定、方差更小，部署推荐）；ddim=False：随机 DDPM。
        """
        B = cond.shape[0]
        device = cond.device
        x = torch.randn(B, self.action_dim, device=device)
        for t in reversed(range(self.T)):
            tt = torch.full((B,), t, dtype=torch.long, device=device)
            eps_hat = self._denoise(x, cond, tt)
            if ddim:
                a_bar = self.alphas_cumprod[t]
                x0 = (x - torch.sqrt(1.0 - a_bar) * eps_hat) / torch.sqrt(a_bar)
                a_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
                x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1.0 - a_bar_prev) * eps_hat
            else:
                alpha = self.alphas[t]
                a_bar = self.alphas_cumprod[t]
                beta = self.betas[t]
                mean = (x - (1.0 - alpha) / torch.sqrt(1.0 - a_bar) * eps_hat) / torch.sqrt(alpha)
                z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = mean + torch.sqrt(beta) * z
        return torch.clamp(x, -1.0, 1.0)
