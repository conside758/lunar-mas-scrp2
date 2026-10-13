"""6 机器人 VarRoleAwareActor 的共享推理/编码工具（run_policy 与 Gazebo BC/DAgger 采集共用）。

保证「策略评估」与「DAgger 采集」使用完全一致的确定性推理与观测编码，避免两者发散。
编码格式与 `lunar_rl.train_vars.collect_var_data` 对齐（fixed/res_ent/depot_ent/team_ent/各 mask）。
"""
import numpy as np
import torch

from lunar_rl.networks import VarRoleAwareActor
from lunar_rl.obs import encode_obs_vars, FIXED_DIM
from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE, ROLE_TOOL_DIM

MAX_RES = 8
MAX_TEAM = 5
MAX_DEPOT = 2


def encode_item(obs, action):
    """把一步 (obs, action) 编码成 VarRoleAwareActor 训练样本（字段对齐 train_vars.collect_var_data）。"""
    enc = encode_obs_vars(obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
    return {
        "fixed": np.stack([enc[n]["fixed"] for n in ROBOT_NAMES]).astype(np.float32),
        "res_ent": np.stack([enc[n]["res_ent"] for n in ROBOT_NAMES]).astype(np.float32),
        "depot_ent": np.stack([enc[n]["depot_ent"] for n in ROBOT_NAMES]).astype(np.float32),
        "team_ent": np.stack([enc[n]["team_ent"] for n in ROBOT_NAMES]).astype(np.float32),
        "res_mask": np.stack([enc[n]["res_mask"] for n in ROBOT_NAMES]).astype(np.float32),
        "depot_mask": np.stack([enc[n]["depot_mask"] for n in ROBOT_NAMES]).astype(np.float32),
        "team_mask": np.stack([enc[n]["team_mask"] for n in ROBOT_NAMES]).astype(np.float32),
        "drive": np.array([action[n]["drive"] for n in ROBOT_NAMES], dtype=np.float32),
        "tools": [action[n].get("tool", 0) for n in ROBOT_NAMES],
    }


def load_var_actors(ckpt_path):
    """加载 {"actors": {name: VarRoleAwareActor.state_dict()}} checkpoint。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    actors = {}
    for name, sd in ckpt["actors"].items():
        a = VarRoleAwareActor(fixed_dim=FIXED_DIM, hidden=128)
        a.load_state_dict(sd)
        a.eval()
        actors[name] = a
    return actors


def det_action(actors, obs):
    """确定性动作 dict：逐机器人取 drive 均值 + tool argmax（role 专属维度）。"""
    enc = encode_obs_vars(obs, max_res=MAX_RES, max_team=MAX_TEAM, max_depot=MAX_DEPOT)
    action = {}
    with torch.no_grad():
        for name in ROBOT_NAMES:
            e = enc[name]
            fixed = torch.tensor(e["fixed"][None], dtype=torch.float32)
            res_ent = torch.tensor(e["res_ent"][None], dtype=torch.float32)
            depot_ent = torch.tensor(e["depot_ent"][None], dtype=torch.float32)
            team_ent = torch.tensor(e["team_ent"][None], dtype=torch.float32)
            res_mask = torch.tensor(e["res_mask"][None], dtype=torch.float32)
            depot_mask = torch.tensor(e["depot_mask"][None], dtype=torch.float32)
            team_mask = torch.tensor(e["team_mask"][None], dtype=torch.float32)
            dm, _, tl = actors[name](fixed, res_ent, team_ent, res_mask, team_mask,
                                     depot_ent, depot_mask)
            sub = {"drive": torch.clamp(dm[0], -1.0, 1.0).numpy().astype(np.float32)}
            k = ROLE_TOOL_DIM[ROBOT_ROLE[name]]
            if k > 0:
                sub["tool"] = int(torch.argmax(tl[0, :k]).item())
            action[name] = sub
    return action
