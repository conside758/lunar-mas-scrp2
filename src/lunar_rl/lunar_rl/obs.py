"""观测编码：把 surrogate/lunar_env 的原始 obs 编码为每车定长向量。

相对编码 + 角色 one-hot + 显式子目标。资源/队友数量固定（n=3）时直接展平拼接。
Stage 2（资源位置未知）：观测只暴露「已发现」资源（未发现资源置零），并附「发现掩码」。

每车向量（n_resources=3）：
  自身[x,y,yaw,cargo] (4)   ← 不含 vx/wz（替身指令速度与 Gazebo 实际速度不一致，去掉以免 OOD）
  + 角色 one-hot (3)
  + 相对 depot [dx,dy] (2)
  + 相对最近「已发现」资源 [dx,dy,amount] (3)
  + 相对所有资源展平（未发现置零）(3*3=9)
  + 资源发现掩码 [d0,d1,d2] (3)
  + 相对所有队友展平 (2*2=4)
  + 相对当前子目标（世界系）[dx,dy] (2)
  + 相对当前子目标（车体系）[dx,dy] (2)
  = 32 维
"""
import math

import numpy as np

from lunar_rl.registry import ROBOT_NAMES, ROBOT_ROLE, ROLES


def encode_obs(raw_obs):
    depot = raw_obs["depot"]
    resources = np.asarray(raw_obs["resources"], dtype=np.float32)  # (n, 3)
    goals = raw_obs.get("goals", {})

    det = raw_obs.get("detected", None)
    if det is None:
        det_mask = np.ones(resources.shape[0], dtype=np.float32)  # Stage 1：全部已知
    else:
        det_mask = np.asarray(det, dtype=np.float32)

    encoded = {}
    for name in ROBOT_NAMES:
        o = raw_obs[name]
        x, y, yaw = float(o[0]), float(o[1]), float(o[2])
        cargo = float(o[5]) if len(o) >= 6 else 0.0

        self_feat = np.array([x, y, yaw, cargo], dtype=np.float32)

        role = np.zeros(3, dtype=np.float32)
        role[ROLES[ROBOT_ROLE.get(name, name)]] = 1.0

        rel_depot = np.array([depot[0] - x, depot[1] - y], dtype=np.float32)

        # 相对最近「已发现且有量」资源（关键：告诉策略去哪挖）
        avail_mask = (resources[:, 2] > 0.01) & (det_mask > 0.5)
        avail = resources[avail_mask]
        if len(avail) > 0:
            nearest = avail[np.argmin((avail[:, 0] - x) ** 2 + (avail[:, 1] - y) ** 2)]
            rel_nearest = np.array([nearest[0] - x, nearest[1] - y, nearest[2]],
                                   dtype=np.float32)
        else:
            rel_nearest = np.zeros(3, dtype=np.float32)

        # 相对所有资源展平（未发现资源置零，位置与量都不可见）
        rel_res = np.stack([
            (resources[:, 0] - x) * det_mask,
            (resources[:, 1] - y) * det_mask,
            resources[:, 2] * det_mask,
        ], axis=1).astype(np.float32).reshape(-1)  # (3*n,)

        # 相对所有队友展平
        team_rel = []
        for tn in ROBOT_NAMES:
            if tn == name:
                continue
            to = raw_obs[tn]
            team_rel += [float(to[0]) - x, float(to[1]) - y]
        team_rel = np.array(team_rel, dtype=np.float32)  # (4,)

        # 相对当前子目标（显式引导「去目标」这一低层技能）
        g = goals.get(name)
        if g is None:
            rel_goal = np.zeros(2, dtype=np.float32)
        else:
            rel_goal = np.array([float(g[0]) - x, float(g[1]) - y],
                                dtype=np.float32)

        # 子目标在车体系下的坐标：免去网络做「世界系目标角度 − 车航向」的
        # 带环绕角度减法，直接给出去目标的航向误差，便于回归 go-to-goal。
        c, s = math.cos(yaw), math.sin(yaw)
        rel_goal_body = np.array([
            rel_goal[0] * c + rel_goal[1] * s,
            -rel_goal[0] * s + rel_goal[1] * c,
        ], dtype=np.float32)

        encoded[name] = np.concatenate(
            [self_feat, role, rel_depot, rel_nearest, rel_res, det_mask,
             team_rel, rel_goal, rel_goal_body])

    return encoded


def encode_global(encoded):
    """集中 critic 的全局输入：拼接三车编码向量。"""
    return np.concatenate([encoded[n] for n in ROBOT_NAMES])


def obs_dim():
    """每车观测向量维度（n_resources=3 时）。"""
    return 4 + 3 + 2 + 3 + 9 + 3 + 4 + 2 + 2


# ---------- 变规模观测（资源/队友/depot 实体 + 掩码，注意力池化用） ----------
FIXED_DIM = 13  # self(4) + role(3) + rel_nearest_depot(2) + rel_goal_world(2) + rel_goal_body(2)


def encode_obs_vars(raw_obs, max_res, max_team=2, max_depot=2):
    """变规模观测：把每车编码成「固定特征 + 资源实体 + depot 实体 + 队友实体 + 掩码」。

    固定特征 fixed（13）：[x,y,yaw,cargo] + 角色 one-hot(3) + 相对最近 depot(2) + 相对子目标世界(2) + 车体(2)。
    资源实体 res_ent（max_res×5）：[dx,dy,amount,value,detected]；未发现资源位置/量/价值置零。
    depot 实体 depot_ent（max_depot×2）：[dx,dy]（目的地选择用）。
    队友实体 team_ent（max_team×2）：[dx,dy]。
    """
    resources = np.asarray(raw_obs["resources"], dtype=np.float32)  # (n,4) = [x,y,amount,value]
    depots = np.asarray(raw_obs["depots"], dtype=np.float32) if "depots" in raw_obs \
        else np.asarray([raw_obs["depot"]], dtype=np.float32)
    goals = raw_obs.get("goals", {})
    det = raw_obs.get("detected", None)
    if det is None:
        det_mask = np.ones(len(resources), dtype=np.float32)
    else:
        det_mask = np.asarray(det, dtype=np.float32)

    n_res = min(len(resources), max_res)
    n_dep = min(len(depots), max_depot)
    out = {}
    for name in ROBOT_NAMES:
        o = raw_obs[name]
        x, y, yaw = float(o[0]), float(o[1]), float(o[2])
        cargo = float(o[5]) if len(o) >= 6 else 0.0
        role = np.zeros(3, dtype=np.float32)
        role[ROLES[ROBOT_ROLE.get(name, name)]] = 1.0
        # 最近 depot（固定特征里的便捷项）
        if len(depots) > 0:
            near_dep = depots[np.argmin((depots[:, 0] - x) ** 2 + (depots[:, 1] - y) ** 2)]
            rel_depot = np.array([near_dep[0] - x, near_dep[1] - y], dtype=np.float32)
        else:
            rel_depot = np.zeros(2, dtype=np.float32)
        g = goals.get(name)
        if g is None:
            g = [0.0, 0.0]
        rel_goal = np.array([float(g[0]) - x, float(g[1]) - y], dtype=np.float32)
        c, s = math.cos(yaw), math.sin(yaw)
        rel_goal_body = np.array([rel_goal[0] * c + rel_goal[1] * s,
                                  -rel_goal[0] * s + rel_goal[1] * c], dtype=np.float32)
        fixed = np.concatenate([[x, y, yaw, cargo], role, rel_depot,
                                rel_goal, rel_goal_body]).astype(np.float32)  # (13,)

        res_ent = np.zeros((max_res, 5), dtype=np.float32)
        res_mask = np.zeros(max_res, dtype=np.float32)
        for i in range(n_res):
            dx = resources[i, 0] - x
            dy = resources[i, 1] - y
            amt = resources[i, 2]
            val = resources[i, 3] if resources.shape[1] >= 4 else 1.0
            d = det_mask[i]
            res_ent[i] = [dx * d, dy * d, amt * d, val * d, d]  # 未发现→[0,0,0,0,0]
            res_mask[i] = 1.0

        depot_ent = np.zeros((max_depot, 2), dtype=np.float32)
        depot_mask = np.zeros(max_depot, dtype=np.float32)
        for i in range(n_dep):
            depot_ent[i] = [depots[i, 0] - x, depots[i, 1] - y]
            depot_mask[i] = 1.0

        team_ent = np.zeros((max_team, 2), dtype=np.float32)
        team_mask = np.zeros(max_team, dtype=np.float32)
        ti = 0
        for tn in ROBOT_NAMES:
            if tn == name:
                continue
            to = raw_obs[tn]
            team_ent[ti] = [float(to[0]) - x, float(to[1]) - y]
            team_mask[ti] = 1.0
            ti += 1

        out[name] = {"fixed": fixed, "res_ent": res_ent, "res_mask": res_mask,
                     "depot_ent": depot_ent, "depot_mask": depot_mask,
                     "team_ent": team_ent, "team_mask": team_mask}
    return out


def flat_vars(enc):
    """把变规模编码展平成一个定长向量（普通 MLP Actor 用）。"""
    return np.concatenate([enc["fixed"], enc["res_ent"].reshape(-1),
                           enc["depot_ent"].reshape(-1), enc["team_ent"].reshape(-1),
                           enc["res_mask"], enc["depot_mask"], enc["team_mask"]])


def flat_vars_dim(max_res, max_team=2, max_depot=2):
    return FIXED_DIM + 5 * max_res + 2 * max_depot + 2 * max_team + max_res + max_depot + max_team
