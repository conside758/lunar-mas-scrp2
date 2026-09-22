"""机器人注册表：机器人名列表、角色映射、角色工具维度（可配置，支持同角色多实例）。

默认 3 台（scout/excavator/hauler 各 1），保持旧场景兼容；6 台场景通过 `set_robots()` 原地更新，
使 `from lunar_rl.registry import ROBOT_NAMES` 这类引用依然指向同一 list/dict 对象而生效。
"""
# 机器人名（顺序即联合动作/观测里的顺序）
ROBOT_NAMES = ["scout", "excavator", "hauler"]
# 机器人名 → 角色（同角色多实例：如 scout_0/scout_1 都 → "scout"）
ROBOT_ROLE = {"scout": "scout", "excavator": "excavator", "hauler": "hauler"}
# 角色 → one-hot 索引（角色类型固定 3 类）
ROLES = {"scout": 0, "excavator": 1, "hauler": 2}
# 角色 → 工具类别数（scout 无工具 / excavator 3 类 / hauler 2 类）
ROLE_TOOL_DIM = {"scout": 0, "excavator": 3, "hauler": 2}
# 角色顺序（联合动作编码 / 奖励里的稳定顺序）
ROLE_ORDER = ["scout", "excavator", "hauler"]


def role_counts():
    """每个角色有几台机器人（用于联合动作编码维度）。"""
    c = {r: 0 for r in ROLE_ORDER}
    for n in ROBOT_NAMES:
        c[ROBOT_ROLE[n]] += 1
    return c


def robots_of_role(role):
    return [n for n in ROBOT_NAMES if ROBOT_ROLE[n] == role]


def set_robots(names, roles):
    """原地更新机器人注册表（不改 list/dict 对象引用）。"""
    ROBOT_NAMES[:] = list(names)
    ROBOT_ROLE.clear()
    ROBOT_ROLE.update(roles)


def joint_action_dim():
    """联合动作编码维度 = 所有机器人 drive(2) + 各角色 tool one-hot 之和。"""
    d = 2 * len(ROBOT_NAMES)
    for r in ROLE_ORDER:
        d += role_counts()[r] * ROLE_TOOL_DIM[r]
    return d


def tool_offsets():
    """每个机器人的 tool one-hot 在联合动作编码里的起始下标（drive 之后按 ROBOT_NAMES 顺序排）。"""
    offs = {}
    off = 2 * len(ROBOT_NAMES)
    for name in ROBOT_NAMES:
        offs[name] = off
        off += ROLE_TOOL_DIM[ROBOT_ROLE[name]]
    return offs
