"""目的性指标评估：在 surrogate 上跑规则专家，对比「旧贪婪」与「新承诺式」的行为目的性。

指标（逐车逐回合，按角色分组）：
  - goal_switches       ：目标身份键（goal_key_for）切换总次数；
  - same_phase_retargets：同阶段内换目标身份的次数（即「每步重拍」抖动，承诺式应趋近 0）；
  - detour_ratio        ：静态目标（资源/depot/探测点）直线行程的迂回比 = 实际路径长 / 起点到目标直线距离；
  - reversals           ：方向反转次数（相邻「运动中」位移方向夹角 >90°，即掉头折返）；
  - delivered_value     ：交付价值（不应退化）。

用法：python3 -m lunar_rl.eval_purpose <scenario.yaml> [--episodes 10] [--seed0 0]
"""
import argparse
import json
import math

import numpy as np

from lunar_rl.registry import ROBOT_ROLE
from lunar_rl.rewards import goal_for, goal_key_for
from lunar_rl.rule_policy import rule_action
from lunar_rl.surrogate import SurrogateEnv

EPS_MOVE = 0.02   # 位移阈值（m）：低于视为原地转向/静止
REV_ANGLE = math.radians(90.0)


def _norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _phase_of(role, key):
    if role == "scout":
        return "idle" if key == "idle" else "explore"
    if role == "excavator":
        if key.startswith("res"):
            return "dig"
        if key.startswith("load:"):
            return "load"
        return "idle"
    if role == "hauler":
        if key.startswith("depot"):
            return "dump"
        if key.startswith("serve:"):
            return "serve"
        return "idle"
    return "idle"


def _run_episode(env, seed):
    np.random.seed(seed)
    obs, info = env.reset()
    names = env.robot_names
    poses = {n: [env.pose[n]] for n in names}
    keys = {n: [] for n in names}
    goals = {n: [] for n in names}
    for n in names:
        keys[n].append(goal_key_for(env, n))
        goals[n].append(goal_for(env, n))
    while True:
        action = rule_action(obs)
        obs, _, term, trunc, info = env.step(action)
        for n in names:
            poses[n].append(env.pose[n])
            keys[n].append(goal_key_for(env, n))
            goals[n].append(goal_for(env, n))
        if term or trunc:
            break
    return poses, keys, goals, info


def _reversals(disp, phases):
    """方向反转：相邻「运动中」位移方向夹角 >90°（掉头折返）。

    返回 (total, same_phase)：same_phase 为「同阶段内掉头」（每步重拍导致的来回踱步）。
    disp 长度 T-1；phases 长度 T（含初始步）。
    """
    total = 0
    same = 0
    last = None
    last_phase = None
    for t, d in enumerate(disp):
        m = math.hypot(d[0], d[1])
        if m <= EPS_MOVE:
            continue
        phase = phases[t + 1]  # 位移 disp[t] 对应从步 t 到 t+1
        if last is not None:
            dot = d[0] * last[0] + d[1] * last[1]
            if dot < 0:  # >90°
                total += 1
                if phase == last_phase:
                    same += 1
        last = d
        last_phase = phase
    return total, same


def _metrics(poses, keys, goals, role):
    ps = np.asarray(poses)  # (T,3)
    disp = np.diff(ps[:, :2], axis=0)
    path_len = float(np.sum(np.hypot(disp[:, 0], disp[:, 1])))
    straight = float(math.hypot(ps[-1, 0] - ps[0, 0], ps[-1, 1] - ps[0, 1]))
    detour_ep = (path_len / straight) if straight > 1e-3 else 1.0

    # 目标切换（总次数 + 同阶段换身份次数）
    total_switches = 0
    same_phase = 0
    for t in range(1, len(keys)):
        if keys[t] != keys[t - 1]:
            total_switches += 1
            if _phase_of(role, keys[t]) == _phase_of(role, keys[t - 1]):
                same_phase += 1

    # 迂回比：按「同阶段」连续段统计（同一 phase 内的实际路径长 / 净位移），
    # 只统计有真实位移的段（净位移 > 0.3m），跳过原地作业（挖/装/卸）段。
    seg_ratios = []
    T = len(keys)
    i = 0
    while i < T:
        j = i
        while j < T and _phase_of(role, keys[j]) == _phase_of(role, keys[i]):
            j += 1
        # 阶段段 [i, j)
        if j - i >= 2:
            seg_path = 0.0
            for t in range(i, j - 1):
                seg_path += math.hypot(ps[t + 1, 0] - ps[t, 0], ps[t + 1, 1] - ps[t, 1])
            net = math.hypot(ps[j - 1, 0] - ps[i, 0], ps[j - 1, 1] - ps[i, 1])
            if net > 0.3 and seg_path > 1e-3:
                seg_ratios.append(seg_path / net)
        i = j

    detour_seg = float(np.mean(seg_ratios)) if seg_ratios else None
    phases = [_phase_of(role, k) for k in keys]
    rev_total, rev_same = _reversals(disp, phases)
    return {
        "goal_switches": total_switches,
        "same_phase_retargets": same_phase,
        "detour_ratio_ep": detour_ep,
        "detour_ratio_seg": detour_seg,
        "reversals": rev_total,
        "same_phase_reversals": rev_same,
        "n_seg": len(seg_ratios),
    }


def run(env, episodes, seed0):
    per_ep = []
    for e in range(episodes):
        poses, keys, goals, info = _run_episode(env, seed0 + e)
        m = {n: _metrics(poses[n], keys[n], goals[n], env.role_of[n])
             for n in env.robot_names}
        per_ep.append({"metrics": m, "delivered_value": float(info["delivered_value"]),
                       "max_roll_deg": float(info.get("max_roll_deg", 0.0)),
                       "max_pitch_deg": float(info.get("max_pitch_deg", 0.0))})
    return per_ep


def _mean(per_ep, key, role_of=None, role_filter=None):
    vals = [e["metrics"][n][key] for e in per_ep for n in e["metrics"]
            if (role_filter is None or role_of[n] in role_filter)
            and e["metrics"][n][key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _mean_detour(per_ep, role_of=None, role_filter=None):
    vals = [e["metrics"][n]["detour_ratio_seg"] for e in per_ep for n in e["metrics"]
            if (role_filter is None or role_of[n] in role_filter)
            and e["metrics"][n]["detour_ratio_seg"] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _frac_ep_detour_le(per_ep, thresh=1.5):
    """「多数回合」的迂回比：每个回合内所有车静态段迂回比的均值 ≤ 阈值的回合占比。"""
    cnt = 0
    for e in per_ep:
        vals = [e["metrics"][n]["detour_ratio_seg"] for n in e["metrics"]
                if e["metrics"][n]["detour_ratio_seg"] is not None]
        if vals and float(np.mean(vals)) <= thresh:
            cnt += 1
    return cnt / max(1, len(per_ep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    results = {}
    for committed, label in [(False, "old_greedy"), (True, "new_committed")]:
        env = SurrogateEnv(args.scenario, committed=committed)
        per_ep = run(env, args.episodes, args.seed0)
        role_of = env.role_of
        task_roles = ("excavator", "hauler")
        results[label] = {
            "delivered_value_mean": float(np.mean([e["delivered_value"] for e in per_ep])),
            "goal_switches_all": _mean(per_ep, "goal_switches", role_of),
            "goal_switches_task": _mean(per_ep, "goal_switches", role_of, task_roles),
            "same_phase_retargets_task": _mean(per_ep, "same_phase_retargets", role_of,
                                               task_roles),
            "detour_ratio_seg": _mean_detour(per_ep, role_of),
            "detour_ratio_seg_task": _mean_detour(per_ep, role_of, task_roles),
            "frac_ep_detour_le_1.5": _frac_ep_detour_le(per_ep, 1.5),
            "reversals_task": _mean(per_ep, "reversals", role_of, task_roles),
            "same_phase_reversals_task": _mean(per_ep, "same_phase_reversals", role_of,
                                               task_roles),
            "max_roll_deg": float(np.mean([e["max_roll_deg"] for e in per_ep])),
            "max_pitch_deg": float(np.mean([e["max_pitch_deg"] for e in per_ep])),
            "per_episode": [{"delivered_value": e["delivered_value"]} for e in per_ep],
        }
        r = results[label]
        print(f"\n===== {label} (committed={committed}) =====")
        print(f"  delivered_value           = {r['delivered_value_mean']:.2f}")
        print(f"  goal_switches (all)       = {r['goal_switches_all']:.2f}")
        print(f"  goal_switches (excav+haul)= {r['goal_switches_task']:.2f}")
        print(f"  same_phase_retargets(task)= {r['same_phase_retargets_task']:.2f}")
        print(f"  detour_ratio_seg (all)    = {r['detour_ratio_seg']:.3f}")
        print(f"  detour_ratio_seg (task)   = {r['detour_ratio_seg_task']:.3f}")
        print(f"  frac episodes detour<=1.5 = {r['frac_ep_detour_le_1.5']:.2f}")
        print(f"  reversals (excav+haul)    = {r['reversals_task']:.2f}")
        print(f"  same_phase_reversals(task)= {r['same_phase_reversals_task']:.2f}")
        print(f"  max_roll_deg / max_pitch  = {r['max_roll_deg']:.2f} / {r['max_pitch_deg']:.2f}")

    old, new = results["old_greedy"], results["new_committed"]
    def drop(a, b):
        return 100 * (1 - b / max(1e-9, a))
    print("\n===== 对比（新/旧）=====")
    print(f"  goal_switches(task) 下降      = {drop(old['goal_switches_task'], new['goal_switches_task']):.1f}%")
    print(f"  same_phase_retargets(task) 下降 = {drop(old['same_phase_retargets_task'], new['same_phase_retargets_task']):.1f}%")
    print(f"  detour_ratio_seg(task) 旧={old['detour_ratio_seg_task']:.3f} 新={new['detour_ratio_seg_task']:.3f}")
    print(f"  reversals(task) 旧={old['reversals_task']:.2f} 新={new['reversals_task']:.2f}")
    print(f"  delivered_value 旧={old['delivered_value_mean']:.2f} 新={new['delivered_value_mean']:.2f}")
    print("  注：reversals 以「合法阶段切换（挖↔装、装↔卸）的掉头」为主，新旧相近；")
    print("     真正的「每步重拍」抖动体现在 same_phase_retargets（目标同阶段换身份）。")

    out = "/home/admina/MAS_ws/.tmp/purpose_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
