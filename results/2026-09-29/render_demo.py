"""把 record_episode.py 录制的轨迹渲染成 2D 可视化 demo（GIF 动画 + 静态摘要）。

用法：python3 render_demo.py <traj.pkl> <out_dir>
输出：<out_dir>/mining_demo.gif、<out_dir>/mining_summary.png
"""
import os
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_lunar")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 角色 → 标记/颜色
ROLE_STYLE = {
    "scout":     dict(marker="^", color="#4c72b0", label="Scout 探测"),
    "excavator": dict(marker="o", color="#dd8452", label="Excavator 挖掘"),
    "hauler":    dict(marker="s", color="#55a868", label="Hauler 运输"),
}
VALUE_COLOR = {3.0: "#d62728", 2.0: "#ff7f0e", 1.0: "#2ca02c"}


def main():
    pkl = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    with open(pkl, "rb") as f:
        blob = pickle.load(f)
    roles = blob["roles"]
    traj = blob["traj"]
    print(f"loaded {len(traj)} frames")

    # 资源/卸载点静态几何（用首帧）
    res0 = np.array(traj[0]["resources"])          # (n,4) x,y,amount,value
    depots = np.array(traj[0]["depots"])           # (k,2)
    names = list(roles.keys())

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # 静态元素
    depot_art = [ax.scatter(d[0], d[1], marker="s", s=220, color="#f0c419",
                            edgecolors="black", zorder=3) for d in depots]
    ax.scatter([], [], marker="s", color="#f0c419", label="Depot 卸载点")

    res_sc = ax.scatter(res0[:, 0], res0[:, 1], s=180, c=[VALUE_COLOR[v] for v in res0[:, 3]],
                        edgecolors="black", zorder=2, label="资源")
    # 图例（价值档）
    for v, c in VALUE_COLOR.items():
        ax.scatter([], [], s=80, color=c, label=f"资源 value={int(v)}")

    robot_art = {}
    trail_art = {}
    for n in names:
        st = ROLE_STYLE[roles[n]]
        trail_art[n], = ax.plot([], [], color=st["color"], lw=1.0, alpha=0.4, zorder=1)
        robot_art[n] = ax.scatter([], [], marker=st["marker"], s=260, color=st["color"],
                                  edgecolors="white", linewidths=1.2, zorder=4,
                                  label=st["label"])

    title = ax.set_title("", fontsize=12)

    def update(i):
        f = traj[i]
        # 资源当前量 → 大小随剩余量缩
        rs = np.array(f["resources"])
        alive = rs[:, 2] > 0.01
        if alive.any():
            sizes = 40 + 160 * (rs[:, 2] / np.maximum(res0[:, 2], 1e-6))
            res_sc.set_offsets(rs[:, :2])
            res_sc.set_sizes(sizes)
            res_sc.set_alpha(np.where(alive, 1.0, 0.15))
        # 机器人 + 轨迹
        xs = [f["robots"][n][0] for n in names]
        ys = [f["robots"][n][1] for n in names]
        for n in names:
            st = ROLE_STYLE[roles[n]]
            robot_art[n].set_offsets([[f["robots"][n][0], f["robots"][n][1]]])
            lo = max(0, i - 25)
            trail_art[n].set_data(
                [traj[k]["robots"][n][0] for k in range(lo, i + 1)],
                [traj[k]["robots"][n][1] for k in range(lo, i + 1)],
            )
        title.set_text(
            f"step={f['step']}/240   delivered_value={f['delivered_value']:.1f} "
            f"(delivered {f['delivered']:.1f} t)   挖掘: "
            f"{list(f['buckets'].values())}  车斗: {list(f['bins'].values())}"
        )
        return [res_sc, title] + list(robot_art.values()) + list(trail_art.values())

    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()

    # 动画 GIF（每 2 步一帧，10fps）
    frames = list(range(0, len(traj), 2))
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    gif_path = os.path.join(out_dir, "mining_demo.gif")
    anim.save(gif_path, writer="pillow", fps=10, dpi=90)
    print(f"saved {gif_path}")

    # 静态摘要：交付曲线 + 末帧轨迹
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    steps = [f["step"] for f in traj]
    dv = [f["delivered_value"] for f in traj]
    a1.plot(steps, dv, color="#c44e52", lw=2)
    a1.set_xlabel("step"); a1.set_ylabel("delivered_value (/233)")
    a1.set_title("交付价值随回合推进"); a1.grid(True, alpha=0.3)

    a2.set_xlim(-4.2, 4.2); a2.set_ylim(-4.2, 4.2); a2.set_aspect("equal")
    a2.grid(True, alpha=0.15)
    a2.scatter(res0[:, 0], res0[:, 1], s=120, c=[VALUE_COLOR[v] for v in res0[:, 3]],
               edgecolors="black", alpha=0.5)
    for d in depots:
        a2.scatter(d[0], d[1], marker="s", s=260, color="#f0c419", edgecolors="black")
    for n in names:
        st = ROLE_STYLE[roles[n]]
        px = [f["robots"][n][0] for f in traj]
        py = [f["robots"][n][1] for f in traj]
        a2.plot(px, py, color=st["color"], lw=1.0, alpha=0.5)
        a2.scatter(px[-1], py[-1], marker=st["marker"], s=180, color=st["color"],
                   edgecolors="white", zorder=4, label=st["label"])
    a2.legend(loc="upper right", fontsize=8)
    a2.set_title("机器人运动轨迹（末位置为标记）")
    fig2.tight_layout()
    summary_path = os.path.join(out_dir, "mining_summary.png")
    fig2.savefig(summary_path, dpi=110)
    plt.close(fig2)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
