"""训练运行日志 + 可视化：把每次训练的曲线图 / 日志 / checkpoint 存到 `results/<当天日期>/`。

约定：
  - results/<YYYY-MM-DD>/ 按训练当天日期自动创建；
  - 同一天多次运行用 run_name 区分（作为文件前缀）；
  - 曲线图用 matplotlib（Agg 后端，headless 安全）画 delivered / reward / value_loss / entropy vs iter。
"""
import datetime
import os

# headless（无显示器）安全 + 可写的 matplotlib 配置目录
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_lunar")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# workspace 根目录：本文件位于 <ws>/src/lunar_rl/lunar_rl/runlog.py，向上三级即 workspace
_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_ROOT = os.path.join(_WS_ROOT, "results")


def _setup_cjk_font():
    """优先用系统中文字体，避免中文标签显示为方块。"""
    for name in ["Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK JP",
                 "Noto Serif CJK SC", "AR PL UMing CN", "WenQuanYi Zen Hei"]:
        try:
            fm.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


_setup_cjk_font()


def run_dir(run_name=None):
    """返回 (日期目录, 文件前缀)。目录 `results/<today>/` 不存在则创建。"""
    today = datetime.date.today().isoformat()
    d = os.path.join(RESULTS_ROOT, today)
    os.makedirs(d, exist_ok=True)
    prefix = run_name if run_name else "run"
    return d, prefix


def plot_curves(metrics, out_png, title="training curves"):
    """把训练曲线画成 2×2 子图并存 PNG。

    metrics: dict，键 delivered / reward / value_loss / entropy，值为 [(iter, val), ...] 列表。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("delivered", "delivered (交付量)"),
        ("reward", "episode reward (总回报)"),
        ("value_loss", "value loss (Q 噪声)"),
        ("entropy", "entropy (探索度)"),
    ]
    for ax, (key, label) in zip(axes.ravel(), panels):
        pts = metrics.get(key, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=".", markersize=2)
        ax.set_title(label)
        ax.set_xlabel("iter")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def plot_bars(labels, values, out_png, title="comparison", ylabel="delivered"):
    """简单柱状图（用于对比类结果，如 RoleAware vs Flat 的交付量）。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def write_summary(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
