"""月面地形高度场：从地形 OBJ 采样构建 world 系高度图，供 surrogate 与 Gazebo 共享。

保证「替身观测」与「Gazebo 观测」里的地形特征（采样点高度/坡度 + roll/pitch 反馈）严格一致：
两者都用本模块的 `get_terrain()` 单例查询同一高度场。roll/pitch 的来源不同——
  - surrogate：由局部坡度按准静态几何计算（roll = 侧坡角、pitch = -前坡角）；
  - Gazebo：由 link_states 四元数解算（真实物理）；
但两者写入观测的编码格式与量纲完全一致。

坐标变换（与 `lunar_gazebo/models/lunar_terrain/model.sdf` 严格一致）：
    x_world = x_obj * S          (S=0.004，水平缩放)
    y_world = -z_obj * S
    z_world = y_obj * HS + Z_OFFSET    (HS=高度缩放，崎岖度开关；Z_OFFSET=0.052)
改 HS 时必须同步改 model.sdf 里 mesh 的 <scale> y 分量与 pose z 平移，否则替身与 Gazebo 地形不一致。
"""
import math
import os

import numpy as np

# ---- 与 model.sdf 保持一致的单点来源 ----
S = 0.004          # 水平缩放
HS = 0.0006        # 高度缩放（崎岖度）：0.0006=压平±0.08m（Scout 腿式车验证无翻倒，max roll/pitch≤6°）
Z_OFFSET = 0.052   # z 平移

_WORKSPACE = "/home/admina/MAS_ws"
_CACHE_DIR = os.path.join(_WORKSPACE, ".tmp")

# 车体周围地形采样点（body 系：(前方, 左侧)，单位 m）：前 3 + 左 2 + 右 2 = 7
TERRAIN_OFFSETS = [
    (0.5, 0.0), (1.0, 0.0), (1.5, 0.0),
    (0.5, 0.5), (0.5, 1.0),
    (0.5, -0.5), (0.5, -1.0),
]
N_TERRAIN_SAMPLES = len(TERRAIN_OFFSETS)
# 每个采样点 3 个特征：相对高度、前向坡度(tan)、侧向坡度(tan)
TERRAIN_PER_POINT = 3
# 追加到 fixed 观测的维度：N_TERRAIN_SAMPLES * 3 (采样) + 2 (roll, pitch)
TERRAIN_DIM = N_TERRAIN_SAMPLES * TERRAIN_PER_POINT + 2


def _obj_path():
    cands = [
        os.environ.get("LUNAR_TERRAIN_OBJ"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     "lunar_gazebo", "models", "lunar_terrain", "meshes", "terrain.obj"),
        os.path.join(_WORKSPACE, "src", "lunar_gazebo", "models",
                     "lunar_terrain", "meshes", "terrain.obj"),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("terrain.obj not found; set LUNAR_TERRAIN_OBJ")


def _cache_path(res):
    return os.path.join(_CACHE_DIR, f"terrain_hs{HS:.4f}_res{res:.2f}.npz")


class TerrainField:
    """world 系高度场：h(x,y) 与梯度 grad(x,y) 由规则网格 + 双线性插值给出。"""

    def __init__(self, res=0.2, bounds=8.0):
        self.res = float(res)
        self.bounds = float(bounds)
        self._load_or_build()

    def _load_or_build(self):
        path = _cache_path(self.res)
        if os.path.exists(path):
            z = np.load(path)
            self.hgrid = z["h"]
            self.gx = z["gx"]
            self.gy = z["gy"]
            self.n = int(z["n"])
            return
        self.hgrid, self.gx, self.gy, self.n = self._build()
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.savez(path, h=self.hgrid, gx=self.gx, gy=self.gy, n=self.n)

    def _build(self):
        verts = []
        with open(_obj_path(), "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split()
                    verts.append((float(p[1]), float(p[2]), float(p[3])))
        v = np.array(verts, dtype=np.float64)
        xw = v[:, 0] * S
        yw = -v[:, 2] * S
        zw = v[:, 1] * HS + Z_OFFSET

        b = self.bounds
        res = self.res
        n = int(2 * b / res) + 1
        ix = np.clip(((xw + b) / res).astype(np.int64), 0, n - 1)
        iy = np.clip(((yw + b) / res).astype(np.int64), 0, n - 1)
        acc = np.zeros((n, n), dtype=np.float64)
        cnt = np.zeros((n, n), dtype=np.float64)
        np.add.at(acc, (iy, ix), zw)
        np.add.at(cnt, (iy, ix), 1.0)
        mask = cnt > 0
        acc[mask] /= cnt[mask]

        # 填补空洞（网格边缘/无顶点单元）→ 最近有效单元
        if not mask.all():
            from scipy.spatial import cKDTree
            xs = np.linspace(-b, b, n)
            ys = np.linspace(-b, b, n)
            gxx, gyy = np.meshgrid(xs, ys, indexing="ij")
            filled = np.stack([gxx[mask], gyy[mask]], axis=-1)
            tree = cKDTree(filled)
            empty = ~mask
            _, idx = tree.query(np.stack([gxx[empty], gyy[empty]], axis=-1))
            acc[empty] = acc[mask][idx]

        gy_, gx_ = np.gradient(acc, res, res)
        return acc.astype(np.float32), gx_.astype(np.float32), gy_.astype(np.float32), n

    def _clip_idx(self, x):
        b = self.bounds
        return max(0.0, min(self.n - 1.001, (x + b) / self.res))

    def h(self, x, y):
        """世界点 (x, y) 的地面高度 z。"""
        i = self._clip_idx(x)
        j = self._clip_idx(y)
        i0, j0 = int(math.floor(i)), int(math.floor(j))
        i0 = max(0, min(self.n - 2, i0))
        j0 = max(0, min(self.n - 2, j0))
        fi, fj = i - i0, j - j0
        h00 = self.hgrid[i0, j0]
        h10 = self.hgrid[i0 + 1, j0]
        h01 = self.hgrid[i0, j0 + 1]
        h11 = self.hgrid[i0 + 1, j0 + 1]
        return (h00 * (1 - fi) * (1 - fj) + h10 * fi * (1 - fj)
                + h01 * (1 - fi) * fj + h11 * fi * fj)

    def grad(self, x, y):
        """世界点 (x, y) 的地面梯度 (dh/dx, dh/dy)（tan 坡度分量）。"""
        i = self._clip_idx(x)
        j = self._clip_idx(y)
        i0, j0 = int(math.floor(i)), int(math.floor(j))
        i0 = max(0, min(self.n - 2, i0))
        j0 = max(0, min(self.n - 2, j0))
        fi, fj = i - i0, j - j0

        def _bil(g):
            return (g[i0, j0] * (1 - fi) * (1 - fj) + g[i0 + 1, j0] * fi * (1 - fj)
                    + g[i0, j0 + 1] * (1 - fi) * fj + g[i0 + 1, j0 + 1] * fi * fj)
        return _bil(self.gx), _bil(self.gy)

    def slopes(self, x, y, yaw):
        """车体系前向/侧向坡度 (tan)：slope_fwd = 前进方向地面升高率；slope_lat = 左侧方向地面升高率。"""
        gx_, gy_ = self.grad(x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        fwd = gx_ * c + gy_ * s
        lat = -gx_ * s + gy_ * c
        return fwd, lat

    def roll_pitch(self, x, y, yaw):
        """准静态 roll/pitch（rad），符号约定与 Gazebo 一致：正 roll=左侧抬起；正 pitch=低头。"""
        fwd, lat = self.slopes(x, y, yaw)
        pitch = -math.atan(fwd)
        roll = math.atan(lat)
        return roll, pitch

    def features(self, x, y, yaw):
        """车体周围 K 个采样点的地形特征，返回 (N_TERRAIN_SAMPLES*3,) float32。

        布局：7 点 × [相对高度, 前向坡度(tan), 侧向坡度(tan)]。
        （roll/pitch 由 env 单独提供：surrogate 用本场准静态 roll_pitch()，Gazebo 用真实 link_states。）
        """
        h0 = self.h(x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        out = np.empty(N_TERRAIN_SAMPLES * TERRAIN_PER_POINT, dtype=np.float32)
        k = 0
        for (fwd, left) in TERRAIN_OFFSETS:
            px = x + fwd * c - left * s
            py = y + fwd * s + left * c
            gx_, gy_ = self.grad(px, py)
            out[k] = self.h(px, py) - h0                       # 相对高度
            out[k + 1] = gx_ * c + gy_ * s                     # 前向坡度(tan)
            out[k + 2] = -gx_ * s + gy_ * c                    # 侧向坡度(tan)
            k += TERRAIN_PER_POINT
        return out


_terrain = None


def get_terrain():
    """地形场单例（surrogate 与 Gazebo 共用，保证观测一致）。"""
    global _terrain
    if _terrain is None:
        _terrain = TerrainField()
    return _terrain


def terrain_feature_dim():
    return TERRAIN_DIM
