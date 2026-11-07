"""固定地图势函数：保守膨胀 + 反向 Dijkstra 距离场。

Phi(s) = -min(D(p), Dmax) / Dmax，D 为到出口内部目标格点的自由空间最短距离。
距离场只用于势函数与粗可达性检查，真实碰撞仍由旋转矩形 SAT 判定。
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math

import numpy as np

from .geometry import circle_intersects_rect

_CACHE: dict[str, "DistanceField"] = {}


class DistanceField:
    def __init__(self, env_spec: dict):
        self.spec = dict(env_spec["potential"])
        self.bounds = tuple(float(v) for v in env_spec["map_bounds"])
        self.obstacles = [tuple(float(v) for v in row) for row in env_spec["obstacles"]]
        self.length = float(env_spec["length"])
        self.width = float(env_spec["width"])
        self.res = float(self.spec["grid_resolution"])
        self.inflation = float(self.spec["inflation_radius"])
        self.dmax = float(self.spec["dmax"])
        self.goal = tuple(float(v) for v in self.spec["goal"])
        self.connectivity = int(self.spec.get("connectivity", 8))
        self.forbid_corner_cutting = bool(self.spec.get("forbid_diagonal_corner_cutting", True))
        self.fallback_lookups = 0
        self._build()

    # ------------------------------------------------------------------ 构建
    def _build(self) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        self.xs = np.arange(xmin, xmax + 1e-9, self.res, dtype=np.float64)
        self.ys = np.arange(ymin, ymax + 1e-9, self.res, dtype=np.float64)
        self.nx, self.ny = len(self.xs), len(self.ys)
        blocked = np.zeros((self.ny, self.nx), dtype=bool)
        for iy, y in enumerate(self.ys):
            for ix, x in enumerate(self.xs):
                if x - self.inflation < xmin - 1e-9 or x + self.inflation > xmax + 1e-9:
                    blocked[iy, ix] = True
                    continue
                if y + self.inflation > ymax + 1e-9:
                    blocked[iy, ix] = True
                    continue
                for oxmin, oxmax, oymin, oymax in self.obstacles:
                    if circle_intersects_rect(x, y, self.inflation, oxmin, oxmax, oymin, oymax):
                        blocked[iy, ix] = True
                        break
        self.blocked = blocked
        self.dist = np.full((self.ny, self.nx), np.inf, dtype=np.float64)

        gx, gy = self.goal
        iy = int(round((gy - ymin) / self.res))
        ix = int(round((gx - xmin) / self.res))
        self.goal_index = (min(max(iy, 0), self.ny - 1), min(max(ix, 0), self.nx - 1))
        if blocked[self.goal_index]:
            raise RuntimeError(
                f"potential goal {self.goal} is blocked after inflation; "
                "map is not usable for PBRS"
            )
        self._dijkstra()

    def _dijkstra(self) -> None:
        iy, ix = self.goal_index
        dist = self.dist
        dist[iy, ix] = 0.0
        heap = [(0.0, iy, ix)]
        if self.connectivity == 8:
            steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                     (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                     (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        else:
            steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        while heap:
            d, cy, cx = heapq.heappop(heap)
            if d > dist[cy, cx]:
                continue
            for dy, dx, scale in steps:
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < self.ny and 0 <= nx < self.nx):
                    continue
                if self.blocked[ny, nx]:
                    continue
                if dy != 0 and dx != 0 and self.forbid_corner_cutting:
                    if self.blocked[cy, nx] or self.blocked[ny, cx]:
                        continue
                nd = d + scale * self.res
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))

    # ------------------------------------------------------------------ 查询
    def _nearest_index(self, x: float, y: float) -> tuple[int, int]:
        xmin, _xmax, ymin, _ymax = self.bounds
        ix = int(round((float(x) - xmin) / self.res))
        iy = int(round((float(y) - ymin) / self.res))
        return min(max(iy, 0), self.ny - 1), min(max(ix, 0), self.nx - 1)

    def distance_at(self, x: float, y: float) -> float:
        """查表距离；未命中有限格点时在邻域回退，仍失败取 Dmax。"""
        iy, ix = self._nearest_index(x, y)
        d = self.dist[iy, ix]
        if math.isfinite(d):
            return float(d)
        for radius in range(1, 5):
            iy0, iy1 = max(iy - radius, 0), min(iy + radius + 1, self.ny)
            ix0, ix1 = max(ix - radius, 0), min(ix + radius + 1, self.nx)
            window = self.dist[iy0:iy1, ix0:ix1]
            if np.isfinite(window).any():
                self.fallback_lookups += 1
                return float(np.nanmin(window))
        self.fallback_lookups += 1
        return float(self.dmax)

    def phi(self, x: float, y: float) -> float:
        return -min(self.distance_at(x, y), self.dmax) / self.dmax

    def is_reachable(self, x: float, y: float) -> bool:
        iy, ix = self._nearest_index(x, y)
        return bool(math.isfinite(self.dist[iy, ix]))

    def reachability_report(self, points: list[tuple[float, float]]) -> dict:
        details = {f"({x:.2f},{y:.2f})": self.is_reachable(x, y) for x, y in points}
        return {
            "goal": list(self.goal),
            "goal_distance": float(self.dist[self.goal_index]),
            "points": details,
            "all_reachable": all(details.values()),
            "fallback_lookups": self.fallback_lookups,
        }

    def hash(self) -> str:
        """场景哈希必须包含所有影响距离场的字段，不能只哈希 potential 子字典。"""
        return hashlib.sha256(_field_key(self.spec, self.bounds, self.obstacles,
                                         self.length, self.width).encode("utf-8")).hexdigest()


def _field_key(potential: dict, map_bounds, obstacles,
               length: float | None, width: float | None) -> str:
    payload = {
        "potential": potential,
        "map_bounds": list(map_bounds),
        "obstacles": [list(row) for row in obstacles],
        "length": length,
        "width": width,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def field_cache_key(env_spec: dict) -> str:
    return _field_key(env_spec["potential"], env_spec["map_bounds"],
                      env_spec["obstacles"], float(env_spec["length"]),
                      float(env_spec["width"]))


def get_distance_field(env_spec: dict) -> DistanceField:
    """按完整地图配置缓存：potential、边界、障碍、车体尺寸任一不同都隔离。"""
    key = field_cache_key(env_spec)
    if key not in _CACHE:
        _CACHE[key] = DistanceField(env_spec)
    return _CACHE[key]
