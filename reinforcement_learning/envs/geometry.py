"""二维几何：旋转矩形角点、多边形 SAT、墙与出口判定。"""
from __future__ import annotations

import math

import numpy as np


def rectangle_vertices(x: float, y: float, theta: float,
                       length: float = 1.8, width: float = 1.2) -> np.ndarray:
    """返回车体四个全局角点 (4,2)，顶点顺序与 SAT 无关但需连续排列。

    局部顶点顺序沿用参考片段：右上、左上、左下、右下。
    旋转必须使用 theta（姿态角），不能误用线速度。
    """
    hx, hy = float(length) / 2.0, float(width) / 2.0
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]], dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.array([float(x), float(y)], dtype=np.float64)


def bounds_vertices(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    """轴对齐矩形边界转凸多边形顶点。"""
    return np.array([[xmax, ymax], [xmin, ymax], [xmin, ymin], [xmax, ymin]], dtype=np.float64)


def sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 1e-9) -> bool:
    """分离轴定理：任一候选轴投影分离则不相交；接触视为相交（collision_touch_counts）。"""
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    if a.shape != (4, 2) or b.shape != (4, 2):
        raise ValueError("SAT expects two (4,2) polygons")
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.array([-edge[1], edge[0]], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm < eps:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() - eps or pb.max() < pa.min() - eps:
                return False
    return True


def crossing_bottom(x_old: float, y_old: float, x_new: float, y_new: float,
                    y_plane: float) -> tuple[bool, float]:
    """判断中心是否本步由 y_old >= 平面 跨到 y_new < 平面，并返回交点 x。"""
    if y_old >= y_plane > y_new:
        if y_new == y_old:
            return False, float("nan")
        t = (y_plane - y_old) / (y_new - y_old)
        return True, float(x_old + t * (x_new - x_old))
    return False, float("nan")


def polygon_hits_walls(poly: np.ndarray, bounds: tuple[float, float, float, float],
                       eps: float = 1e-9) -> bool:
    """侧墙与顶墙为硬边界；底边中央缺口开放，不在此判定。"""
    xmin, xmax, _ymin, ymax = bounds
    return bool(np.any(poly[:, 0] < xmin - eps) or np.any(poly[:, 0] > xmax + eps)
                or np.any(poly[:, 1] > ymax + eps))


def circle_intersects_rect(cx: float, cy: float, radius: float,
                           xmin: float, xmax: float, ymin: float, ymax: float,
                           eps: float = 0.0) -> bool:
    """圆心到轴对齐矩形的距离 <= radius（含接触）。"""
    nearest_x = min(max(cx, xmin), xmax)
    nearest_y = min(max(cy, ymin), ymax)
    return math.hypot(cx - nearest_x, cy - nearest_y) <= radius + eps


def point_rect_distance(cx: float, cy: float, xmin: float, xmax: float,
                        ymin: float, ymax: float) -> float:
    """点到轴对齐矩形的欧氏距离（矩形内部为 0）。"""
    nearest_x = min(max(cx, xmin), xmax)
    nearest_y = min(max(cy, ymin), ymax)
    return math.hypot(cx - nearest_x, cy - nearest_y)


def path_length(points: list[tuple[float, float]]) -> float:
    """折线累计欧氏长度。"""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return float(total)
