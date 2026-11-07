"""BattlefieldNavigation-v0：无武器二维运输/避障抽象场景。

在 CourseWhRobot-v1 的动力学与出口定义上，新增两处碎石路障与两个静态风险区；
每个外层 action 用 5 个 0.05 秒子步积分，逐子步检查碰撞/成功；
基础奖励新增风险代价 r_risk，可选 PBRS 势函数塑形（beta=0 时严格等于 r_base）。
"""
from __future__ import annotations

from .course_whrobot import CourseWhRobotEnv


class BattlefieldNavigationEnv(CourseWhRobotEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, spec: dict, render_mode: str | None = None,
                 shaping_gamma: float = 0.98):
        if spec.get("id") not in ("BattlefieldNavigation-v0",):
            raise ValueError(f"unexpected env id for battlefield: {spec.get('id')}")
        if int(spec.get("physics_substeps", 1)) != 5:
            raise ValueError("BattlefieldNavigation-v0 requires physics_substeps=5")
        if "risk" not in spec or "potential" not in spec:
            raise ValueError("battlefield spec requires risk and potential sections")
        if float(spec["dt"]) != 0.25:
            raise ValueError("battlefield spec requires dt=0.25 with 5 substeps")
        super().__init__(spec, render_mode=render_mode, shaping_gamma=shaping_gamma)

    def reachability_report(self, points: list[tuple[float, float]] | None = None) -> dict:
        if self._field is None:
            raise RuntimeError("battlefield requires a potential distance field")
        if points is None:
            # 保守膨胀会把贴近底障(y<2.58)或贴顶(y>6.92)的初始姿态判为不可达；
            # 报告点选择保守自由区内的代表位置，真实碰撞仍由 SAT 判定。
            points = [(-3.5, 4.0), (3.5, 4.0), (-6.0, 6.0), (6.0, 6.0),
                      (-2.6, 2.6), (2.6, 2.6), (0.0, 2.6)]
        return self._field.reachability_report(points)
