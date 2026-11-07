"""CourseWhRobot-v1：修正后的仓库机器人 Gymnasium 环境。

相对课程原版的必要修正（不影响课程公开的动力学/奖励/地图）：
- 角点旋转使用姿态角 theta（原版误用线速度 state[3]）；
- 随机源使用 self.np_random，同 seed 可复现，reset 时重算角点；
- 观测/奖励/终止标志返回 NumPy float32 / float / bool；
- 碰撞使用旋转矩形与 AABB 的 SAT，接触算碰撞；侧/顶墙硬边界，中央底出口开放；
- success 与 collision 互斥，碰撞优先；200 步上限是 truncated。
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .geometry import (bounds_vertices, crossing_bottom, point_rect_distance,
                       polygon_hits_walls, rectangle_vertices, sat_overlap)
from .potential import get_distance_field

TWO_PI = 2.0 * math.pi


class CourseWhRobotEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, spec: dict, render_mode: str | None = None,
                 shaping_gamma: float = 0.98):
        super().__init__()
        self.env_spec = spec
        self.render_mode = render_mode
        self.render_mode = render_mode if render_mode in (None, "rgb_array") else None

        self.mass = float(spec["mass"])
        self.length = float(spec["length"])
        self.width = float(spec["width"])
        self.dt = float(spec["dt"])
        self.substeps = int(spec.get("physics_substeps", 1))
        self.dt_sub = self.dt / self.substeps
        self.j = float(spec.get("inertia", self.mass * (self.length ** 2 + self.width ** 2) / 12.0))
        self.v_limit = float(spec["speed_limit"])
        self.w_limit = float(spec["angular_speed_limit"])
        self.map_bounds = tuple(float(v) for v in spec["map_bounds"])
        self.exit_half_width = float(spec["exit_half_width"])
        self.max_steps = int(spec["max_episode_steps"])
        self.bounds_poly = bounds_vertices(*self.map_bounds)

        self.obstacles = [tuple(float(v) for v in row) for row in spec["obstacles"]]
        self.obstacle_polys = [bounds_vertices(*row) for row in self.obstacles]
        self.initial_spec = dict(spec["initial_state"])
        self.reward_spec = dict(spec["reward"])

        self.risk_spec = spec.get("risk")
        self.risk_centers = []
        self.risk_radius = 1.0
        self.kappa = 0.0
        self.potential_spec = spec.get("potential")
        self.shaping_gamma = float(shaping_gamma)
        self._field = None
        if self.risk_spec:
            self.risk_centers = [tuple(map(float, c)) for c in self.risk_spec["centers"]]
            self.risk_radius = float(self.risk_spec["radius"])
            self.kappa = float(self.risk_spec["kappa"])
        if self.potential_spec:
            self._field = get_distance_field(spec)
            self.beta = float(self.potential_spec["beta"])

        obs_low = np.array([-np.inf, -np.inf, -1.0, -1.0, -self.v_limit, -self.w_limit],
                           dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, 1.0, 1.0, self.v_limit, self.w_limit],
                            dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 物理状态 q=[x, y, theta, v, omega]
        self.q = np.zeros(5, dtype=np.float64)
        self.corners = None
        self.episode_steps = 0
        self.elapsed_sim_time = 0.0
        self._done = True
        self._fig = None
        self._canvas = None
        self._robot_patch = None
        self._ax = None

    # ------------------------------------------------------------------ 基础
    @property
    def observation(self) -> np.ndarray:
        x, y, theta, v, omega = self.q
        return np.array([x, y, math.sin(theta), math.cos(theta), v, omega], dtype=np.float32)

    def _is_collision(self, corners: np.ndarray) -> bool:
        for poly in self.obstacle_polys:
            if sat_overlap(corners, poly):
                return True
        return polygon_hits_walls(corners, self.map_bounds)

    def _risk_h(self, x: float, y: float) -> float:
        if not self.risk_centers:
            return 0.0
        best = 0.0
        for cx, cy in self.risk_centers:
            value = math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * self.risk_radius ** 2))
            best = max(best, value)
        return float(best)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        init = self.initial_spec
        ax_min, ax_max = init["abs_x_range"]
        y_min, y_max = init["y_range"]
        th_min, th_max = init["theta_range"]
        rejections = 0
        for attempt in range(int(init["max_attempts"])):
            x = rng.uniform(ax_min, ax_max)
            if rng.random() >= float(init["left_probability"]):
                x = -x
            y = rng.uniform(y_min, y_max)
            theta = rng.uniform(th_min, th_max)
            v = float(np.clip(rng.normal(init["v_normal_mean"], init["v_normal_std"]),
                             -self.v_limit, self.v_limit))
            omega = float(np.clip(rng.normal(init["omega_normal_mean"], init["omega_normal_std"]),
                                  -self.w_limit, self.w_limit))
            q = np.array([x, y, theta, v, omega], dtype=np.float64)
            corners = rectangle_vertices(x, y, theta, self.length, self.width)
            if not self._is_collision(corners):
                self.q = q
                self.corners = corners
                break
            rejections += 1
        else:
            raise RuntimeError(
                f"failed to sample a collision-free initial state after {init['max_attempts']} attempts"
            )

        self.episode_steps = 0
        self.elapsed_sim_time = 0.0
        self._done = False
        info = {"initial_state": self.q.copy(), "initial_collision_rejections": rejections}
        return self.observation, info

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self._done:
            raise RuntimeError("episode has ended; call reset() before step()")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        executed = np.clip(action, -1.0, 1.0)

        self.episode_steps += 1
        x_old, y_old = float(self.q[0]), float(self.q[1])
        phi_old = self._field.phi(x_old, y_old) if (self._field is not None and self.beta != 0.0) else 0.0

        executed_substeps = 0
        collision = False
        success = False
        assemble = {"time": 0.0, "omega": 0.0, "action": 0.0, "risk": 0.0}
        risk_integral = 0.0
        risk_h_max = 0.0
        clearance = float("inf")
        x_cross = float("nan")

        for _ in range(self.substeps):
            sub_x_old, sub_y_old = float(self.q[0]), float(self.q[1])
            v_new = float(np.clip(self.q[3] + self.dt_sub * executed[0] / self.mass,
                                  -self.v_limit, self.v_limit))
            omega_new = float(np.clip(self.q[4] + self.dt_sub * executed[1] / self.j,
                                      -self.w_limit, self.w_limit))
            theta_new = self.q[2] + self.dt_sub * omega_new
            theta_new = theta_new - math.floor((theta_new + math.pi) / TWO_PI) * TWO_PI
            x_new = float(self.q[0] + self.dt_sub * v_new * math.cos(theta_new))
            y_new = float(self.q[1] + self.dt_sub * v_new * math.sin(theta_new))

            self.q = np.array([x_new, y_new, theta_new, v_new, omega_new], dtype=np.float64)
            self.corners = rectangle_vertices(x_new, y_new, theta_new, self.length, self.width)
            executed_substeps += 1

            weight = self.dt_sub / 0.25
            assemble["time"] += weight * self.reward_spec["time_cost"]
            assemble["omega"] += weight * self.reward_spec["omega_cost"] * omega_new ** 2
            assemble["action"] += weight * self.reward_spec["action_squared_cost"] * float(np.sum(executed ** 2))
            h = self._risk_h(x_new, y_new)
            risk_h_max = max(risk_h_max, h)
            if h > 0.0:
                cost = weight * h
                assemble["risk"] += cost
                risk_integral += weight * h
            if self.obstacles:
                nearest = min(point_rect_distance(x_new, y_new, *rect) for rect in self.obstacles)
                clearance = min(clearance, nearest - 0.5 * math.hypot(self.length, self.width))

            collision = self._is_collision(self.corners)
            if collision:
                break

            crossed, cross_x = crossing_bottom(sub_x_old, sub_y_old, x_new, y_new,
                                               self.map_bounds[2])
            if crossed:
                x_cross = cross_x
                if abs(cross_x) >= self.exit_half_width:
                    collision = True
                else:
                    success = True
                break

        self.elapsed_sim_time += self.dt_sub * executed_substeps

        if collision:
            success = False
        terminated = bool(success or collision)
        truncated = bool(self.episode_steps >= self.max_steps and not terminated)
        self._done = bool(terminated or truncated)

        x_now, y_now = float(self.q[0]), float(self.q[1])
        center = self.reward_spec["center_coef"] * math.exp(
            -self.reward_spec["center_exponent"] * x_now ** 2)
        exit_term = self.reward_spec["exit_coef"] * math.exp(
            -self.reward_spec["exit_exponent"] * (y_now + self.reward_spec["exit_y_shift"]))
        success_bonus = self.reward_spec["success_bonus"] if success else 0.0
        collision_penalty = self.reward_spec["collision_penalty"] if collision else 0.0
        reward_task = (center + exit_term - assemble["time"] - assemble["omega"]
                       - assemble["action"] + success_bonus - collision_penalty)
        reward_risk = -self.kappa * assemble["risk"] if self.risk_centers else 0.0

        if self._field is not None and self.beta != 0.0:
            phi_new = 0.0 if terminated else self._field.phi(x_now, y_now)
            shaping = self.beta * (self.shaping_gamma * phi_new - phi_old)
        else:
            shaping = 0.0

        raw_reward = float(reward_task + reward_risk)
        reward = float(raw_reward + shaping)

        info: dict[str, Any] = {
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(truncated),
            "episode_step": int(self.episode_steps),
            "elapsed_sim_time": float(self.elapsed_sim_time),
            "position": [float(self.q[0]), float(self.q[1]), float(self.q[2])],
            "velocity": [float(self.q[3]), float(self.q[4])],
            "substeps_executed": int(executed_substeps),
            "exit_cross_x": None if math.isnan(x_cross) else float(x_cross),
            "reward_terms": {
                "center": float(center),
                "exit": float(exit_term),
                "time": float(-assemble["time"]),
                "omega": float(-assemble["omega"]),
                "action": float(-assemble["action"]),
                "success_bonus": float(success_bonus),
                "collision_penalty": float(-collision_penalty),
                "risk": float(reward_risk),
                "shaping": float(shaping),
            },
            "reward_task": float(reward_task),
            "reward_risk": float(reward_risk),
            "reward_shaping": float(shaping),
            "raw_reward": raw_reward,
            "reward": reward,
            "risk_integral_step": float(risk_integral),
            "risk_h": float(risk_h_max),
            "min_clearance_step": float(clearance) if math.isfinite(clearance) else None,
        }
        return self.observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ 状态
    def get_state(self) -> dict:
        """完整环境状态（供精确续训），不使用 reset 替代。"""
        return {
            "q": [float(v) for v in self.q],
            "episode_steps": int(self.episode_steps),
            "elapsed_sim_time": float(self.elapsed_sim_time),
            "done": bool(self._done),
            "np_random": self.np_random.bit_generator.state,
        }

    def set_state(self, state: dict) -> None:
        self.q = np.asarray(state["q"], dtype=np.float64).copy()
        self.episode_steps = int(state["episode_steps"])
        self.elapsed_sim_time = float(state["elapsed_sim_time"])
        self._done = bool(state["done"])
        self.np_random.bit_generator.state = state["np_random"]
        self.corners = rectangle_vertices(self.q[0], self.q[1], self.q[2], self.length, self.width)

    # ------------------------------------------------------------------ render
    def _init_render(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, Polygon, Rectangle

        self._fig = Figure(figsize=(6.4, 4.8), dpi=100)
        self._canvas = FigureCanvasAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlim(self.map_bounds[0], self.map_bounds[1])
        self._ax.set_ylim(self.map_bounds[2], self.map_bounds[3])
        self._ax.set_aspect("equal")
        self._ax.set_title("CourseWhRobot-v1")
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        for xmin, xmax, ymin, ymax in self.obstacles:
            self._ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         facecolor="0.2", edgecolor="0.1"))
        if self.risk_centers:
            for cx, cy in self.risk_centers:
                self._ax.add_patch(Circle((cx, cy), self.risk_radius, facecolor="red",
                                          alpha=0.25, edgecolor="darkred"))
        self._ax.plot([-self.exit_half_width, self.exit_half_width],
                      [self.map_bounds[2]] * 2, "g--", linewidth=1.5, label="exit")
        self._robot_patch = Polygon([(0, 0)] * 4, closed=True, facecolor=[0, 0.75, 0],
                                    edgecolor="black")
        self._ax.add_patch(self._robot_patch)
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc="upper right", fontsize=7)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._fig is None:
            self._init_render()
        self._robot_patch.set_xy(self.corners)
        self._ax.set_title(
            f"{self.env_spec['id']} - step {self.episode_steps} - t={self.elapsed_sim_time:5.2f}")
        self._fig.canvas.draw()
        buffer = np.asarray(self._fig.canvas.buffer_rgba())
        return buffer[:, :, :3].copy()

    def close(self) -> None:
        self._fig = None
        self._canvas = None
        self._robot_patch = None
        self._ax = None
