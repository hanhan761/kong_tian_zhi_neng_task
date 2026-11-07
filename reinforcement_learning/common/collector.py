"""精确 transition 采集器：跨块保留未结束回合，恰好采集 N 条。"""
from __future__ import annotations

import math

import numpy as np

from .metrics import ensure_finite

TERM_KEYS = ("center", "exit", "time", "omega", "action", "success_bonus",
             "collision_penalty", "risk", "shaping")


class EpisodeTracker:
    """累计单个回合的回报、分项成本、路径与结局。"""

    def __init__(self, episode_id: int, start_env_step: int, seed_hint: int,
                 initial_state, keep_points: bool = False):
        self.episode_id = int(episode_id)
        self.start_env_step = int(start_env_step)
        self.seed_hint = int(seed_hint)
        self.initial_state = [float(v) for v in initial_state]
        self.return_ = 0.0
        self.raw_base_return = 0.0
        self.shaped_return = 0.0
        self.reward_task = 0.0
        self.reward_risk = 0.0
        self.terms = {key: 0.0 for key in TERM_KEYS}
        self.length = 0
        self.path_length = 0.0
        self.energy = 0.0
        self.risk_integral = 0.0
        self.risk_zone_entries = 0
        self.prev_risk_h = 0.0
        self.min_clearance = float("inf")
        self.keep_points = bool(keep_points)
        self.points = [(self.initial_state[0], self.initial_state[1])] if keep_points else []
        self.final_position = self.initial_state[:3]

    def update(self, action, reward: float, info: dict) -> None:
        self.return_ += float(reward)
        self.raw_base_return += float(info["reward_task"]) + float(info["reward_risk"])
        self.shaped_return += float(info["reward_shaping"])
        self.reward_task += float(info["reward_task"])
        self.reward_risk += float(info["reward_risk"])
        for key in TERM_KEYS:
            self.terms[key] += float(info["reward_terms"][key])
        self.length += 1
        self.energy += -float(info["reward_terms"]["action"]) / 0.01
        self.risk_integral += float(info["risk_integral_step"])
        risk_h = float(info.get("risk_h", 0.0))
        if risk_h >= 0.05 and self.prev_risk_h < 0.05:
            self.risk_zone_entries += 1
        self.prev_risk_h = risk_h
        clearance = info.get("min_clearance_step")
        if clearance is not None:
            self.min_clearance = min(self.min_clearance, float(clearance))
        x, y = float(info["position"][0]), float(info["position"][1])
        if self.keep_points:
            self.points.append((x, y))
        self.path_length += math.hypot(x - self.final_position[0], y - self.final_position[1])
        self.final_position = list(info["position"])

    def finalize(self, info: dict, end_env_step: int, phase: str = "train") -> dict:
        record = {
            "phase": phase,
            "episode_id": self.episode_id,
            "episode_seed": self.seed_hint,
            "start_env_step": self.start_env_step,
            "end_env_step": int(end_env_step),
            "initial_state": self.initial_state,
            "return": float(self.return_),
            "raw_base_return": float(self.raw_base_return),
            "shaped_return": float(self.shaped_return),
            "reward_task": float(self.reward_task),
            "reward_risk": float(self.reward_risk),
            "reward_terms": {key: float(v) for key, v in self.terms.items()},
            "length": int(self.length),
            "elapsed_sim_time": float(info["elapsed_sim_time"]),
            "path_length": float(self.path_length),
            "energy": float(self.energy),
            "risk_integral": float(self.risk_integral),
            "risk_zone_entries": int(self.risk_zone_entries),
            "min_clearance": (None if math.isinf(self.min_clearance)
                              else float(self.min_clearance)),
            "success": bool(info["success"]),
            "collision": bool(info["collision"]),
            "timeout": bool(info["timeout"]),
            "final_position": list(self.final_position),
            "exit_cross_x": info.get("exit_cross_x"),
        }
        ensure_finite({"return": record["return"], "raw_base_return": record["raw_base_return"]},
                      context=f"episode {self.episode_id}")
        if self.keep_points:
            record["trajectory"] = [[float(x), float(y)] for x, y in self.points]
        return record

    def state_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "start_env_step": self.start_env_step,
            "seed_hint": self.seed_hint,
            "initial_state": self.initial_state,
            "return": self.return_,
            "raw_base_return": self.raw_base_return,
            "shaped_return": self.shaped_return,
            "reward_task": self.reward_task,
            "reward_risk": self.reward_risk,
            "terms": dict(self.terms),
            "length": self.length,
            "path_length": self.path_length,
            "energy": self.energy,
            "risk_integral": self.risk_integral,
            "risk_zone_entries": self.risk_zone_entries,
            "prev_risk_h": self.prev_risk_h,
            "min_clearance": None if math.isinf(self.min_clearance) else self.min_clearance,
            "keep_points": self.keep_points,
            "points": [[float(x), float(y)] for x, y in self.points],
            "final_position": list(self.final_position),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "EpisodeTracker":
        tracker = cls(state["episode_id"], state["start_env_step"], state["seed_hint"],
                      state["initial_state"], keep_points=state.get("keep_points", False))
        tracker.return_ = float(state["return"])
        tracker.raw_base_return = float(state["raw_base_return"])
        tracker.shaped_return = float(state["shaped_return"])
        tracker.reward_task = float(state["reward_task"])
        tracker.reward_risk = float(state["reward_risk"])
        tracker.terms = {key: float(state["terms"][key]) for key in TERM_KEYS}
        tracker.length = int(state["length"])
        tracker.path_length = float(state["path_length"])
        tracker.energy = float(state["energy"])
        tracker.risk_integral = float(state["risk_integral"])
        tracker.risk_zone_entries = int(state["risk_zone_entries"])
        tracker.prev_risk_h = float(state["prev_risk_h"])
        tracker.min_clearance = (float("inf") if state["min_clearance"] is None
                                 else float(state["min_clearance"]))
        tracker.points = [(float(x), float(y)) for x, y in state.get("points", [])]
        tracker.final_position = list(state["final_position"])
        return tracker


class Collector:
    def __init__(self, env, agent, replay, *, exploration_rng: np.random.Generator,
                 seed_rng: np.random.Generator, noise_decay: float, eval_interval: int,
                 keep_points: bool = False, start_random_steps: int = 0):
        self.env = env
        self.agent = agent
        self.replay = replay
        self.exploration_rng = exploration_rng
        self.seed_rng = seed_rng
        self.noise_decay = float(noise_decay)
        self.eval_interval = int(eval_interval)
        self.keep_points = bool(keep_points)
        self.start_random_steps = int(start_random_steps)
        if self.start_random_steps < 0:
            raise ValueError("start_random_steps must be nonnegative")
        self.obs = None
        self.tracker = None
        self.episode_index = 0
        self.t_env = 0

    def start(self, train_seed: int) -> dict:
        obs, info = self.env.reset(seed=int(train_seed))
        self.obs = np.asarray(obs, dtype=np.float32)
        self.t_env = 0
        self.episode_index = 0
        self.tracker = self._new_tracker(obs, info)
        return info

    def _new_tracker(self, obs, info: dict) -> EpisodeTracker:
        seed_hint = int(self.seed_rng.integers(0, 2 ** 31 - 1))
        tracker = EpisodeTracker(self.episode_index, self.t_env, seed_hint,
                                 info.get("initial_state", [float(obs[0]), float(obs[1]), 0.0, 0.0, 0.0]),
                                 keep_points=self.keep_points)
        self.episode_index += 1
        return tracker

    def noise_sigma(self, sigma0: float) -> float:
        return float(sigma0) * self.noise_decay ** (self.t_env // self.eval_interval)

    def collect(self, num_steps: int, sigma0: float, on_episode) -> int:
        """恰好采集 num_steps 条 transition；返回实际采集数。"""
        if num_steps < 0:
            raise ValueError("num_steps must be nonnegative")
        if self.tracker is None:
            raise RuntimeError("collector must be started or restored before collect()")
        collected = 0
        sigma = self.noise_sigma(sigma0)
        while collected < num_steps:
            if self.t_env < self.start_random_steps:
                action = self.exploration_rng.uniform(
                    self.env.action_space.low, self.env.action_space.high
                ).astype(np.float32)
            else:
                action = self.agent.act(self.obs, sigma, self.exploration_rng)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay.add(self.obs, action, reward, next_obs, terminated, truncated)
            self.t_env += 1
            collected += 1
            self.tracker.update(action, reward, info)
            self.obs = np.asarray(next_obs, dtype=np.float32)
            if terminated or truncated:
                record = self.tracker.finalize(info, self.t_env, phase="train")
                on_episode(record)
                self.obs, reset_info = self.env.reset(seed=None)
                self.tracker = self._new_tracker(self.obs, reset_info)
        return collected

    def state_dict(self) -> dict:
        return {
            "obs": [float(v) for v in self.obs] if self.obs is not None else None,
            "t_env": self.t_env,
            "episode_index": self.episode_index,
            "tracker": None if self.tracker is None else self.tracker.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.obs = None if state["obs"] is None else np.asarray(state["obs"], dtype=np.float32)
        self.t_env = int(state["t_env"])
        self.episode_index = int(state["episode_index"])
        self.tracker = None if state["tracker"] is None else EpisodeTracker.from_state_dict(state["tracker"])
