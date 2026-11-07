"""独立固定 seed 评估：不更新网络、不触碰训练回放池与探索 RNG。"""
from __future__ import annotations

import numpy as np

from agents.continuous import ContinuousAgent
from envs import make_env

from .collector import EpisodeTracker
from .metrics import summarize_episodes


def run_episode(agent: ContinuousAgent, env, seed: int, *, phase: str,
                keep_points: bool = False, episode_id: int = 0,
                frame_callback=None) -> dict:
    obs, info = env.reset(seed=int(seed))
    tracker = EpisodeTracker(episode_id, 0, int(seed), info["initial_state"],
                             keep_points=keep_points)
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs, 0.0, None)
        obs, reward, terminated, truncated, info = env.step(action)
        tracker.update(action, reward, info)
        if frame_callback is not None:
            frame_callback(env)
    record = tracker.finalize(info, tracker.length, phase=phase)
    record["seed"] = int(seed)
    return record


def evaluate_policy(agent: ContinuousAgent, env_spec: dict, seeds: list[int], *,
                    gamma: float, phase: str = "validation",
                    keep_points: bool = False,
                    render_mode: str | None = None) -> tuple[list[dict], dict]:
    env = make_env(env_spec, render_mode=render_mode, shaping_gamma=gamma)
    records = []
    try:
        for i, seed in enumerate(seeds):
            records.append(run_episode(agent, env, seed, phase=phase,
                                       keep_points=keep_points, episode_id=i))
    finally:
        env.close()
    return records, summarize_episodes(records)
