"""环境注册与构造。"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import gymnasium as gym

from .battlefield import BattlefieldNavigationEnv
from .course_whrobot import CourseWhRobotEnv

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def make_env(env_spec: dict, render_mode: str | None = None, shaping_gamma: float = 0.98):
    env_id = env_spec.get("id")
    if env_id == "CourseWhRobot-v1":
        return CourseWhRobotEnv(env_spec, render_mode=render_mode, shaping_gamma=shaping_gamma)
    if env_id == "BattlefieldNavigation-v0":
        return BattlefieldNavigationEnv(env_spec, render_mode=render_mode, shaping_gamma=shaping_gamma)
    raise ValueError(f"unknown env id: {env_id}")


def _default_spec(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as fh:
        return json.load(fh)["env"]


_registered = False


def register_default_envs() -> None:
    """注册无超参入口（默认 spec 取自 configs），便于 gym.make 与 env_checker 复核。"""
    global _registered
    if _registered:
        return
    try:
        course = functools.partial(CourseWhRobotEnv, _default_spec("ddpg_baseline.json"))
        battle = functools.partial(BattlefieldNavigationEnv, _default_spec("battle_W1.json"))
        gym.register(id="CourseWhRobot-v1", entry_point=course,
                     vector_entry_point=None, disable_env_checker=False)
        gym.register(id="BattlefieldNavigation-v0", entry_point=battle,
                     vector_entry_point=None, disable_env_checker=False)
        _registered = True
    except FileNotFoundError:
        # 配置尚未复制完成时跳过注册，实验入口使用 make_env(resolved_spec)。
        pass


register_default_envs()
