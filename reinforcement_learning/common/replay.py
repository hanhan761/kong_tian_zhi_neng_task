"""NumPy 预分配环形经验回放池。"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.truncated = np.empty((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.obs, self.actions, self.rewards, self.next_obs,
                self.terminated, self.truncated)

    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,) or next_obs.shape != obs.shape:
            raise ValueError("observation shape mismatch")
        if action.shape != (self.action_dim,):
            raise ValueError("action shape mismatch")
        if not all(np.isfinite(x).all() for x in (obs, action, next_obs)):
            raise ValueError("non-finite transition")
        if not math.isfinite(float(reward)):
            raise ValueError("non-finite reward")
        j = self.pos
        self.obs[j] = obs
        self.actions[j] = action
        self.rewards[j, 0] = float(reward)
        self.next_obs[j] = next_obs
        self.terminated[j, 0] = float(bool(terminated))
        self.truncated[j, 0] = float(bool(truncated))
        self.pos = (j + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device = "cpu"):
        if self.size < batch_size:
            raise ValueError("not enough samples in replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return tuple(torch.from_numpy(a[indices]).to(device) for a in self.arrays)

    def state_dict(self) -> dict:
        names = ("obs", "actions", "rewards", "next_obs", "terminated", "truncated")
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "pos": self.pos,
            "size": self.size,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "arrays": {name: getattr(self, name)[: self.size].copy() for name in names},
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("buffer capacity mismatch")
        if int(state["obs_dim"]) != self.obs_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("buffer dimension mismatch")
        size, pos = int(state["size"]), int(state["pos"])
        if not 0 <= size <= self.capacity or not 0 <= pos < max(self.capacity, 1):
            raise ValueError("invalid buffer counters")
        if size < self.capacity and pos != size:
            raise ValueError("invalid unfilled buffer position")
        expected = {"obs", "actions", "rewards", "next_obs", "terminated", "truncated"}
        if set(state["arrays"]) != expected:
            raise ValueError("missing or unknown buffer arrays")
        for name, values in state["arrays"].items():
            target = getattr(self, name)
            if values.shape != (size,) + target.shape[1:]:
                raise ValueError("buffer array shape mismatch")
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("invalid buffer array")
        for name, values in state["arrays"].items():
            getattr(self, name)[:size] = values
        self.pos, self.size = pos, size
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
