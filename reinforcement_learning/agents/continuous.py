"""手写连续控制智能体（DDPG / TD3），共享同一接口。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AgentConfig:
    algorithm: str = "ddpg"
    obs_dim: int = 6
    action_dim: int = 2
    hidden_dim: int = 256
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    gamma: float = 0.98
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    device: str = "cpu"


def mlp(input_dim: int, output_dim: int, hidden_dim: int, tanh: bool = False) -> nn.Sequential:
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(),
              nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for p_target, p_online in zip(target.parameters(), online.parameters()):
        p_target.lerp_(p_online, tau)


@torch.no_grad()
def compute_target(reward: torch.Tensor, terminated: torch.Tensor,
                   next_q: torch.Tensor, gamma: float) -> torch.Tensor:
    """Bellman 目标只屏蔽真实 terminated；人工 timeout 继续 bootstrap。"""
    if reward.ndim != 2 or reward.shape[1] != 1 or reward.shape != terminated.shape \
            or reward.shape != next_q.shape:
        raise ValueError("target inputs must all have the same B x 1 shape")
    return reward + gamma * (1.0 - terminated) * next_q


class ContinuousAgent:
    def __init__(self, cfg: AgentConfig):
        if cfg.algorithm not in ("ddpg", "td3"):
            raise ValueError("unknown algorithm")
        if not 0 < cfg.gamma < 1 or not 0 < cfg.tau <= 1:
            raise ValueError("invalid gamma/tau")
        if cfg.policy_delay < 1:
            raise ValueError("policy_delay must be positive")
        if min(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim) < 1:
            raise ValueError("dimensions must be positive")
        if cfg.target_noise < 0 or cfg.noise_clip < 0:
            raise ValueError("noise scales must be nonnegative")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = mlp(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, True).to(self.device)
        self.device = next(self.actor.parameters()).device
        self.q1 = mlp(cfg.obs_dim + cfg.action_dim, 1, cfg.hidden_dim).to(self.device)
        self.q2 = (mlp(cfg.obs_dim + cfg.action_dim, 1, cfg.hidden_dim).to(self.device)
                   if cfg.algorithm == "td3" else None)
        self.actor_target = copy.deepcopy(self.actor)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2) if self.q2 is not None else None
        for network in (self.actor_target, self.q1_target, self.q2_target):
            if network is not None:
                network.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        critic_params = list(self.q1.parameters())
        if self.q2 is not None:
            critic_params += list(self.q2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
        self.gradient_steps = 0
        self.actor_updates = 0

    # ------------------------------------------------------------------ 推理
    @torch.no_grad()
    def act(self, observation: np.ndarray, noise_std: float = 0.0,
            noise_rng: np.random.Generator | None = None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.cfg.obs_dim,):
            raise ValueError("act expects exactly one observation with shape (obs_dim,)")
        if not np.isfinite(obs).all() or not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("invalid observation or noise scale")
        action = self.actor(torch.as_tensor(obs, device=self.device).unsqueeze(0))[0].cpu().numpy()
        if noise_std > 0:
            if noise_rng is None:
                raise ValueError("exploration requires an explicit RNG")
            action = action + noise_rng.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ 更新
    def update(self, batch: tuple[torch.Tensor, ...]) -> dict:
        if len(batch) != 6 or any(not isinstance(t, torch.Tensor) for t in batch):
            raise ValueError("batch must contain six tensors")
        obs, actions, rewards, next_obs, terminated, truncated = batch
        b = obs.shape[0] if obs.ndim == 2 else 0
        expected = ((b, self.cfg.obs_dim), (b, self.cfg.action_dim), (b, 1),
                    (b, self.cfg.obs_dim), (b, 1), (b, 1))
        if b < 1 or any(tuple(t.shape) != s for t, s in zip(batch, expected)):
            raise ValueError("batch shape mismatch; scalar columns must be B x 1")
        if any(t.device != self.device for t in batch):
            raise ValueError("batch device mismatch")
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in batch):
            raise ValueError("batch must be finite float32 tensors")
        if any(not torch.all((t == 0) | (t == 1)) for t in (terminated, truncated)):
            raise ValueError("termination flags must be binary")

        c = self.cfg
        self.gradient_steps += 1
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            if c.algorithm == "td3":
                noise = (torch.randn_like(next_actions) * c.target_noise).clamp(-c.noise_clip, c.noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            next_input = torch.cat((next_obs, next_actions), dim=1)
            next_q = self.q1_target(next_input)
            if self.q2 is not None:
                next_q = torch.minimum(next_q, self.q2_target(next_input))
            target = compute_target(rewards, terminated, next_q, c.gamma)

        qa = torch.cat((obs, actions), dim=1)
        current_q = self.q1(qa)
        critic_loss = F.mse_loss(current_q, target)
        if self.q2 is not None:
            critic_loss = critic_loss + F.mse_loss(self.q2(qa), target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("critic loss is not finite")
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        update_policy = c.algorithm == "ddpg" or self.gradient_steps % c.policy_delay == 0
        actor_loss_value = None
        if update_policy:
            self.q1.requires_grad_(False)
            policy_q = self.q1(torch.cat((obs, self.actor(obs)), dim=1))
            actor_loss = -policy_q.mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("actor loss is not finite")
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.q1.requires_grad_(True)
            soft_update(self.actor_target, self.actor, c.tau)
            soft_update(self.q1_target, self.q1, c.tau)
            if self.q2 is not None:
                soft_update(self.q2_target, self.q2, c.tau)
            actor_loss_value = float(actor_loss.detach())
            self.actor_updates += 1

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": actor_loss_value,
            "actor_updated": bool(update_policy),
            "gradient_steps": self.gradient_steps,
            "actor_updates": self.actor_updates,
            "q_mean": float(current_q.detach().mean()),
            "target_mean": float(target.mean()),
        }

    # ------------------------------------------------------------------ 状态
    def num_parameters(self) -> int:
        nets = [self.actor, self.q1] + ([self.q2] if self.q2 is not None else [])
        return int(sum(p.numel() for net in nets for p in net.parameters()))

    def checkpoint_state(self) -> dict:
        state = {
            "config": asdict(self.cfg),
            "gradient_steps": self.gradient_steps,
            "actor_updates": self.actor_updates,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(),
        }
        if self.q2 is not None:
            state["q2"] = self.q2.state_dict()
            state["q2_target"] = self.q2_target.state_dict()
        return state

    @classmethod
    def from_checkpoint_state(cls, state: dict, device: str = "cpu") -> "ContinuousAgent":
        config = dict(state["config"])
        config["device"] = device
        agent = cls(AgentConfig(**config))
        for name in ("actor", "actor_target", "q1", "q1_target", "q2", "q2_target"):
            network = getattr(agent, name)
            if network is not None:
                network.load_state_dict(state[name])
        agent.actor_opt.load_state_dict(state["actor_optimizer"])
        agent.critic_opt.load_state_dict(state["critic_optimizer"])
        agent.gradient_steps = int(state["gradient_steps"])
        agent.actor_updates = int(state.get("actor_updates", 0))
        return agent

    def policy_state(self) -> dict:
        """轻量推理快照：只含 actor 与环境重建所需配置。"""
        return {"config": asdict(self.cfg), "actor": self.actor.state_dict()}

    @classmethod
    def from_policy_state(cls, state: dict, device: str = "cpu") -> "ContinuousAgent":
        config = dict(state["config"])
        config["device"] = device
        agent = cls(AgentConfig(**config))
        agent.actor.load_state_dict(state["actor"])
        agent.actor.eval()
        return agent
