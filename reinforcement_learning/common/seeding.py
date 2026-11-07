"""随机源管理：训练、探索、验证互不干扰。"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, torch_num_threads: int = 1) -> None:
    """设置全局随机源；网络初始化发生在恢复 checkpoint 之前，因此调用顺序由训练器保证。"""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - 本机默认 CPU
        torch.cuda.manual_seed_all(seed)
    if torch_num_threads >= 1:
        torch.set_num_threads(int(torch_num_threads))


def make_rng(seed: int, salt: int = 0) -> np.random.Generator:
    """派生独立 Generator，避免与全局 NumPy 随机源耦合。"""
    return np.random.default_rng((int(seed) * 1000003 + int(salt)) % (2 ** 64))
