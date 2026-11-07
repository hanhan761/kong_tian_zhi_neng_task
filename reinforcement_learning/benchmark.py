"""端到端吞吐 benchmark：采集 + UTD 更新 + 周期评估，为预算决策提供实测数据。"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agents.continuous import AgentConfig, ContinuousAgent  # noqa: E402
from common.collector import Collector  # noqa: E402
from common.config import load_config, resolve_config  # noqa: E402
from common.evaluation import evaluate_policy  # noqa: E402
from common.replay import ReplayBuffer  # noqa: E402
from common.seeding import make_rng, set_global_seed  # noqa: E402
from envs import make_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练循环吞吐 benchmark")
    parser.add_argument("--config", default="configs/ddpg_baseline.json")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="仅用于吞吐对照，不改写配置")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(BASE_DIR / args.config)
    resolved = resolve_config(cfg, seed=0, base_dir=BASE_DIR)
    seed = 0
    threads = int(args.torch_threads or cfg["training"]["torch_num_threads"])
    set_global_seed(seed, threads)
    agent_cfg = AgentConfig(**{k: cfg["agent"][k] for k in
                               ("algorithm", "obs_dim", "action_dim", "hidden_dim", "actor_lr",
                                "critic_lr", "gamma", "tau", "policy_delay", "target_noise",
                                "noise_clip", "device")})
    env = make_env(cfg["env"], shaping_gamma=cfg["agent"]["gamma"])
    agent = ContinuousAgent(agent_cfg)
    replay = ReplayBuffer(int(cfg["training"]["buffer_capacity"]), agent_cfg.obs_dim,
                          agent_cfg.action_dim, seed=1)
    collector = Collector(env, agent, replay, exploration_rng=make_rng(seed, 1),
                          seed_rng=make_rng(seed, 2),
                          noise_decay=float(cfg["training"]["noise_decay_per_eval"]),
                          eval_interval=int(cfg["training"]["eval_interval"]))
    collector.start(seed)
    batch_size = int(cfg["training"]["batch_size"])
    gpb = int(cfg["training"]["gradient_steps_per_block"])
    cpb = int(cfg["training"]["collect_steps"])
    sigma0 = float(cfg["training"]["exploration_noise"])

    t0 = time.perf_counter()
    collector.collect(args.warmup_steps, sigma0, lambda record: None)
    warmup_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    remaining = int(args.steps)
    gradient_steps = 0
    while remaining > 0:
        n = min(cpb, remaining)
        collector.collect(n, sigma0, lambda record: None)
        desired = (collector.t_env * gpb) // cpb
        pending = max(desired - agent.gradient_steps, 0)
        if replay.size >= max(batch_size, int(cfg["training"]["learning_starts"])):
            for _ in range(pending):
                agent.update(replay.sample(batch_size, agent.device))
                gradient_steps += 1
        remaining -= n
    steps_seconds = time.perf_counter() - t0

    # 纯环境步进（不更新）参照
    t0 = time.perf_counter()
    collector.collect(3000, sigma0, lambda record: None)
    env_only_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    _records, summary = evaluate_policy(agent, cfg["env"],
                                        cfg["evaluation"]["validation_seeds"], gamma=cfg["agent"]["gamma"],
                                        phase="validation")
    eval_seconds = time.perf_counter() - t0

    rss_mb = None
    try:
        import psutil
        rss_mb = round(psutil.Process().memory_info().rss / 2 ** 20, 1)
    except Exception:
        pass

    payload = {
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "numpy": np.__version__,
                    "cpu_count": __import__("os").cpu_count(),
                    "torch_threads": torch.get_num_threads(),
                    "cuda_available": torch.cuda.is_available()},
        "algorithm": cfg["agent"]["algorithm"],
        "env_id": cfg["env"]["id"],
        "warmup_steps": args.warmup_steps,
        "warmup_seconds": round(warmup_seconds, 3),
        "measured_env_steps": args.steps,
        "measured_seconds": round(steps_seconds, 3),
        "env_steps_per_second": round(args.steps / steps_seconds, 1),
        "gradient_steps": gradient_steps,
        "gradient_steps_per_second": round(gradient_steps / steps_seconds, 1),
        "env_only_steps_per_second": round(3000 / env_only_seconds, 1),
        "eval_episodes": len(cfg["evaluation"]["validation_seeds"]),
        "eval_seconds": round(eval_seconds, 3),
        "eval_overhead_ratio": round(eval_seconds / (steps_seconds + eval_seconds), 3),
        "rss_mb": rss_mb,
        "estimated_hours_per_1m_steps_with_eval": round(
            (1_000_000 / (args.steps / steps_seconds)
             + (1_000_000 / int(cfg["training"]["eval_interval"])
                * eval_seconds)) / 3600, 2),
        "validation_summary": summary,
        "config_hash": resolved["resolved"]["config_hash"],
        "env_hash": resolved["resolved"]["env_hash"],
        "code_hash": resolved["resolved"]["code_hash"],
    }
    out = Path(args.output) if args.output else BASE_DIR / "outputs" / \
        f"benchmark_{cfg['agent']['algorithm']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
