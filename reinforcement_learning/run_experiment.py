"""单实验训练入口。

示例：
    python run_experiment.py --config configs/ddpg_baseline.json --seed 0 \
        --max-env-steps 30000 --experiment-id smoke_ddpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import torch  # noqa: E402

from common.config import load_config, resolve_config  # noqa: E402
from common.training import Trainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DDPG/TD3 训练入口（课程航路规划实验06）")
    parser.add_argument("--config", required=True, help="JSON 配置路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 training.seed")
    parser.add_argument("--max-env-steps", type=int, default=None, help="覆盖训练预算")
    parser.add_argument("--experiment-id", default=None, help="覆盖 experiment_id")
    parser.add_argument("--output-root", default=None, help="覆盖 artifacts.output_root")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="覆盖 agent.device")
    parser.add_argument("--test-episodes", type=int, default=None,
                        help="最终测试回合数（默认使用配置中的测试 seed bank 全量）")
    parser.add_argument("--no-test", action="store_true", help="跳过最终测试（仅烟测使用）")
    parser.add_argument("--no-resume", action="store_true", help="禁止从已有快照续训")
    parser.add_argument("--keep-training-points", action="store_true",
                        help="训练回合也保存轨迹点（默认只保存测试/视频轨迹）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(Path(args.config))
        if args.device:
            cfg["agent"]["device"] = args.device
        if cfg["agent"]["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
        resolved = resolve_config(cfg, seed=args.seed, max_env_steps=args.max_env_steps,
                                  experiment_id=args.experiment_id,
                                  output_root=args.output_root, base_dir=BASE_DIR)
        trainer = Trainer(resolved, base_dir=BASE_DIR, resume=not args.no_resume,
                          test_episodes=args.test_episodes,
                          run_final_test=not args.no_test,
                          keep_training_points=args.keep_training_points)
        trainer.run()
        summary = trainer.summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:  # CLI 失败返回非零退出码
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
