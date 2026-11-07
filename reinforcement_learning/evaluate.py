"""独立评估入口：只在冻结配置与选定模型上运行固定 seed bank。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from common.checkpoint import load_policy  # noqa: E402
from common.evaluation import evaluate_policy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定 seed bank 评估")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-bank", choices=["validation", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config", default=None, help="可选：用配置中的 seed bank 覆盖")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        agent, payload = load_policy(Path(args.checkpoint))
        env_spec = payload["env_spec"]
        gamma = float(payload["policy"]["config"]["gamma"])
        if args.config:
            with open(args.config, "r", encoding="utf-8") as fh:
                seeds = json.load(fh)["evaluation"][f"{args.seed_bank}_seeds"]
        else:
            base = BASE_DIR / "configs" / "ddpg_baseline.json"
            with open(base, "r", encoding="utf-8") as fh:
                seeds = json.load(fh)["evaluation"][f"{args.seed_bank}_seeds"]
        if args.episodes:
            seeds = seeds[: args.episodes]
        records, summary = evaluate_policy(agent, env_spec, seeds, gamma=gamma, phase=args.seed_bank)
        out = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".eval.json")
        payload_out = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_t_env": payload["t_env"],
            "seed_bank": args.seed_bank,
            "n_episodes": len(seeds),
            **summary,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload_out, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
