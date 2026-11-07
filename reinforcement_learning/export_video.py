"""固定 seed 视频导出：rgb_array 流式写帧，事后解码校验帧数与尺寸。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from common.checkpoint import load_policy  # noqa: E402
from envs import make_env  # noqa: E402


def validate_video(path: Path) -> dict:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        frames = 0
        first_shape = None
        for frame in reader:
            if frames == 0:
                first_shape = tuple(np.asarray(frame).shape)
            frames += 1
    finally:
        reader.close()
    fps = float(meta.get("fps", 0) or 0)
    duration = frames / fps if fps else 0.0
    if frames <= 1:
        raise RuntimeError(f"video {path} has only {frames} frame(s)")
    if not first_shape or first_shape[-1] != 3:
        raise RuntimeError(f"video {path} frame shape is not HxWx3: {first_shape}")
    return {"frames": frames, "fps": fps, "duration_s": round(duration, 2),
            "frame_shape": list(first_shape), "size_bytes": path.stat().st_size}


def export_video(checkpoint: Path, seeds: list[int], out_dir: Path, fps: int = 10,
                 filename_prefix: str = "episode") -> list[dict]:
    agent, payload = load_policy(checkpoint)
    env_spec = payload["env_spec"]
    gamma = float(payload["policy"]["config"]["gamma"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        env = make_env(env_spec, render_mode="rgb_array", shaping_gamma=gamma)
        path = out_dir / f"{filename_prefix}_seed_{seed}.mp4"
        writer = imageio.get_writer(path, fps=int(fps), macro_block_size=None, quality=8)
        obs, info = env.reset(seed=int(seed))
        terminated = truncated = False
        frames = 0
        try:
            frame = env.render()
            writer.append_data(frame)
            frames += 1
            while not (terminated or truncated):
                action = agent.act(obs, 0.0, None)
                obs, _reward, terminated, truncated, info = env.step(action)
                writer.append_data(env.render())
                frames += 1
        finally:
            writer.close()
            env.close()
        info_record = {
            "seed": int(seed),
            "success": bool(info["success"]),
            "collision": bool(info["collision"]),
            "timeout": bool(info["timeout"]),
            "steps": int(info["episode_step"]),
            "elapsed_sim_time": float(info["elapsed_sim_time"]),
            "playback_speedup": round(float(fps) * 0.25, 2),
        }
        info_record.update(validate_video(path))
        info_record["file"] = str(path)
        results.append(info_record)
        print(f"video seed {seed}: {'SUCCESS' if info_record['success'] else 'FAIL'} "
              f"{info_record['frames']} frames")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出固定 seed 测试视频")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="episode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checkpoint = Path(args.checkpoint)
        out_dir = Path(args.output_dir) if args.output_dir else \
            checkpoint.parent.parent / "videos"
        results = export_video(checkpoint, args.seeds, out_dir, args.fps, args.prefix)
        (out_dir / "video_manifest.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
