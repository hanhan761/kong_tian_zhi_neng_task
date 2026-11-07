"""训练器：采集/更新/验证/快照/最终测试的闭环，支持精确续训。"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from agents.continuous import AgentConfig, ContinuousAgent
from envs import make_env

from .checkpoint import (load_full_checkpoint, load_policy, read_latest_manifest,
                         save_full_checkpoint, save_policy, sha256_file)
from .collector import Collector
from .config import IMPLEMENTATION_DIR, output_dir, write_resolved
from .evaluation import evaluate_policy
from .metrics import (CsvLogger, JsonlLogger, convergence_step, read_csv,
                      summarize_episodes, write_json_atomic)
from .replay import ReplayBuffer
from .seeding import make_rng, set_global_seed

TRAIN_COLUMNS = ["t_env", "gradient_steps", "actor_updates", "noise_std", "critic_loss",
                 "actor_loss", "q_mean", "target_mean", "block_env_steps", "wall_seconds",
                 "samples_per_second"]
EVAL_COLUMNS = ["checkpoint_id", "t_env", "wall_seconds", "n", "success_rate",
                "collision_rate", "timeout_rate", "mean_return", "std_return",
                "mean_raw_base_return", "mean_path_length_all", "mean_path_length_success",
                "mean_elapsed_all", "mean_elapsed_success", "mean_energy",
                "mean_risk_integral", "mean_min_clearance"]


def _selection_key(success_rate, collision_rate, mean_return, t_env):
    return (float(success_rate if success_rate is not None else -1.0),
            -float(collision_rate if collision_rate is not None else 1.0),
            float(mean_return if mean_return is not None else -1e9),
            -int(t_env))


class Trainer:
    def __init__(self, resolved: dict, *, base_dir: Path = IMPLEMENTATION_DIR,
                 resume: bool = True, test_episodes: int | None = None,
                 run_final_test: bool = True, keep_training_points: bool = False,
                 stop_after_env_steps: int | None = None, log=print):
        self.cfg = resolved
        self.base_dir = Path(base_dir)
        self.dir = output_dir(resolved, self.base_dir)
        self.ckpt_dir = self.dir / "checkpoints"
        self.resume = bool(resume)
        self.test_episodes = test_episodes
        self.run_final_test = bool(run_final_test)
        self.keep_training_points = bool(keep_training_points)
        self.stop_after = None if stop_after_env_steps is None else int(stop_after_env_steps)
        self.log = log
        self.hashes = {
            "config_hash": resolved["resolved"]["config_hash"],
            "env_hash": resolved["resolved"]["env_hash"],
            "code_hash": resolved["resolved"]["code_hash"],
        }
        self.seed = int(resolved["training"]["seed"])
        self.gamma = float(resolved["agent"]["gamma"])
        self.max_env_steps = int(resolved["training"]["max_env_steps"])
        self.eval_interval = int(resolved["training"]["eval_interval"])
        self.collect_steps = int(resolved["training"]["collect_steps"])
        self.gradient_per_block = int(resolved["training"]["gradient_steps_per_block"])
        self.sigma0 = float(resolved["training"]["exploration_noise"])
        self.snapshot_steps = sorted(set(int(v) for v in
                                         resolved["training"].get("policy_snapshot_steps", [])))
        self.agent = None
        self.env = None
        self.replay = None
        self.collector = None
        self.rngs = None
        self.state = None
        self._meta = {}

    # ------------------------------------------------------------------ 初始化
    def _make_agent(self) -> ContinuousAgent:
        agent = self.cfg["agent"]
        return ContinuousAgent(AgentConfig(
            algorithm=agent["algorithm"], obs_dim=agent["obs_dim"], action_dim=agent["action_dim"],
            hidden_dim=agent["hidden_dim"], actor_lr=agent["actor_lr"], critic_lr=agent["critic_lr"],
            gamma=agent["gamma"], tau=agent["tau"], policy_delay=agent["policy_delay"],
            target_noise=agent["target_noise"], noise_clip=agent["noise_clip"],
            device=agent["device"]))

    def initialize(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        resolved_path = self.dir / "config_resolved.json"
        latest = read_latest_manifest(self.ckpt_dir) if self.resume else None
        if resolved_path.exists():
            old = json.loads(resolved_path.read_text(encoding="utf-8"))
            old_meta = old.get("resolved", {})
            for key in ("config_hash", "env_hash"):
                if old_meta.get(key) and old_meta[key] != self.hashes[key]:
                    raise RuntimeError(
                        f"refusing to reuse {self.dir}: {key} changed "
                        f"({old_meta[key][:8]} -> {self.hashes[key][:8]})")
            if not self.resume and not latest:
                raise RuntimeError(
                    f"output directory already contains a run: {self.dir}; "
                    "use --resume or a new experiment id")
        write_resolved(self.cfg, resolved_path)

        set_global_seed(self.seed, int(self.cfg["training"]["torch_num_threads"]))
        self.env = make_env(self.cfg["env"], render_mode=None, shaping_gamma=self.gamma)
        self.agent = self._make_agent()
        self.replay = ReplayBuffer(int(self.cfg["training"]["buffer_capacity"]),
                                   int(self.cfg["agent"]["obs_dim"]),
                                   int(self.cfg["agent"]["action_dim"]),
                                   seed=self.seed * 7919 + 17)
        self.rngs = {
            "exploration": make_rng(self.seed, 1),
            "seed": make_rng(self.seed, 2),
            "noise": make_rng(self.seed, 3),
            "eval": make_rng(self.seed, 4),
        }
        self.collector = Collector(
            self.env, self.agent, self.replay,
            exploration_rng=self.rngs["exploration"], seed_rng=self.rngs["seed"],
            noise_decay=float(self.cfg["training"]["noise_decay_per_eval"]),
            eval_interval=self.eval_interval, keep_points=self.keep_training_points,
            start_random_steps=int(self.cfg["training"]["start_random_steps"]))

        if latest is None:
            self.collector.start(self.seed)
            self.state = {
                "t_env": 0,
                "next_eval": self.eval_interval,
                "next_checkpoint": int(self.cfg["training"]["full_checkpoint_interval"]),
                "episode_count": 0,
                "wall_seconds": 0.0,
                "eval_count": 0,
                "best_validation": None,
                "best_reward": None,
                "replayed_env_steps": 0,
                "last_eval_t_env": None,
                "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            self.log(f"[{self.cfg['resolved']['run_id']}] 新训练：seed={self.seed}, "
                     f"max_env_steps={self.max_env_steps}, algo={self.cfg['agent']['algorithm']}")
        else:
            if latest["config_hash"] != self.hashes["config_hash"] or \
                    latest["env_hash"] != self.hashes["env_hash"]:
                raise RuntimeError("latest checkpoint hash does not match resolved config; refuse resume")
            self.state = load_full_checkpoint(
                self.ckpt_dir, agent=self.agent, replay=self.replay,
                collector=self.collector, env=self.env, rngs=self.rngs,
                expected_hashes=self.hashes)
            self.state["t_env"] = int(self.collector.t_env)
            self.log(f"[{self.cfg['resolved']['run_id']}] 从 {latest['directory']} 恢复："
                     f"t_env={self.state['t_env']}")

        self.train_log = CsvLogger(self.dir / "train_metrics.csv", TRAIN_COLUMNS)
        self.eval_log = CsvLogger(self.dir / "eval_metrics.csv", EVAL_COLUMNS)
        self.episode_log = JsonlLogger(self.dir / "episodes.jsonl")
        self.snapshot_dir = self.dir / "policy_snapshots"
        self.snapshot_manifest_path = self.snapshot_dir / "manifest.json"
        self.snapshot_manifest = self._load_snapshot_manifest()
        self._snapshots_done = {int(e["requested_t_env"]) for e in
                                self.snapshot_manifest.get("entries", [])
                                if e.get("requested_t_env") is not None}
        self._write_run_metadata(started=self.state.get("run_started_at"))

    # ------------------------------------------------------------------ 元数据
    def _machine_info(self) -> dict:
        import os
        info = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(self.agent.device) if self.agent is not None else self.cfg["agent"]["device"],
            "torch_num_threads": int(torch.get_num_threads()),
            "cpu_count": os.cpu_count(),
        }
        try:
            import psutil  # 课程依赖外的可选包；缺失时退回 /proc 信息
            info["rss_mb"] = round(psutil.Process().memory_info().rss / 2 ** 20, 1)
        except Exception:
            info["rss_mb"] = None
        return info

    def _write_run_metadata(self, started=None, finished=None, status="running") -> None:
        meta = {
            "schema_version": 1,
            "run_id": self.cfg["resolved"]["run_id"],
            "experiment_id": self.cfg["experiment_id"],
            "seed": self.seed,
            "algorithm": self.cfg["agent"]["algorithm"],
            "started_at": started or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_at": finished,
            "status": status,
            "machine": self._machine_info(),
            "env_hash": self.hashes["env_hash"],
            "config_hash": self.hashes["config_hash"],
            "code_hash": self.hashes["code_hash"],
            "shaping_gamma": self.gamma,
            "num_parameters": self.agent.num_parameters(),
            "env_spec": self.cfg["env"],
            "agent_config": self.cfg["agent"],
            "training": self.cfg["training"],
            "python_executable": __import__("sys").executable,
        }
        write_json_atomic(self.dir / "run_metadata.json", meta)
        self._meta = meta

    def _write_status(self, status: str, extra: dict | None = None) -> None:
        t_env = int(self.collector.t_env) if self.collector is not None else int(self.state["t_env"])
        self.state["t_env"] = t_env
        payload = {
            "status": status,
            "t_env": t_env,
            "max_env_steps": self.max_env_steps,
            "gradient_steps": int(self.agent.gradient_steps),
            "actor_updates": int(self.agent.actor_updates),
            "wall_seconds": float(self.state["wall_seconds"]),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if extra:
            payload.update(extra)
        write_json_atomic(self.dir / "status.json", payload)

    # ------------------------------------------------------------------ 回调
    def _on_episode(self, record: dict) -> None:
        self.state["episode_count"] += 1
        self.episode_log.write(record)

    def _write_train_row(self, *, block_steps: int, block_seconds: float,
                         losses: list[dict], sigma: float) -> None:
        actor = [m["actor_loss"] for m in losses if m["actor_loss"] is not None]
        row = {
            "t_env": int(self.collector.t_env),
            "gradient_steps": int(self.agent.gradient_steps),
            "actor_updates": int(self.agent.actor_updates),
            "noise_std": float(sigma),
            "critic_loss": float(np.mean([m["critic_loss"] for m in losses])) if losses else None,
            "actor_loss": float(np.mean(actor)) if actor else None,
            "q_mean": float(np.mean([m["q_mean"] for m in losses])) if losses else None,
            "target_mean": float(np.mean([m["target_mean"] for m in losses])) if losses else None,
            "block_env_steps": int(block_steps),
            "wall_seconds": float(self.state["wall_seconds"]),
            "samples_per_second": float(block_steps / block_seconds) if block_seconds > 0 else None,
        }
        self.train_log.write(row)

    # ------------------------------------------------------------------ 评估
    def evaluate_now(self, checkpoint_id: str) -> dict:
        records, summary = evaluate_policy(
            self.agent, self.cfg["env"], self.cfg["evaluation"]["validation_seeds"],
            gamma=self.gamma, phase="validation")
        row = {"checkpoint_id": checkpoint_id, "t_env": int(self.collector.t_env),
               "wall_seconds": float(self.state["wall_seconds"]), **summary}
        self.eval_log.write(row)
        self.state["eval_count"] += 1
        self.state["last_eval_t_env"] = int(self.collector.t_env)

        batch_dir = self.dir / "eval_episodes"
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / f"{checkpoint_id}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

        key = _selection_key(summary["success_rate"], summary["collision_rate"],
                             summary["mean_return"], row["t_env"])
        best = self.state.get("best_validation")
        if best is None or key > _selection_key(best["success_rate"], best["collision_rate"],
                                                best["mean_return"], best["t_env"]):
            self.state["best_validation"] = {**summary, "t_env": row["t_env"], "key": list(key)}
            save_policy(self.ckpt_dir / "best_validation.pth", self.agent,
                        env_spec=self.cfg["env"], t_env=row["t_env"], metrics=summary,
                        config_hash=self.hashes["config_hash"], env_hash=self.hashes["env_hash"],
                        code_hash=self.hashes["code_hash"], kind="best_validation")
            self.log(f"  [best_validation] t={row['t_env']} success={summary['success_rate']:.2f} "
                     f"return={summary['mean_return']:.2f}")

        best_reward = self.state.get("best_reward")
        if best_reward is None or (summary["mean_return"] or -1e9) > (best_reward["mean_return"] if best_reward else -1e9):
            self.state["best_reward"] = {**summary, "t_env": row["t_env"]}
            save_policy(self.ckpt_dir / "best_reward.pth", self.agent,
                        env_spec=self.cfg["env"], t_env=row["t_env"], metrics=summary,
                        config_hash=self.hashes["config_hash"], env_hash=self.hashes["env_hash"],
                        code_hash=self.hashes["code_hash"], kind="best_reward")

        self.log(f"  eval t={row['t_env']:>8d} | success={summary['success_rate']:.2f} "
                 f"collide={summary['collision_rate']:.2f} timeout={summary['timeout_rate']:.2f} "
                 f"return={summary['mean_return']:7.2f}")
        return row

    def _save_checkpoint(self) -> None:
        self.state["t_env"] = int(self.collector.t_env)
        save_full_checkpoint(
            self.ckpt_dir, step=int(self.collector.t_env), agent=self.agent, replay=self.replay,
            collector=self.collector, env=self.env, trainer_state=self.state, rngs=self.rngs,
            config_hash=self.hashes["config_hash"], env_hash=self.hashes["env_hash"],
            code_hash=self.hashes["code_hash"],
            keep_last=int(self.cfg["training"]["keep_last_full_checkpoints"]))
        self.log(f"  checkpoint saved at t={self.collector.t_env}")

    # ------------------------------------------------------------------ 中途策略快照
    def _load_snapshot_manifest(self) -> dict:
        if self.snapshot_manifest_path.exists():
            return json.loads(self.snapshot_manifest_path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "entries": []}

    def _write_snapshot_manifest(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json_atomic(self.snapshot_manifest_path, self.snapshot_manifest)

    def _save_policy_snapshot(self, requested: int, actual: int) -> None:
        if requested in self._snapshots_done:
            return
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if actual != requested:
            self.snapshot_manifest["entries"].append({
                "requested_t_env": int(requested),
                "actual_t_env": int(actual),
                "skipped": True,
                "reason": "threshold crossed before block-aligned snapshot (resume/stop overshoot); "
                          "no policy renamed to the requested step",
            })
            self._snapshots_done.add(int(requested))
            self._write_snapshot_manifest()
            self.log(f"  [snapshot skipped] requested={requested} actual={actual}")
            return
        path = self.snapshot_dir / f"policy_t{actual:09d}.pth"
        save_policy(path, self.agent, env_spec=self.cfg["env"], t_env=int(actual),
                    metrics={"best_validation": self.state.get("best_validation")},
                    config_hash=self.hashes["config_hash"], env_hash=self.hashes["env_hash"],
                    code_hash=self.hashes["code_hash"], kind="policy_snapshot_online",
                    extra={"requested_t_env": int(requested), "actual_t_env": int(actual),
                           "snapshot_role": "online_actor_at_block_boundary",
                           "exact_resume_note": "policy-only snapshot; full replay/optimizer/RNG "
                                                "checkpoint is still required for exact resume"})
        entry = {
            "requested_t_env": int(requested),
            "actual_t_env": int(actual),
            "path": path.relative_to(self.dir).as_posix(),
            "sha256": sha256_file(path),
            "kind": "policy_snapshot_online",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_hash": self.hashes["config_hash"],
            "env_hash": self.hashes["env_hash"],
            "code_hash": self.hashes["code_hash"],
            "skipped": False,
        }
        self.snapshot_manifest["entries"].append(entry)
        self._snapshots_done.add(int(requested))
        self._write_snapshot_manifest()
        self.log(f"  [policy snapshot] requested={requested} actual={actual} -> {path.name}")

    def _maybe_save_snapshots(self, t_env: int) -> None:
        for requested in self.snapshot_steps:
            if requested <= t_env and requested not in self._snapshots_done:
                self._save_policy_snapshot(requested, int(t_env))

    # ------------------------------------------------------------------ 主循环
    def run(self) -> dict:
        self.initialize()
        session_start = time.time()
        try:
            while self.collector.t_env < self.max_env_steps:
                block_start = time.time()
                current = int(self.collector.t_env)
                self._maybe_save_snapshots(current)
                n = min(self.collect_steps, self.max_env_steps - current)
                # 在快照阈值处切块，保证保存的 policy 以实际 t_env 命名，不能把 810k 标成 750k。
                upcoming = next((s for s in self.snapshot_steps if current < s <= current + n), None)
                if upcoming is not None:
                    n = min(n, upcoming - current)
                sigma = self.collector.noise_sigma(self.sigma0)
                self.collector.collect(n, self.sigma0, self._on_episode)
                t_env = self.collector.t_env
                self.state["t_env"] = int(t_env)

                desired = (t_env * self.gradient_per_block) // self.collect_steps
                pending = max(desired - self.agent.gradient_steps, 0)
                losses = []
                if self.replay.size >= max(int(self.cfg["training"]["batch_size"]),
                                           int(self.cfg["training"]["learning_starts"])):
                    for _ in range(pending):
                        losses.append(self.agent.update(
                            self.replay.sample(int(self.cfg["training"]["batch_size"]),
                                               self.agent.device)))
                block_seconds = time.time() - block_start
                self.state["wall_seconds"] += block_seconds

                self._write_train_row(block_steps=n, block_seconds=block_seconds,
                                      losses=losses, sigma=sigma)
                self._maybe_save_snapshots(t_env)

                if t_env >= self.state["next_eval"]:
                    self.evaluate_now(f"t{t_env}")
                    self.state["next_eval"] = t_env + self.eval_interval

                if t_env >= self.state["next_checkpoint"]:
                    self._save_checkpoint()
                    self.state["next_checkpoint"] = t_env + int(
                        self.cfg["training"]["full_checkpoint_interval"])

                if self.stop_after is not None and t_env >= self.stop_after:
                    self._save_checkpoint()
                    self._write_status("stopped")
                    self.log(f"[{self.cfg['resolved']['run_id']}] 在 t_env={t_env} 安全点停止（续训测试）")
                    return {"t_env": t_env, "status": "stopped",
                            "best_validation": self.state["best_validation"]}

            if self.state["last_eval_t_env"] != self.collector.t_env:
                self.evaluate_now("final")
            self._save_checkpoint()
            self._write_status("training_complete")

            test_summary = None
            if self.run_final_test:
                test_summary = self.final_test()
            self._write_run_metadata(started=self._meta.get("started_at"), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                     status="completed")
            self._write_status("completed", {"run_seconds": time.time() - session_start,
                                             "final_test": test_summary})
            self.log(f"[{self.cfg['resolved']['run_id']}] 完成：t_env={self.collector.t_env}, "
                     f"best_validation={self.state['best_validation']}")
            return {"t_env": self.collector.t_env, "best_validation": self.state["best_validation"],
                    "final_test": test_summary, "status": "completed"}
        except Exception as exc:
            self._write_status("failed", {"error": repr(exc)})
            self._write_run_metadata(started=self._meta.get("started_at"), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                     status="failed")
            raise
        finally:
            self.train_log.close()
            self.eval_log.close()
            self.episode_log.close()
            if self.env is not None:
                self.env.close()

    # ------------------------------------------------------------------ 测试
    def final_test(self) -> dict:
        policy_path = self.ckpt_dir / "best_validation.pth"
        if not policy_path.exists():
            raise RuntimeError("best_validation policy is missing; cannot run final test")
        agent, payload = load_policy(policy_path, device=str(self.agent.device))
        seeds = list(self.cfg["evaluation"]["test_seeds"])
        if self.test_episodes is not None:
            seeds = seeds[: max(int(self.test_episodes), 1)]
        records, summary = evaluate_policy(agent, self.cfg["env"], seeds, gamma=self.gamma,
                                           phase="test", keep_points=True)
        out = self.dir / "final_test_episodes.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps({k: v for k, v in record.items() if k != "trajectory"},
                                    ensure_ascii=False) + "\n")
        traj_dir = self.dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            (traj_dir / f"test_seed_{record['seed']}.json").write_text(
                json.dumps({"seed": record["seed"],
                            "success": record["success"],
                            "collision": record["collision"],
                            "timeout": record["timeout"],
                            "trajectory": record["trajectory"]}, ensure_ascii=False), encoding="utf-8")
        write_json_atomic(self.dir / "final_test_summary.json", {
            "model": "checkpoints/best_validation.pth",
            "model_t_env": payload["t_env"],
            "model_metrics": payload["metrics"],
            "n_episodes": len(seeds),
            **summary,
        })
        self.log(f"  final test on {len(seeds)} seeds: success={summary['success_rate']:.3f} "
                 f"collision={summary['collision_rate']:.3f} return={summary['mean_return']:.2f}")
        return {"n_episodes": len(seeds), **summary}

    # ------------------------------------------------------------------ 汇总
    def summary(self) -> dict:
        eval_rows = read_csv(self.dir / "eval_metrics.csv")
        conv = convergence_step(eval_rows,
                                float(self.cfg["evaluation"]["convergence_return_threshold"]),
                                float(self.cfg["evaluation"]["convergence_success_threshold"]),
                                int(self.cfg["evaluation"]["convergence_consecutive_checkpoints"]))
        test_path = self.dir / "final_test_summary.json"
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
        return {
            "run_id": self.cfg["resolved"]["run_id"],
            "experiment_id": self.cfg["experiment_id"],
            "seed": self.seed,
            "algorithm": self.cfg["agent"]["algorithm"],
            "t_env": int(eval_rows[-1]["t_env"]) if eval_rows else 0,
            "convergence_t_env": conv,
            "converged": conv is not None,
            "best_validation": self.state.get("best_validation") if self.state else None,
            "final_test": test,
            "eval_rows": len(eval_rows),
            "env_hash": self.hashes["env_hash"],
            "config_hash": self.hashes["config_hash"],
            "code_hash": self.hashes["code_hash"],
        }
