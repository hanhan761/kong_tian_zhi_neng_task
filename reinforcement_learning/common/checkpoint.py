"""完整训练快照与轻量 policy 导出（原子写入、哈希校验）。"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from agents.continuous import ContinuousAgent

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        fh.flush()


def save_full_checkpoint(directory: Path, *, step: int, agent: ContinuousAgent,
                         replay, collector, env, trainer_state: dict,
                         rngs: dict[str, np.random.Generator],
                         config_hash: str, env_hash: str, code_hash: str,
                         keep_last: int = 2) -> Path:
    """在同卷临时目录写完、校验后原子 rename，最后原子替换 latest_manifest.json。"""
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"step_{step:09d}"
    tmp = directory / f".tmp_{final.name}_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        torch.save(agent.checkpoint_state(), tmp / "agent.pt")
        replay_state = replay.state_dict()
        np.savez(tmp / "replay.npz", **replay_state["arrays"])
        _write_json(tmp / "replay_meta.json", {
            "capacity": replay_state["capacity"], "pos": replay_state["pos"],
            "size": replay_state["size"], "obs_dim": replay_state["obs_dim"],
            "action_dim": replay_state["action_dim"],
            "rng": _json_safe(replay_state["rng"]),
        })
        _write_json(tmp / "env_state.json", _json_safe(env.get_state()))
        _write_json(tmp / "collector_state.json", _json_safe(collector.state_dict()))
        _write_json(tmp / "trainer_state.json", _json_safe(trainer_state))

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        rng_payload = {
            "schema_version": SCHEMA_VERSION,
            "torch": np.frombuffer(torch.get_rng_state().numpy().tobytes(), dtype=np.uint8),
            "cuda": [np.frombuffer(s.cpu().numpy().tobytes(), dtype=np.uint8)
                     for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "python_version": python_state[0],
            "python_internal": np.asarray(python_state[1], dtype=np.uint32),
            "python_gauss": python_state[2],
            "numpy_key": numpy_state[1].astype(np.uint32),
            "numpy_pos": np.array([numpy_state[2]], dtype=np.int64),
            "numpy_has_gauss": np.array([numpy_state[3]], dtype=np.int64),
            "numpy_cached": np.array([numpy_state[4]], dtype=np.float64),
            "exploration": _json_safe(rngs["exploration"].bit_generator.state),
            "seed_rng": _json_safe(rngs["seed"].bit_generator.state),
            "noise_rng": _json_safe(rngs["noise"].bit_generator.state),
            "eval_rng": _json_safe(rngs["eval"].bit_generator.state),
        }
        np.savez(tmp / "rng.npz", **{k: v for k, v in rng_payload.items()
                                     if isinstance(v, np.ndarray)})
        _write_json(tmp / "rng_meta.json", {k: _json_safe(v) for k, v in rng_payload.items()
                                            if not isinstance(v, np.ndarray)})

        hashes = {p.name: sha256_file(p) for p in sorted(tmp.glob("*")) if p.is_file()}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "step": int(step),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_hash": config_hash,
            "env_hash": env_hash,
            "code_hash": code_hash,
            "files": hashes,
        }
        _write_json(tmp / "manifest.json", manifest)
        # 读回校验后再原子改名
        for name, expected in hashes.items():
            if sha256_file(tmp / name) != expected:
                raise IOError(f"checkpoint write verification failed for {name}")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    latest = {
        "schema_version": SCHEMA_VERSION,
        "step": int(step),
        "directory": final.name,
        "manifest_sha256": sha256_file(final / "manifest.json"),
        "config_hash": config_hash,
        "env_hash": env_hash,
        "code_hash": code_hash,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp_manifest = directory / ".latest_manifest.tmp"
    _write_json(tmp_manifest, latest)
    tmp_manifest.replace(directory / "latest_manifest.json")

    # 只保留最近 keep_last 份完整快照
    step_dirs = sorted(directory.glob("step_*"))
    for old in step_dirs[:-max(int(keep_last), 1)]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def read_latest_manifest(directory: Path) -> dict | None:
    path = directory / "latest_manifest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_checkpoint(directory: Path) -> dict:
    """校验 manifest 与全部文件哈希；损坏时抛错。"""
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest in {directory}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint file: {name}")
        if sha256_file(path) != expected:
            raise IOError(f"checkpoint hash mismatch: {name}")
    return manifest


def restore_agent_inplace(agent: ContinuousAgent, state: dict) -> None:
    for name in ("actor", "actor_target", "q1", "q1_target", "q2", "q2_target"):
        network = getattr(agent, name)
        if network is not None:
            network.load_state_dict(state[name])
    agent.actor_opt.load_state_dict(state["actor_optimizer"])
    agent.critic_opt.load_state_dict(state["critic_optimizer"])
    agent.gradient_steps = int(state["gradient_steps"])
    agent.actor_updates = int(state.get("actor_updates", 0))


def load_full_checkpoint(directory: Path, *, agent: ContinuousAgent, replay, collector,
                         env, rngs: dict[str, np.random.Generator],
                         expected_hashes: dict | None = None) -> dict:
    """恢复顺序：元数据 → 对象/数组 → 环境 → 累计项 → 最后恢复 RNG。"""
    dir_path = directory / read_latest_manifest(directory)["directory"] if (directory / "latest_manifest.json").exists() else directory
    manifest = verify_checkpoint(dir_path)
    if expected_hashes:
        for key in ("config_hash", "env_hash"):
            if key in expected_hashes and key in manifest and manifest[key] != expected_hashes[key]:
                raise ValueError(f"checkpoint {key} mismatch: refuses pseudo-resume")

    agent_state = torch.load(dir_path / "agent.pt", map_location=agent.device, weights_only=True)
    restore_agent_inplace(agent, agent_state)

    with np.load(dir_path / "replay.npz", allow_pickle=False) as arrays:
        replay_meta = json.loads((dir_path / "replay_meta.json").read_text(encoding="utf-8"))
        replay.load_state_dict({
            **replay_meta,
            "arrays": {name: arrays[name] for name in arrays.files},
        })

    collector_state = json.loads((dir_path / "collector_state.json").read_text(encoding="utf-8"))
    collector.load_state_dict(collector_state)
    env.set_state(json.loads((dir_path / "env_state.json").read_text(encoding="utf-8")))
    trainer_state = json.loads((dir_path / "trainer_state.json").read_text(encoding="utf-8"))

    # RNG 最后恢复
    rng_meta = json.loads((dir_path / "rng_meta.json").read_text(encoding="utf-8"))
    with np.load(dir_path / "rng.npz", allow_pickle=False) as rng_arrays:
        torch.set_rng_state(torch.from_numpy(rng_arrays["torch"].copy()))
        if torch.cuda.is_available() and len(rng_meta.get("cuda", [])) == torch.cuda.device_count():
            states = [torch.from_numpy(arr.copy()) for arr in rng_arrays["cuda"]]
            torch.cuda.set_rng_state_all(states)
        random.setstate((rng_meta["python_version"],
                         tuple(int(v) for v in rng_arrays["python_internal"].tolist()),
                         rng_meta["python_gauss"]))
        np.random.set_state(("MT19937", rng_arrays["numpy_key"].copy(),
                             int(rng_arrays["numpy_pos"][0]),
                             int(rng_arrays["numpy_has_gauss"][0]),
                             float(rng_arrays["numpy_cached"][0])))
    rngs["exploration"].bit_generator.state = rng_meta["exploration"]
    rngs["seed"].bit_generator.state = rng_meta["seed_rng"]
    rngs["noise"].bit_generator.state = rng_meta["noise_rng"]
    rngs["eval"].bit_generator.state = rng_meta["eval_rng"]
    return trainer_state


def save_policy(path: Path, agent: ContinuousAgent, *, env_spec: dict, t_env: int,
                metrics: dict, config_hash: str, env_hash: str, code_hash: str,
                kind: str, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "t_env": int(t_env),
        "metrics": metrics,
        "config_hash": config_hash,
        "env_hash": env_hash,
        "code_hash": code_hash,
        "env_spec": env_spec,
        "policy": agent.policy_state(),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_policy(path: Path, device: str = "cpu") -> tuple[ContinuousAgent, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    agent = ContinuousAgent.from_policy_state(payload["policy"], device=device)
    return agent, payload
