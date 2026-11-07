"""配置加载、严格 schema 校验与 resolved 配置生成。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent

AGENT_FIELDS = {"algorithm", "obs_dim", "action_dim", "hidden_dim", "actor_lr", "critic_lr",
                "gamma", "tau", "policy_delay", "target_noise", "noise_clip", "device"}
ENV_FIELDS = {"id", "spec_version", "mass", "length", "width", "inertia", "dt",
              "physics_substeps", "speed_limit", "angular_speed_limit", "map_bounds",
              "bottom_exit_open", "exit_half_width", "collision_touch_counts",
              "collision_epsilon", "collision_method", "max_episode_steps", "time_limit_owner",
              "time_limit_is_truncation", "observation_normalization", "obstacles",
              "initial_state", "reward", "render_mode", "risk", "potential"}
INITIAL_FIELDS = {"abs_x_range", "left_probability", "y_range", "theta_range",
                  "v_normal_mean", "v_normal_std", "omega_normal_mean", "omega_normal_std",
                  "reject_collision", "max_attempts"}
REWARD_FIELDS = {"center_coef", "center_exponent", "exit_coef", "exit_exponent",
                 "exit_y_shift", "time_cost", "omega_cost", "action_squared_cost",
                 "success_bonus", "collision_penalty"}
RISK_FIELDS = {"centers", "radius", "aggregation", "kappa"}
POTENTIAL_FIELDS = {"kind", "grid_resolution", "inflation_radius", "goal", "dmax", "beta",
                    "gamma_source", "connectivity", "forbid_diagonal_corner_cutting",
                    "true_terminal_phi", "truncate_does_not_zero_phi", "fallback_policy"}
TRAINING_FIELDS = {"seed", "max_env_steps", "collect_steps", "gradient_steps_per_block",
                   "learning_starts", "start_random_steps", "buffer_capacity", "batch_size",
                   "exploration_noise", "noise_decay_per_eval", "eval_interval",
                   "full_checkpoint_interval", "keep_last_full_checkpoints", "early_stop",
                   "torch_num_threads", "exact_resume_required", "policy_snapshot_steps"}
EVAL_FIELDS = {"validation_seeds", "test_seeds", "video_seeds",
               "convergence_return_threshold", "convergence_success_threshold",
               "convergence_consecutive_checkpoints", "checkpoint_selection",
               "test_only_after_config_freeze"}
ARTIFACT_FIELDS = {"output_root", "video_fps", "save_raw_metrics", "save_trajectories"}
TOP_FIELDS = {"schema_version", "experiment_id", "experiment_role", "budget_profile",
              "template_only", "agent", "env", "training", "evaluation", "artifacts",
              "deferred_selection", "resolved", "deferred_selection_applied",
              "selection_source"}
SELECTION_ORDER = ["success_rate_desc", "collision_rate_asc", "mean_return_desc", "t_env_asc"]


class ConfigError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_keys(block: dict, allowed: set[str], name: str) -> None:
    unknown = set(block) - allowed
    if unknown:
        raise ConfigError(f"unknown fields in {name}: {sorted(unknown)}")


def _number(value: Any, name: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}")
    return float(value)


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _range(value: Any, name: str) -> tuple[float, float]:
    _require(isinstance(value, (list, tuple)) and len(value) == 2, f"{name} must be [lo, hi]")
    lo, hi = _number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")
    _require(lo < hi, f"{name} requires lo < hi")
    return lo, hi


def validate_config(cfg: dict) -> None:
    """严格校验；未知字段和非法数值直接报错，不静默忽略。"""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a JSON object")
    _check_keys(cfg, TOP_FIELDS, "config")
    for key in ("schema_version", "experiment_id", "agent", "env", "training",
                "evaluation", "artifacts"):
        _require(key in cfg, f"missing required field: {key}")
    _require(int(cfg["schema_version"]) == 1, "only schema_version=1 is supported")
    _require(isinstance(cfg["experiment_id"], str) and cfg["experiment_id"], "experiment_id required")

    agent = cfg["agent"]
    _require(isinstance(agent, dict), "agent must be an object")
    _check_keys(agent, AGENT_FIELDS, "agent")
    _require(agent["algorithm"] in ("ddpg", "td3"), "agent.algorithm must be ddpg or td3")
    _integer(agent["obs_dim"], "agent.obs_dim", 1)
    _integer(agent["action_dim"], "agent.action_dim", 1)
    _integer(agent["hidden_dim"], "agent.hidden_dim", 1)
    _number(agent["actor_lr"], "agent.actor_lr", 1e-9)
    _number(agent["critic_lr"], "agent.critic_lr", 1e-9)
    _number(agent["gamma"], "agent.gamma", 1e-6, 1 - 1e-6)
    _number(agent["tau"], "agent.tau", 1e-9, 1.0)
    _integer(agent["policy_delay"], "agent.policy_delay", 1)
    _number(agent["target_noise"], "agent.target_noise", 0.0)
    _number(agent["noise_clip"], "agent.noise_clip", 0.0)
    _require(isinstance(agent["device"], str) and agent["device"] in ("cpu", "cuda"),
             "agent.device must be cpu or cuda")

    env = cfg["env"]
    _require(isinstance(env, dict), "env must be an object")
    _check_keys(env, ENV_FIELDS, "env")
    _require(env["id"] in ("CourseWhRobot-v1", "BattlefieldNavigation-v0"), "unknown env id")
    _integer(env["spec_version"], "env.spec_version", 0)
    _number(env["mass"], "env.mass", 1e-6)
    _number(env["length"], "env.length", 1e-6)
    _number(env["width"], "env.width", 1e-6)
    _number(env["inertia"], "env.inertia", 1e-6)
    _number(env["dt"], "env.dt", 1e-9)
    _integer(env["physics_substeps"], "env.physics_substeps", 1)
    _number(env["speed_limit"], "env.speed_limit", 1e-6)
    _number(env["angular_speed_limit"], "env.angular_speed_limit", 1e-6)
    _require(isinstance(env["map_bounds"], list) and len(env["map_bounds"]) == 4,
             "env.map_bounds must be [xmin, xmax, ymin, ymax]")
    xmin, xmax, ymin, ymax = [_number(v, "env.map_bounds") for v in env["map_bounds"]]
    _require(xmin < xmax and ymin < ymax, "env.map_bounds requires min < max")
    _bool(env["bottom_exit_open"], "env.bottom_exit_open")
    _require(env["bottom_exit_open"] is True, "bottom central gap must stay open")
    _number(env["exit_half_width"], "env.exit_half_width", 1e-6)
    _bool(env["collision_touch_counts"], "env.collision_touch_counts")
    _number(env["collision_epsilon"], "env.collision_epsilon", 0.0)
    _require(env["collision_method"] == "convex_polygon_SAT",
             "only convex_polygon_SAT is implemented")
    _integer(env["max_episode_steps"], "env.max_episode_steps", 1)
    _require(env["time_limit_owner"] == "environment", "time limit must be owned by environment")
    _bool(env["time_limit_is_truncation"], "env.time_limit_is_truncation")
    _require(env["time_limit_is_truncation"] is True, "200-step cap must be truncation")
    _bool(env["observation_normalization"], "env.observation_normalization")
    _require(env["observation_normalization"] is False, "baseline forbids observation normalization")
    _require(isinstance(env["obstacles"], list) and env["obstacles"], "env.obstacles required")
    for i, row in enumerate(env["obstacles"]):
        _require(isinstance(row, list) and len(row) == 4,
                 f"env.obstacles[{i}] must be [xmin, xmax, ymin, ymax]")
        a, b, c, d = [_number(v, f"env.obstacles[{i}]") for v in row]
        _require(a < b and c < d, f"env.obstacles[{i}] requires min < max")
        _require(a >= xmin and b <= xmax and c >= ymin and d <= ymax,
                 f"env.obstacles[{i}] must lie inside map bounds")
    _require(env.get("render_mode") in (None, "rgb_array"), "env.render_mode must be null or rgb_array")

    init = env["initial_state"]
    _require(isinstance(init, dict), "env.initial_state must be an object")
    _check_keys(init, INITIAL_FIELDS, "env.initial_state")
    ax_lo, ax_hi = _range(init["abs_x_range"], "env.initial_state.abs_x_range")
    y_lo, y_hi = _range(init["y_range"], "env.initial_state.y_range")
    th_lo, th_hi = _range(init["theta_range"], "env.initial_state.theta_range")
    _require(all(v >= 0 for v in (ax_lo, ax_hi)), "abs_x_range must be nonnegative")
    _require(ax_hi <= xmax and y_hi <= ymax and y_lo >= ymin,
             "initial ranges must stay inside map")
    _require(th_lo >= -math.pi and th_hi <= math.pi, "theta_range must lie in [-pi, pi]")
    _number(init["left_probability"], "env.initial_state.left_probability", 0.0, 1.0)
    _number(init["v_normal_std"], "env.initial_state.v_normal_std", 0.0)
    _number(init["omega_normal_std"], "env.initial_state.omega_normal_std", 0.0)
    _bool(init["reject_collision"], "env.initial_state.reject_collision")
    _require(init["reject_collision"] is True, "initial collision rejection is mandatory")
    _integer(init["max_attempts"], "env.initial_state.max_attempts", 1)

    reward = env["reward"]
    _check_keys(reward, REWARD_FIELDS, "env.reward")
    for key in REWARD_FIELDS:
        _number(reward[key], f"env.reward.{key}")

    if "risk" in env:
        risk = env["risk"]
        _check_keys(risk, RISK_FIELDS, "env.risk")
        _require(isinstance(risk["centers"], list) and risk["centers"], "env.risk.centers required")
        for i, center in enumerate(risk["centers"]):
            _require(isinstance(center, list) and len(center) == 2, "risk center must be [x, y]")
            _number(center[0], f"env.risk.centers[{i}]")
            _number(center[1], f"env.risk.centers[{i}]")
        _number(risk["radius"], "env.risk.radius", 1e-6)
        _require(risk["aggregation"] == "max_gaussian", "risk aggregation must be max_gaussian")
        _number(risk["kappa"], "env.risk.kappa", 0.0)

    if "potential" in env:
        pot = env["potential"]
        _check_keys(pot, POTENTIAL_FIELDS, "env.potential")
        _require(pot["kind"] == "reverse_dijkstra_distance", "potential kind unsupported")
        _number(pot["grid_resolution"], "env.potential.grid_resolution", 1e-6)
        _number(pot["inflation_radius"], "env.potential.inflation_radius", 0.0)
        _require(isinstance(pot["goal"], list) and len(pot["goal"]) == 2, "goal must be [x, y]")
        _number(pot["goal"][0], "env.potential.goal[0]")
        _number(pot["goal"][1], "env.potential.goal[1]")
        _number(pot["dmax"], "env.potential.dmax", 1e-6)
        _number(pot["beta"], "env.potential.beta", 0.0)
        _require(pot["gamma_source"] == "agent.gamma", "potential gamma must come from agent.gamma")
        _require(int(pot["connectivity"]) in (4, 8), "potential connectivity must be 4 or 8")
        _bool(pot["forbid_diagonal_corner_cutting"], "potential forbid_diagonal_corner_cutting")
        _number(pot["true_terminal_phi"], "env.potential.true_terminal_phi")
        _bool(pot["truncate_does_not_zero_phi"], "potential truncate_does_not_zero_phi")
        _require(pot["truncate_does_not_zero_phi"] is True, "timeout must not zero the potential")

    tr = cfg["training"]
    _check_keys(tr, TRAINING_FIELDS, "training")
    _integer(tr["seed"], "training.seed", 0)
    _integer(tr["max_env_steps"], "training.max_env_steps", 1)
    _integer(tr["collect_steps"], "training.collect_steps", 1)
    _integer(tr["gradient_steps_per_block"], "training.gradient_steps_per_block", 0)
    _integer(tr["learning_starts"], "training.learning_starts", 0)
    _integer(tr["start_random_steps"], "training.start_random_steps", 0)
    _require(tr["start_random_steps"] == 0 or cfg.get("experiment_role") == "parameter_tuning_v1",
             "uniform random warmup is only allowed for parameter_tuning_v1")
    _integer(tr["buffer_capacity"], "training.buffer_capacity", 1)
    _integer(tr["batch_size"], "training.batch_size", 1)
    _number(tr["exploration_noise"], "training.exploration_noise", 0.0)
    _number(tr["noise_decay_per_eval"], "training.noise_decay_per_eval", 0.0, 1.0)
    _integer(tr["eval_interval"], "training.eval_interval", 1)
    _integer(tr["full_checkpoint_interval"], "training.full_checkpoint_interval", 1)
    _integer(tr["keep_last_full_checkpoints"], "training.keep_last_full_checkpoints", 1)
    _bool(tr["early_stop"], "training.early_stop")
    _integer(tr["torch_num_threads"], "training.torch_num_threads", 1)
    _bool(tr["exact_resume_required"], "training.exact_resume_required")
    snapshots = tr.get("policy_snapshot_steps", [])
    _require(isinstance(snapshots, list), "training.policy_snapshot_steps must be a list")
    previous = 0
    for i, step in enumerate(snapshots):
        step = _integer(step, f"training.policy_snapshot_steps[{i}]", 1)
        _require(step > previous, "training.policy_snapshot_steps must be strictly increasing")
        _require(step <= int(tr["max_env_steps"]),
                 "training.policy_snapshot_steps must not exceed max_env_steps")
        previous = step

    ev = cfg["evaluation"]
    _check_keys(ev, EVAL_FIELDS, "evaluation")
    for key in ("validation_seeds", "test_seeds", "video_seeds"):
        _require(isinstance(ev[key], list) and ev[key], f"evaluation.{key} required")
        for i, seed in enumerate(ev[key]):
            _integer(seed, f"evaluation.{key}[{i}]", 0)
    _require(len(set(ev["validation_seeds"])) == len(ev["validation_seeds"]),
             "validation seeds must be unique")
    _require(len(set(ev["test_seeds"])) == len(ev["test_seeds"]), "test seeds must be unique")
    _require(not set(ev["validation_seeds"]) & set(ev["test_seeds"]),
             "validation and test seed banks must be disjoint")
    _number(ev["convergence_return_threshold"], "evaluation.convergence_return_threshold")
    _number(ev["convergence_success_threshold"], "evaluation.convergence_success_threshold", 0.0, 1.0)
    _integer(ev["convergence_consecutive_checkpoints"],
             "evaluation.convergence_consecutive_checkpoints", 1)
    _require(ev["checkpoint_selection"] == SELECTION_ORDER,
             "checkpoint_selection must be the documented order")
    _bool(ev["test_only_after_config_freeze"], "evaluation.test_only_after_config_freeze")

    ar = cfg["artifacts"]
    _check_keys(ar, ARTIFACT_FIELDS, "artifacts")
    _require(isinstance(ar["output_root"], str) and ar["output_root"], "artifacts.output_root")
    _integer(ar["video_fps"], "artifacts.video_fps", 1)
    _bool(ar["save_raw_metrics"], "artifacts.save_raw_metrics")
    _bool(ar["save_trajectories"], "artifacts.save_trajectories")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def env_hash(env_spec: dict) -> str:
    return sha256_json(env_spec)


def code_hash(files: list[Path] | None = None) -> str:
    if files is None:
        files = sorted((IMPLEMENTATION_DIR / name) for name in (
            "envs/geometry.py", "envs/potential.py", "envs/course_whrobot.py",
            "envs/battlefield.py", "envs/__init__.py", "agents/continuous.py",
            "common/replay.py", "common/collector.py", "common/checkpoint.py",
            "common/evaluation.py", "common/training.py", "common/metrics.py",
            "common/config.py", "common/seeding.py"))
    digest = hashlib.sha256()
    for path in files:
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    digest.update((IMPLEMENTATION_DIR / "run_experiment.py").read_bytes()
                  if (IMPLEMENTATION_DIR / "run_experiment.py").exists() else b"")
    return digest.hexdigest()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    validate_config(cfg)
    return cfg


def apply_deferred_selection(cfg: dict, base_dir: Path | None = None) -> dict:
    """按 selection 文件解析战场配置；缺失时明确报错，不静默假定赢家。"""
    deferred = cfg.get("deferred_selection")
    if not deferred:
        return cfg
    base = Path(base_dir) if base_dir is not None else IMPLEMENTATION_DIR
    selection_path = Path(deferred["selection_file"])
    if not selection_path.is_absolute():
        selection_path = base / selection_path
    if not selection_path.exists():
        raise ConfigError(f"deferred selection file missing: {selection_path}")
    with open(selection_path, "r", encoding="utf-8") as fh:
        selection = json.load(fh)
    for field in deferred["copy_agent_fields"]:
        if field not in selection["agent"]:
            raise ConfigError(f"selection missing agent field {field}")
        cfg["agent"][field] = selection["agent"][field]
    for field in deferred["copy_training_fields"]:
        if field not in selection["training"]:
            raise ConfigError(f"selection missing training field {field}")
        cfg["training"][field] = selection["training"][field]
    cfg["deferred_selection_applied"] = {
        "selection_file": str(selection_path),
        "selected": selection.get("selected"),
        "source_experiment": selection.get("source_experiment"),
    }
    return cfg


def resolve_config(cfg: dict, *, seed: int | None = None, max_env_steps: int | None = None,
                   experiment_id: str | None = None, output_root: str | None = None,
                   base_dir: Path | None = None) -> dict:
    """生成 resolved 配置：CLI 覆盖、去模板标记、解析 deferred selection、计算哈希。"""
    resolved = json.loads(json.dumps(cfg))  # 深拷贝
    resolved = apply_deferred_selection(resolved, base_dir)
    resolved.pop("template_only", None)
    if experiment_id:
        resolved["experiment_id"] = experiment_id
    if seed is not None:
        resolved["training"]["seed"] = int(seed)
    if max_env_steps is not None:
        _integer(int(max_env_steps), "max_env_steps", 1)
        resolved["training"]["max_env_steps"] = int(max_env_steps)
    if output_root:
        resolved["artifacts"]["output_root"] = output_root
    resolved["training"].setdefault("policy_snapshot_steps", [])
    validate_config(resolved)
    cfg_hash = sha256_json({
        "schema_version": resolved["schema_version"],
        "experiment_id": resolved["experiment_id"],
        "agent": resolved["agent"],
        "env": resolved["env"],
        "training": resolved["training"],
        "evaluation": resolved["evaluation"],
    })
    seed_value = int(resolved["training"]["seed"])
    resolved["resolved"] = {
        "seed": seed_value,
        "run_id": f"{resolved['experiment_id']}_seed{seed_value}_{cfg_hash[:8]}",
        "config_hash": cfg_hash,
        "env_hash": env_hash(resolved["env"]),
        "code_hash": code_hash(),
        "shaping_gamma": float(resolved["agent"]["gamma"]),
    }
    return resolved


def output_dir(resolved: dict, base_dir: Path | None = None) -> Path:
    base = Path(base_dir) if base_dir is not None else IMPLEMENTATION_DIR
    root = base / resolved["artifacts"]["output_root"]
    return root / resolved["experiment_id"] / f"seed_{resolved['resolved']['seed']}"


def write_resolved(resolved: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=2, ensure_ascii=False, sort_keys=True)
