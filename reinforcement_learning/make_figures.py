"""从真实运行日志生成图表：奖励/成功率对比、战场地图与路径、奖励分项。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.metrics import moving_average, read_csv  # noqa: E402
from envs import make_env  # noqa: E402
from envs.geometry import rectangle_vertices  # noqa: E402

OUTPUTS = BASE_DIR / "outputs"
FIG_DIR = OUTPUTS / "figures"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run_path(exp: str, seed: int) -> Path:
    return OUTPUTS / exp / f"seed_{seed}"


def eval_rows(exp: str, seed: int = 0) -> list[dict]:
    return read_csv(run_path(exp, seed) / "eval_metrics.csv")


def final_test(exp: str, seed: int = 0) -> dict | None:
    path = run_path(exp, seed) / "final_test_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_episodes(exp: str, seed: int = 0) -> list[dict]:
    path = run_path(exp, seed) / "final_test_episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def plot_curve(ax, exp: str, metric: str, label: str, color=None, max_steps=None):
    rows = eval_rows(exp)
    if not rows:
        return False
    xs = [int(r["t_env"]) for r in rows]
    ys = [float(r[metric]) if r.get(metric) is not None else np.nan for r in rows]
    if max_steps:
        keep = [i for i, x in enumerate(xs) if x <= max_steps]
        xs = [xs[i] for i in keep]
        ys = [ys[i] for i in keep]
    if not xs:
        return False
    ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.0, alpha=0.45,
            color=color, label=label)
    xs_ma, ys_ma = moving_average(xs, ys, 3)
    if xs_ma:
        ax.plot(xs_ma, ys_ma, linewidth=1.8, color=color)
    return True


def figure_comparison(runs: list[tuple[str, str]], metric: str, ylabel: str, title: str,
                      out: Path, max_steps: int | None = None) -> bool:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    any_data = False
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#8c564b", "#7f7f7f",
               "#bcbd22", "#17becf"]
    for i, (exp, label) in enumerate(runs):
        any_data |= plot_curve(ax, exp, metric, label, color=palette[i % len(palette)],
                               max_steps=max_steps)
    if not any_data:
        plt.close(fig)
        return False
    ax.set_xlabel("Environment steps (t_env)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def per_run_figure(exp: str) -> bool:
    rows = eval_rows(exp)
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [int(r["t_env"]) for r in rows]
    ys = [float(r["mean_return"]) for r in rows]
    ax.plot(xs, ys, marker="o", markersize=3, alpha=0.5, label="raw")
    xa, ya = moving_average(xs, ys, 3)
    if xa:
        ax.plot(xa, ya, linewidth=1.8, label="moving average (w=3)")
    ax.set_xlabel("Environment steps (t_env)")
    ax.set_ylabel("Validation mean return")
    ax.set_title(f"{exp} seed0 - validation return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = run_path(exp, 0) / "figures" / "reward_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def battle_map() -> bool:
    cfg_path = BASE_DIR / "configs" / "battle_W1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    spec = cfg["env"]
    env = make_env(spec, shaping_gamma=0.98)
    field = env._field
    if field is None:
        return False
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    extent = [field.xs[0], field.xs[-1], field.ys[0], field.ys[-1]]
    masked = np.where(np.isfinite(field.dist), field.dist, np.nan)
    im = ax.imshow(masked, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label="Dijkstra distance to goal (m)")
    for xmin, xmax, ymin, ymax in spec["obstacles"]:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   facecolor="0.15", edgecolor="black"))
    for cx, cy in spec["risk"]["centers"]:
        ax.add_patch(plt.Circle((cx, cy), spec["risk"]["radius"], color="red", alpha=0.28))
    gx, gy = spec["potential"]["goal"]
    ax.plot([gx], [gy], "w*", markersize=14, markeredgecolor="black", label="potential goal")
    ax.plot([-spec["exit_half_width"], spec["exit_half_width"]],
            [spec["map_bounds"][2]] * 2, "g--", linewidth=2, label="exit gap")
    half_l, half_w = spec["length"] / 2, spec["width"] / 2
    ax.add_patch(plt.Rectangle((-half_w, spec["map_bounds"][2] - half_l),
                               2 * half_w, 2 * half_l, fill=False, edgecolor="lime",
                               linewidth=1.5, label="robot footprint"))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BattlefieldNavigation-v0 map / inflated free space / PBRS distance field")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "battle_map.png", dpi=200)
    plt.close(fig)
    return True


def _traj_record(exp: str, seed: int) -> dict | None:
    path = OUTPUTS / "round2_review" / "trajectories" / exp / f"test_seed_{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _robot_pose_vertices(record: dict, index: int) -> np.ndarray:
    pose = record["poses"][index]
    return rectangle_vertices(pose[0], pose[1], pose[2],
                              record["robot"]["length"], record["robot"]["width"])


def battle_paths(seeds: list[int]) -> bool:
    """按 seed 分面，绘制 W0/W1 真实姿态车体与完整终点（不裁剪 y<-1.5）。"""
    records = {(exp, seed): _traj_record(exp, seed)
               for exp in ("battle_W0", "battle_W1") for seed in seeds}
    if not any(records.values()):
        return False
    spec = json.loads((BASE_DIR / "configs" / "battle_W1.json").read_text(encoding="utf-8"))["env"]
    all_vertices = []
    for record in records.values():
        if not record:
            continue
        for i in (0, len(record["poses"]) // 2, len(record["poses"]) - 1):
            all_vertices.append(_robot_pose_vertices(record, i))

    ncols = 2
    nrows = int(np.ceil(len(seeds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 2.9 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {"battle_W0": "#1f77b4", "battle_W1": "#ff7f0e"}
    labels = {"battle_W0": "W0 (beta=0)", "battle_W1": "W1 (beta=1)"}
    for ax, seed in zip(axes, seeds):
        for xmin, xmax, ymin, ymax in spec["obstacles"]:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       facecolor="0.25", edgecolor="black", zorder=2))
        for cx, cy in spec["risk"]["centers"]:
            ax.add_patch(plt.Circle((cx, cy), spec["risk"]["radius"], color="red",
                                      alpha=0.18, zorder=1))
        ax.axhline(spec["map_bounds"][2], color="green", linestyle="--", linewidth=1.0,
                   zorder=1)
        ax.plot([-spec["exit_half_width"], spec["exit_half_width"]],
                [spec["map_bounds"][2]] * 2, color="green", linewidth=3, zorder=3)
        notes = []
        for exp in ("battle_W0", "battle_W1"):
            record = records.get((exp, seed))
            if not record:
                continue
            pose = np.asarray(record["poses"], dtype=float)
            ax.plot(pose[:, 0], pose[:, 1], color=colors[exp], linewidth=1.2, zorder=4,
                    label=labels[exp])
            ax.plot(pose[0, 0], pose[0, 1], "o", color=colors[exp], markersize=4.5,
                    markeredgecolor="white", zorder=5)
            outcome = record["reason"]
            symbol = "*" if outcome == "success" else ("X" if outcome == "collision" else "s")
            ax.plot(pose[-1, 0], pose[-1, 1], symbol, color=colors[exp], markersize=9,
                    markeredgecolor="black", zorder=6)
            for i in (0, len(record["poses"]) // 2, len(record["poses"]) - 1):
                verts = _robot_pose_vertices(record, i)
                ax.add_patch(plt.Polygon(verts, closed=True, fill=False,
                                         edgecolor=colors[exp], linewidth=0.9, alpha=0.65,
                                         zorder=4))
            notes.append(f"{labels[exp].split()[0]} {outcome[0].upper()} "
                         f"({record['steps']} steps)")
        ax.set_title(f"seed {seed} — " + " | ".join(notes), fontsize=9.5)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")
    for ax in axes[len(seeds):]:
        ax.axis("off")
    # 视野：地图边界与所有真实车体顶点的包围盒再各加 0.3 m，不截断出口外终点。
    if all_vertices:
        arr = np.concatenate(all_vertices, axis=0)
        xmin = min(spec["map_bounds"][0], float(arr[:, 0].min())) - 0.3
        xmax = max(spec["map_bounds"][1], float(arr[:, 0].max())) + 0.3
        ymin = min(spec["map_bounds"][2], float(arr[:, 1].min())) - 0.3
        ymax = max(spec["map_bounds"][3], float(arr[:, 1].max())) + 0.3
    else:
        xmin, xmax, ymin, ymax = spec["map_bounds"]
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=2, fontsize=10, frameon=False)
    fig.supxlabel("X (m)", fontsize=10)
    fig.supylabel("Y (m)", fontsize=10)
    fig.suptitle("Battlefield fixed test seeds: real pose (theta) trajectories, "
                 "start o / success * / collision X / timeout s", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "battle_paths_fixed_seeds.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "battle_paths_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def baseline_demo() -> bool:
    """B0 失败对照与 tuned DDPG 成功展示（同一 1M 预算、同一验证 seed）。"""
    base_rows = eval_rows("ddpg_baseline")
    tuned_rows = eval_rows("tuned_ddpg")
    if not base_rows or not tuned_rows:
        return False
    final_b = final_test("ddpg_baseline") or {}
    final_t = final_test("tuned_ddpg") or {}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for exp, color, label in (("ddpg_baseline", "#d62728", "B0 baseline"),
                              ("tuned_ddpg", "#2ca02c", "tuned DDPG")):
        rows = base_rows if exp == "ddpg_baseline" else tuned_rows
        xs = [int(r["t_env"]) for r in rows]
        ys = [float(r["mean_return"]) for r in rows]
        axes[0].plot(xs, ys, marker="o", markersize=2.5, alpha=0.4, color=color, label=label)
        xa, ya = moving_average(xs, ys, 3)
        axes[0].plot(xa, ya, linewidth=1.8, color=color)
    axes[0].set_xlabel("Environment steps (t_env)")
    axes[0].set_ylabel("Validation mean return (moving avg w=3)")
    axes[0].set_title("B0 failure vs tuned DDPG success (1M budget)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    labels = ["success", "collision", "timeout"]
    values_b = [final_b.get(k) for k in ("success_rate", "collision_rate", "timeout_rate")]
    values_t = [final_t.get(k) for k in ("success_rate", "collision_rate", "timeout_rate")]
    xs = np.arange(3)
    axes[1].bar(xs - 0.2, [float(v or 0) for v in values_b], width=0.4, color="#d62728",
                label=f"B0 test success {final_b.get('success_rate')}")
    axes[1].bar(xs + 0.2, [float(v or 0) for v in values_t], width=0.4, color="#2ca02c",
                label=f"tuned test success {final_t.get('success_rate')}")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Fixed 100 test seeds outcome rates")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "baseline_success_demo.png", dpi=200)
    plt.close(fig)
    return True


def ddpg_1m_comparison() -> bool:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for exp, color, label in (("ddpg_baseline", "#d62728", "B0 baseline"),
                              ("tuned_ddpg", "#2ca02c", "tuned DDPG")):
        rows = eval_rows(exp)
        if not rows:
            continue
        xs = [int(r["t_env"]) for r in rows]
        for ax, metric, ylabel in ((axes[0], "mean_return", "Validation mean return"),
                                   (axes[1], "success_rate", "Validation success rate")):
            ys = [float(r[metric]) for r in rows]
            ax.plot(xs, ys, marker="o", markersize=2.5, alpha=0.4, color=color, label=label)
            xa, ya = moving_average(xs, ys, 3)
            ax.plot(xa, ya, linewidth=1.8, color=color)
            ax.set_xlabel("Environment steps (t_env)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    axes[0].set_title("Full 1M validation return (B0 vs tuned; w=3)")
    axes[1].set_title("Full 1M validation success rate")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "ddpg_baseline_vs_tuned_1m.png", dpi=200)
    fig.savefig(FIG_DIR / "tuning_1m_comparison.png", dpi=200)
    plt.close(fig)
    return True


def battle_best_vs_final() -> bool:
    path = OUTPUTS / "round2_review" / "checkpoint_comparison.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    exps = [e for e in ("battle_W0", "battle_W1") if e in data]
    if not exps:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    xs = np.arange(len(exps))
    for ax, metric, title in ((axes[0], "success_rate", "Validation success rate"),
                              (axes[1], "mean_return", "Validation mean return")):
        best = [float(data[e]["best_validation"][metric] or 0) for e in exps]
        final = [float(data[e]["final_online"][metric] or 0) for e in exps]
        ax.bar(xs - 0.2, best, width=0.4, color="#1f77b4",
               label="best_validation (selected)")
        ax.bar(xs + 0.2, final, width=0.4, color="#ff7f0e",
               label="final online (1M agent.pt)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{e}\nbest={data[e]['best_validation']['step'] // 1000}k, "
                            f"final={data[e]['final_online']['step'] // 1000}k" for e in exps],
                           fontsize=8)
        ax.set_title(title + " (10 validation seeds)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "battle_best_vs_final.png", dpi=200)
    plt.close(fig)
    return True


def battle_reward_terms() -> bool:
    terms = ["center", "exit", "time", "omega", "action", "success_bonus",
             "collision_penalty", "risk", "shaping"]
    data = {}
    for exp in ("battle_W0", "battle_W1"):
        records = test_episodes(exp)
        if not records:
            continue
        data[exp] = {key: float(np.mean([float(r["reward_terms"][key]) for r in records]))
                     for key in terms}
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    width = 0.38
    xs = np.arange(len(terms))
    for i, (exp, values) in enumerate(data.items()):
        ax.bar(xs + (i - 0.5) * width, [values[k] for k in terms], width=width,
               label=exp)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(terms, rotation=30, ha="right")
    ax.set_ylabel("Mean accumulated reward term per test episode")
    ax.set_title("Battle reward decomposition (final test episodes)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "battle_reward_terms.png", dpi=200)
    plt.close(fig)
    return True


def write_tables(manifest: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = manifest.get("runs", {})
    rows = []
    for key, entry in sorted(runs.items()):
        last = entry.get("last_eval") or {}
        final = entry.get("final_test") or {}
        rows.append({
            "run": key,
            "experiment_id": entry["experiment_id"],
            "seed": entry["seed"],
            "config_hash": entry.get("config_hash"),
            "env_hash": entry.get("env_hash"),
            "t_env": last.get("t_env"),
            "val_success": last.get("success_rate"),
            "val_collision": last.get("collision_rate"),
            "val_return": last.get("mean_return"),
            "test_success": final.get("success_rate"),
            "test_collision": final.get("collision_rate"),
            "test_timeout": final.get("timeout_rate"),
            "test_return": final.get("mean_return"),
            "test_n": final.get("n_episodes"),
        })
    for name in ("tuning_runs.csv", "algorithm_summary.csv", "battle_summary.csv"):
        with open(FIG_DIR / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                                    ["run", "experiment_id", "seed"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    selection = OUTPUTS / "selections" / "tuning_selection.json"
    if selection.exists():
        payload = json.loads(selection.read_text(encoding="utf-8"))
        with open(FIG_DIR / "tuning_summary.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["run_id", "changes", "success_rate", "collision_rate",
                             "mean_return", "not_worse_than_b0"])
            for record in payload.get("records", []):
                stats = record.get("stats") or {}
                writer.writerow([record.get("run_id"), json.dumps(record.get("changes", {})),
                                 stats.get("success_rate"), stats.get("collision_rate"),
                                 stats.get("mean_return"), record.get("not_worse_than_b0")])
            if payload.get("combo"):
                stats = payload["combo"]["stats"]
                writer.writerow([payload["combo"]["run_id"],
                                 json.dumps(payload["combo"]["changes"]),
                                 stats.get("success_rate"), stats.get("collision_rate"),
                                 stats.get("mean_return"), None])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成课程实验06图表")
    parser.add_argument("--suite", default=str(OUTPUTS / "suite_manifest.json"))
    args = parser.parse_args(argv)
    suite_path = Path(args.suite)
    manifest = json.loads(suite_path.read_text(encoding="utf-8")) if suite_path.exists() else {"runs": {}}
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    made.append(figure_comparison(
        [("ddpg_baseline", "B0"), ("T1_actor_lr", "T1 actor_lr 3e-4"),
         ("T2_noise_005", "T2 sigma 0.05"), ("T3_noise_010", "T3 sigma 0.10"),
         ("T4_gamma_099", "T4 gamma 0.99"), ("T5_tau_001", "T5 tau 0.01"),
         ("T6_batch_128", "T6 batch 128"), ("combo_screen", "combo")],
        "mean_return", "Validation mean return",
        "Single-factor screening (300k steps, train seed 0)",
        FIG_DIR / "tuning_reward_comparison.png", max_steps=300000))
    made.append(figure_comparison(
        [("ddpg_baseline", "B0"), ("T1_actor_lr", "T1"), ("T2_noise_005", "T2"),
         ("T3_noise_010", "T3"), ("T4_gamma_099", "T4"), ("T5_tau_001", "T5"),
         ("T6_batch_128", "T6"), ("combo_screen", "combo")],
        "success_rate", "Validation success rate",
        "Single-factor screening success (300k steps)", FIG_DIR / "tuning_success_comparison.png",
        max_steps=300000))
    made.append(figure_comparison(
        [("ddpg_baseline", "DDPG B0"), ("tuned_ddpg", "tuned DDPG"),
         ("td3_matched", "TD3 matched"), ("td3_practical", "TD3 practical")],
        "mean_return", "Validation mean return",
        "DDPG vs TD3 (same env, same budget)", FIG_DIR / "algorithms_reward_comparison.png"))
    made.append(figure_comparison(
        [("ddpg_baseline", "DDPG B0"), ("tuned_ddpg", "tuned DDPG"),
         ("td3_matched", "TD3 matched"), ("td3_practical", "TD3 practical")],
        "success_rate", "Validation success rate",
        "Algorithms success comparison", FIG_DIR / "algorithms_success_comparison.png"))
    made.append(figure_comparison(
        [("battle_W0", "W0 beta=0"), ("battle_W1", "W1 beta=1")],
        "mean_return", "Validation mean return",
        "Battlefield PBRS ablation (shaped return shown; compare raw return in CSV)",
        FIG_DIR / "battle_ablation_comparison.png"))
    made.append(battle_map())
    made.append(battle_paths([10000, 10001, 10002, 10003, 10004]))
    made.append(battle_reward_terms())
    made.append(baseline_demo())
    made.append(ddpg_1m_comparison())
    made.append(battle_best_vs_final())
    # 300k 筛选同图另存更明确文件名，便于报告引用。
    if (FIG_DIR / "tuning_reward_comparison.png").exists():
        import shutil
        shutil.copy2(FIG_DIR / "tuning_reward_comparison.png",
                     FIG_DIR / "tuning_300k_comparison.png")
    for exp in ("ddpg_baseline", "tuned_ddpg", "td3_matched", "td3_practical",
                "battle_W0", "battle_W1"):
        made.append(per_run_figure(exp))
    write_tables(manifest)
    import hashlib
    review = OUTPUTS / "round2_review"
    review.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(FIG_DIR.glob("*.png")):
        entries.append({"name": path.name, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    figure_manifest = {
        "schema_version": 1,
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "smoothing_window": 3,
        "trajectory_seeds": [10000, 10001, 10002, 10003, 10004],
        "notes": {
            "tuning_300k_comparison.png": "B0 与 T1–T6/combo 的共同 300k 前缀；验证 seed 1000–1009",
            "tuning_1m_comparison.png": "B0 与 tuned 的完整 1M 验证曲线；窗口 3",
            "algorithms_reward_comparison.png": "DDPG/tuned/TD3 matched/practical，同一环境组与 1M 预算",
            "battle_paths_fixed_seeds.png": "冻结模型重播 seed 10000–10004，真实 theta 车体；每 seed 分面",
            "battle_best_vs_final.png": "best_validation 与末次在线 actor 在 10 个验证 seed 上比较",
            "baseline_success_demo.png": "B0 失败对照 + tuned DDPG 成功；固定 100 测试 seed",
        },
        "figures": entries,
    }
    (review / "figure_manifest.json").write_text(json.dumps(figure_manifest, ensure_ascii=False,
                                                            indent=2), encoding="utf-8")
    print(json.dumps({"figures": [str(p) for p in FIG_DIR.glob('*.png')],
                      "made": sum(bool(m) for m in made)}, ensure_ascii=False, indent=2))
    if not all(m for m in made[:6]):
        print("WARNING: 部分图缺少数据，已标记为未完成", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
