"""日志、统计与收敛判定工具。"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


def ensure_finite(values: dict, context: str = "") -> None:
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite value in {context}: {key}={value}")


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if new_file:
            self._writer.writeheader()

    def write(self, row: dict) -> None:
        missing = [f for f in self.fieldnames if f not in row]
        if missing:
            raise ValueError(f"missing csv columns: {missing}")
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if value in ("", "None", None):
                converted[key] = None
                continue
            try:
                converted[key] = int(value)
            except ValueError:
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
        out.append(converted)
    return out


def moving_average(x: list[float], y: list[float], window: int) -> tuple[list[float], list[float]]:
    """m=min(w,n)，返回与 valid 卷积等长的横轴；n=0 时返回空，不外推。"""
    n = len(y)
    if n == 0 or window < 1:
        return [], []
    m = min(window, n)
    values = np.convolve(np.asarray(y, dtype=np.float64), np.ones(m) / m, mode="valid")
    return list(x[m - 1:]), [float(v) for v in values]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def summarize_episodes(records: list[dict]) -> dict:
    """结局互斥比例 + 回报/路径统计；失败样本不填 0 混入成功路径均值。"""
    n = len(records)
    if n == 0:
        return {"n": 0, "success_rate": None, "collision_rate": None, "timeout_rate": None,
                "mean_return": None, "std_return": None, "mean_raw_base_return": None,
                "mean_path_length_all": None, "mean_path_length_success": None,
                "mean_elapsed_all": None, "mean_elapsed_success": None,
                "mean_energy": None, "mean_risk_integral": None, "mean_min_clearance": None}
    success = [r for r in records if r.get("success")]
    collisions = [r for r in records if r.get("collision")]
    timeouts = [r for r in records if r.get("timeout")]
    total = len(success) + len(collisions) + len(timeouts)
    if total != n:
        raise ValueError(f"episode outcomes are not mutually exclusive: {total} vs {n}")
    returns = [float(r["return"]) for r in records]
    raw = [float(r["raw_base_return"]) for r in records]
    paths = [float(r["path_length"]) for r in records]
    elapsed = [float(r["elapsed_sim_time"]) for r in records]
    energy = [float(r.get("energy", 0.0)) for r in records]
    risk = [float(r.get("risk_integral", 0.0)) for r in records]
    clearance = [float(r["min_clearance"]) for r in records if r.get("min_clearance") is not None]
    mean_return, std_return = mean_std(returns)
    return {
        "n": n,
        "success_rate": len(success) / n,
        "collision_rate": len(collisions) / n,
        "timeout_rate": len(timeouts) / n,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_raw_base_return": float(np.mean(raw)),
        "mean_path_length_all": float(np.mean(paths)),
        "mean_path_length_success": float(np.mean([float(r["path_length"]) for r in success])) if success else None,
        "mean_elapsed_all": float(np.mean(elapsed)),
        "mean_elapsed_success": float(np.mean([float(r["elapsed_sim_time"]) for r in success])) if success else None,
        "mean_energy": float(np.mean(energy)),
        "mean_risk_integral": float(np.mean(risk)),
        "mean_min_clearance": float(np.mean(clearance)) if clearance else None,
    }


def convergence_step(eval_rows: list[dict], return_threshold: float,
                     success_threshold: float, consecutive: int) -> int | None:
    """连续 checkpoint 同时满足回报与成功率，返回首个 checkpoint 的 t_env。"""
    streak = 0
    for i, row in enumerate(eval_rows):
        if row.get("mean_return") is None or row.get("success_rate") is None:
            streak = 0
            continue
        if float(row["mean_return"]) > return_threshold and float(row["success_rate"]) >= success_threshold:
            streak += 1
            if streak >= consecutive:
                return int(eval_rows[i - consecutive + 1]["t_env"])
        else:
            streak = 0
    return None
