"""预登记学习速度指标：去重验证点、T90、旧规则段首与AUC。

本模块只读取验证日志，不修改原始 CSV。重复 t_env 且核心指标一致时只保留一份；
核心指标冲突则报错，禁止任取一条。
"""
from __future__ import annotations

import math

CORE = ("success_rate", "collision_rate", "timeout_rate", "mean_return",
        "mean_raw_base_return", "n")


def distinct_rows(rows):
    by_step, duplicates = {}, []
    for original in rows:
        row = dict(original)
        step = int(row["t_env"])
        row["t_env"] = step
        for key in CORE:
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"Non-finite {key} at {step}")
        if row.get("success_rate") is None or row.get("mean_return") is None:
            raise ValueError(f"Missing core validation metrics at {step}")
        if not 0 <= row["success_rate"] <= 1:
            raise ValueError(f"Invalid success rate at {step}")
        if step in by_step:
            for key in CORE:
                a, b = by_step[step].get(key), row.get(key)
                if a is None and b is None:
                    continue
                if a is None or b is None or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"Conflicting {key} at duplicate step {step}")
            duplicates.append({"t_env": step, "same_core_metrics": True})
        by_step[step] = row
    return [by_step[t] for t in sorted(by_step)], duplicates


def first_confirmation(rows, qualifies, *, interval=30000):
    streak = []
    previous = None
    for row in rows:
        step = row["t_env"]
        if previous is not None and step - previous > interval:
            streak = []
        if qualifies(row):
            streak.append(step)
            if len(streak) >= 3:
                return {"segment_start": streak[-3], "confirmed_at": step, "reached": True}
        else:
            streak = []
        previous = step
    return {"segment_start": None, "confirmed_at": None, "reached": False}


def summarize_learning(rows, *, budget, interval=30000):
    rows, duplicates = distinct_rows(rows)
    expected = list(range(interval, budget + 1, interval))
    if not expected or expected[-1] != budget:
        expected.append(budget)
    steps = [r["t_env"] for r in rows]
    if steps != expected:
        raise ValueError(f"Validation grid mismatch: expected {expected}, got {steps}")
    if len(rows) < 5:
        raise ValueError("At least five real validation points are needed")
    area = sum((b["t_env"] - a["t_env"]) * (a["success_rate"] + b["success_rate"]) / 2
               for a, b in zip(rows, rows[1:]))
    late = sum(r["success_rate"] for r in rows[-5:]) / 5
    best5 = max(sum(r["success_rate"] for r in rows[i:i + 5]) / 5
                for i in range(len(rows) - 4))
    return {
        "budget": budget, "n_distinct_points": len(rows), "duplicates": duplicates,
        "T90": first_confirmation(rows, lambda r: r["success_rate"] >= .9, interval=interval),
        "legacy": first_confirmation(rows, lambda r: r["success_rate"] >= .8 and r["mean_return"] > 2,
                                     interval=interval),
        "success_auc": area / (budget - interval),
        "last_five_success_mean": late, "best_five_success_mean": best5,
        "tail_drop": best5 - late,
        "last_five_all_at_least_90pct": all(r["success_rate"] >= .9 for r in rows[-5:]),
        "last_five_mean_raw_return": sum(float(r.get("mean_raw_base_return", r["mean_return"]))
                                         for r in rows[-5:]) / 5,
    }


def screening_eligible(candidate, baseline, *, budget=300000):
    """两个输入都把相同训练种子映射到 summarize_learning 结果。"""
    if set(candidate) != set(baseline) or len(candidate) != 2:
        raise ValueError("Screening requires the same two training seeds")
    seeds = sorted(candidate)
    if any(candidate[s]["budget"] != budget or baseline[s]["budget"] != budget for s in seeds):
        raise ValueError("Screening summaries must have the same prescribed budget")
    deltas = [candidate[s]["success_auc"] - baseline[s]["success_auc"] for s in seeds]
    if any(d < -.02 - 1e-12 for d in deltas):
        return False
    ccount = sum(candidate[s]["T90"]["reached"] for s in seeds)
    bcount = sum(baseline[s]["T90"]["reached"] for s in seeds)
    mean_delta = sum(deltas) / 2
    more_reached = ccount > bcount and mean_delta >= -.02 - 1e-12
    faster = False
    if ccount == bcount == 2:
        cmean = sum(candidate[s]["T90"]["confirmed_at"] for s in seeds) / 2
        bmean = sum(baseline[s]["T90"]["confirmed_at"] for s in seeds) / 2
        faster = cmean <= .9 * bmean + 1e-12
    higher_auc = ccount >= bcount and mean_delta >= .02 - 1e-12
    return more_reached or faster or higher_auc


def screening_rank(seed_summaries, candidate_id, *, budget=300000, updates=0):
    """升序元组：达标种子数、T90、AUC、末段原始回报、更新次数、候选ID。"""
    values = list(seed_summaries.values())
    placeholder = budget + 30000
    times = [v["T90"]["confirmed_at"] if v["T90"]["reached"] else placeholder for v in values]
    return (-sum(v["T90"]["reached"] for v in values), sum(times) / len(times),
            -sum(v["success_auc"] for v in values) / len(values),
            -sum(v["last_five_mean_raw_return"] for v in values) / len(values),
            updates, candidate_id)
