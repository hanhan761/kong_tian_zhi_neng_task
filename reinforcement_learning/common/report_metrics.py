"""报告派生指标：只做名称/单位映射与说明，不改变环境、日志或模型。

历史字段 `min_clearance` 实际是“车体中心到实体障碍的欧氏距离 - 半对角线包络圆半径”，
且不包含侧/顶墙；它不是旋转矩形真实最小距离，也不是连续时间安全证明。
"""
from __future__ import annotations

import math

# 车体包络圆半径 rho = 0.5 * sqrt(l^2 + w^2)，l=1.8 m, w=1.2 m
RHO_CIRCLE = 0.5 * math.hypot(1.8, 1.2)
# risk_integral = sum(dt_sub/0.25 * h)，单位为“按 0.25 s 归一化的风险暴露”
RISK_TIME_NORMALIZER_S = 0.25

SOURCE_MIN_CLEARANCE_FIELD = "min_clearance"
DERIVED_MIN_CLEARANCE_FIELD = "min_circle_envelope_gap_to_obstacles_m"
SOURCE_RISK_FIELD = "risk_integral"
DERIVED_RISK_EXPOSURE_FIELD = "risk_exposure_s"

MIN_CLEARANCE_DEFINITION = (
    "g_circle(t)=min_j d(p_t,O_j)-rho, rho=0.5*sqrt(l^2+w^2)=1.081665 m; "
    "g_ep=min_t g_circle(t). d 为中心到轴对齐实体障碍的欧氏距离；"
    "只采样已执行子步，不含侧/顶墙，也不是旋转车体真实最小距离。"
)

RISK_DEFINITION = (
    "risk_integral = sum_k (dt_sub/0.25) * h(p_k)，h(p)=max_j exp(-||p-c_j||^2/(2*sigma^2))；"
    "原值是 0.25 s 归一化暴露，不是秒。物理积分 sum(dt_sub*h)=risk_integral*0.25 s。"
)

REWARD_TERMS = [
    {"id": "r_x", "name": "center", "config_key": "center_coef", "exponent_key": "center_exponent",
     "formula": "a1 exp(-a2 x^2)", "purpose": "鼓励靠近通道中心 x=0"},
    {"id": "r_y", "name": "exit", "config_key": "exit_coef", "exponent_key": "exit_exponent",
     "formula": "a3 exp(-a4 (y+1.5))", "purpose": "鼓励向底部出口前进"},
    {"id": "r_t", "name": "time", "config_key": "time_cost",
     "formula": "-a5（每 0.25 s 归一化步）", "purpose": "时间惩罚，鼓励更快到达"},
    {"id": "r_o", "name": "omega", "config_key": "omega_cost",
     "formula": "-a6 omega^2", "purpose": "抑制无效旋转"},
    {"id": "r_a", "name": "action", "config_key": "action_squared_cost",
     "formula": "-a7 (u_F^2+u_T^2)", "purpose": "抑制大幅动作"},
    {"id": "r_s", "name": "success_bonus", "config_key": "success_bonus",
     "formula": "a8 * 1[success]", "purpose": "稀疏成功奖励"},
    {"id": "r_c", "name": "collision_penalty", "config_key": "collision_penalty",
     "formula": "-a9 * 1[collision]", "purpose": "碰撞惩罚"},
]

# 战场附加项（不替代基础 7 项）
BATTLE_EXTRA_TERMS = [
    {"id": "risk", "name": "risk", "formula": "-kappa * sum(dt_sub/0.25 * h(p_k))",
     "purpose": "降低进入风险区的暴露"},
    {"id": "shaping", "name": "shaping",
     "formula": "beta[gamma Phi(s') - Phi(s)]",
     "purpose": "PBRS 势差塑形；真实 terminal 时 Phi=0，timeout 不清零"},
]


def derived_metrics(row: dict) -> dict:
    """返回派生字段，保留 source_field 与原值，不修改输入。"""
    out = dict(row)
    if row.get(SOURCE_MIN_CLEARANCE_FIELD) is not None:
        out[DERIVED_MIN_CLEARANCE_FIELD] = float(row[SOURCE_MIN_CLEARANCE_FIELD])
        out["min_clearance_source_field"] = SOURCE_MIN_CLEARANCE_FIELD
        out["min_clearance_definition"] = MIN_CLEARANCE_DEFINITION
    if row.get(SOURCE_RISK_FIELD) is not None:
        out[DERIVED_RISK_EXPOSURE_FIELD] = float(row[SOURCE_RISK_FIELD]) * RISK_TIME_NORMALIZER_S
        out["risk_definition"] = RISK_DEFINITION
    return out


def circle_envelope_radius(length: float = 1.8, width: float = 1.2) -> float:
    return 0.5 * math.hypot(length, width)


def circle_envelope_gap(center_x: float, center_y: float, rect: tuple[float, float, float, float],
                        length: float = 1.8, width: float = 1.2) -> float:
    """复现历史 min_clearance 口径（只到实体障碍、不含墙）。"""
    xmin, xmax, ymin, ymax = rect
    nx = min(max(center_x, xmin), xmax)
    ny = min(max(center_y, ymin), ymax)
    return math.hypot(center_x - nx, center_y - ny) - circle_envelope_radius(length, width)
