"""
health_score.py
-----------------
"Prototype Engine Health Score" (0-100): a transparent, rule-based
combination of three components. This is NOT a certified/industrial
health index -- it exists to demonstrate the concept.

Weights are intentionally exposed as a simple dict so they are easy
to tune for the demo.
"""

from __future__ import annotations
import numpy as np

DEFAULT_WEIGHTS = {"rul": 0.5, "anomaly": 0.3, "trend": 0.2}


def rul_component(predicted_rul: float, rul_cap: float) -> float:
    """0-100: 100 at/above rul_cap cycles remaining, 0 at RUL=0."""
    rul_cap = max(rul_cap, 1.0)
    val = np.clip(predicted_rul, 0, rul_cap) / rul_cap * 100.0
    return float(val)


def anomaly_component(raw_score: float, score_min: float, score_max: float) -> float:
    """0-100: normalises the IsolationForest raw score (higher=more normal)."""
    if score_max <= score_min:
        return 100.0
    norm = (raw_score - score_min) / (score_max - score_min)
    return float(np.clip(norm, 0, 1) * 100.0)


def trend_component(sensor_slopes: dict) -> float:
    """
    0-100: penalises large-magnitude normalised sensor slopes
    (fast-changing sensors suggest active degradation).
    """
    if not sensor_slopes:
        return 100.0
    avg_abs_slope = float(np.mean([abs(v) for v in sensor_slopes.values()]))
    score = 100.0 - np.clip(avg_abs_slope * 60.0, 0, 100)
    return float(score)


def compute_health_score(
    predicted_rul: float,
    raw_anomaly_score: float,
    score_min: float,
    score_max: float,
    sensor_slopes: dict,
    rul_cap: float,
    weights: dict | None = None,
) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    r = rul_component(predicted_rul, rul_cap)
    a = anomaly_component(raw_anomaly_score, score_min, score_max)
    t = trend_component(sensor_slopes)
    score = weights["rul"] * r + weights["anomaly"] * a + weights["trend"] * t
    score = float(np.clip(score, 0, 100))
    return {"score": score, "rul_component": r, "anomaly_component": a, "trend_component": t}


def condition_from_score(score: float) -> str:
    if score >= 80:
        return "HEALTHY"
    elif score >= 60:
        return "WARNING"
    elif score >= 40:
        return "DEGRADED"
    else:
        return "CRITICAL"


CONDITION_COLORS = {
    "HEALTHY": "#2ecc71",
    "WARNING": "#f1c40f",
    "DEGRADED": "#e67e22",
    "CRITICAL": "#e74c3c",
}
