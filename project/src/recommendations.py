"""
recommendations.py
--------------------
Simple, transparent rule-based "Prototype Maintenance Recommendation"
and alert generation. All text is generated from calculated values --
nothing here is hardcoded per-engine.
"""

from __future__ import annotations


def maintenance_recommendation(health_score: float, anomaly_stat: str, predicted_rul: float,
                                rul_low_threshold: float = 20.0) -> dict:
    if health_score >= 80:
        status = "No Immediate Action"
        reason = "Prototype health score indicates normal operating condition."
        action = "Continue operation. No immediate maintenance required."
    elif health_score >= 60:
        status = "Monitor"
        reason = "Prototype health score shows early signs of degradation."
        action = "Monitor engine parameters and schedule inspection."
    elif health_score >= 40:
        status = "Preventive Maintenance Recommended"
        reason = "Predicted RUL and sensor trends indicate ongoing degradation."
        action = "Schedule preventive maintenance and inspect degrading parameters."
    else:
        status = "Critical - Detailed Inspection Required"
        reason = "Prototype health score is critically low."
        action = "Critical condition. Recommend detailed inspection before next mission."

    extra = []
    if anomaly_stat == "ANOMALY":
        extra.append("Abnormal sensor behavior detected. Inspect affected sensor trends.")
    if predicted_rul is not None and predicted_rul < rul_low_threshold:
        extra.append("Remaining useful life is low. Consider maintenance before next mission.")

    return {"status": status, "reason": reason, "action": action, "extra": extra}


def alerts(health_score: float, anomaly_stat: str, predicted_rul: float,
           rul_low_threshold: float = 20.0) -> list:
    msgs = []
    if health_score < 40:
        if predicted_rul is not None and predicted_rul < rul_low_threshold:
            msgs.append(("CRITICAL", "Predicted RUL is critically low."))
        else:
            msgs.append(("CRITICAL", "Prototype health score is critically low."))
    elif health_score < 60:
        msgs.append(("WARNING", "Degradation trend detected."))

    if anomaly_stat == "ANOMALY":
        msgs.append(("ANOMALY", "Abnormal sensor behavior detected."))

    if not msgs:
        msgs.append(("NORMAL", "No critical alerts."))
    return msgs


def predictive_insight_text(engine_id, predicted_rul, health_score, anomaly_stat, trend_direction: str) -> str:
    """Builds the natural-language 'Predictive Insights' paragraph from calculated values only."""
    trend_phrase = {
        "up": "an increasing degradation trend",
        "down": "a stabilising trend",
        "flat": "a broadly stable trend",
    }.get(trend_direction, "a mixed trend")

    anomaly_phrase = {
        "NORMAL": "No significant anomalous behavior was detected.",
        "WARNING": "Sensor behavior shows early signs of deviation from healthy operation.",
        "ANOMALY": "Abnormal sensor behavior was detected relative to healthy operation.",
    }.get(anomaly_stat, "")

    return (
        f"Engine {engine_id} shows {trend_phrase} across the selected sensor parameters. "
        f"The current predicted RUL is {predicted_rul:.0f} cycles and the prototype health "
        f"score is {health_score:.0f}/100. {anomaly_phrase}"
    )
