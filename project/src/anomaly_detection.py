"""
anomaly_detection.py
---------------------
Unsupervised anomaly detection using IsolationForest, trained on
relatively healthy engine data (top half of RUL in the training set).

IMPORTANT: This model flags *abnormal sensor behavior* relative to
healthy operation. It does NOT identify a specific physical fault
(e.g. injector fault, misfire, lubrication failure). FD001 has no
fault-type labels, so this prototype intentionally avoids any such
claim -- see recommendations.py for the exact wording used in the UI.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest


def train_anomaly_model(X_healthy, contamination: float = 0.05, **overrides) -> IsolationForest:
    params = dict(n_estimators=200, contamination=contamination, random_state=42)
    params.update(overrides)
    model = IsolationForest(**params)
    model.fit(X_healthy)
    return model


def score_anomaly(model: IsolationForest, X):
    """Returns (raw_score, label). raw_score: higher = more normal."""
    raw_score = model.decision_function(X)
    label = model.predict(X)  # 1 = normal, -1 = anomaly (per IsolationForest's own cutoff)
    return raw_score, label


def anomaly_status(raw_score, warn_threshold: float, anomaly_threshold: float):
    """
    Maps a raw anomaly score (higher = more normal) to one of
    NORMAL / WARNING / ANOMALY using thresholds calibrated on the
    training distribution (see train_model.py).
    """
    raw_score = np.atleast_1d(raw_score)
    status = np.where(
        raw_score < anomaly_threshold,
        "ANOMALY",
        np.where(raw_score < warn_threshold, "WARNING", "NORMAL"),
    )
    return status if status.shape[0] > 1 else status[0]
