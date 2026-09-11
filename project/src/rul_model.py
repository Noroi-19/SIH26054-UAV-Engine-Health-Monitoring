"""
rul_model.py
------------
Simple, reliable RUL regression model (RandomForestRegressor).
No deep learning, as required by the prototype constraints.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_rul_model(X_train, y_train, **overrides) -> RandomForestRegressor:
    params = dict(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    params.update(overrides)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict_rul(model, X):
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def evaluate(model, X, y_true) -> dict:
    y_pred = predict_rul(model, X)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "y_pred": y_pred}
