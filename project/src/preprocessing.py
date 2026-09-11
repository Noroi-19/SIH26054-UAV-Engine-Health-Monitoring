"""
preprocessing.py
-----------------
Preprocessing pipeline for the C-MAPSS FD001 prototype:
  1. RUL label generation for training rows
  2. Simple degradation/temporal feature engineering (rolling mean,
     rolling std, cycle-to-cycle difference)
  3. Engine-wise (unit_id-wise) train/validation split to avoid
     row-level leakage between the same engine's cycles
  4. Feature scaling
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import SELECTED_SENSORS

ROLL_WINDOW = 5


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL = max_cycle_of_engine - current_cycle (per problem statement)."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).astype(float)
    return df


def add_engineered_features(df: pd.DataFrame, sensors=None, window: int = ROLL_WINDOW):
    """
    Adds, for each of the 7 selected sensors:
      - raw value (already present)
      - rolling mean over `window` cycles (within the same engine)
      - rolling std over `window` cycles
      - cycle-to-cycle difference

    Returns (df_with_features, feature_column_list)
    """
    sensors = sensors or SELECTED_SENSORS
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True).copy()

    feature_cols = []
    for s in sensors:
        grp = df.groupby("unit")[s]
        roll_mean_col = f"{s}_roll_mean"
        roll_std_col = f"{s}_roll_std"
        diff_col = f"{s}_diff"

        df[roll_mean_col] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[roll_std_col] = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0.0))
        df[diff_col] = grp.diff().fillna(0.0)

        feature_cols.extend([s, roll_mean_col, roll_std_col, diff_col])

    return df, feature_cols


def sensor_trend_slope(engine_df: pd.DataFrame, sensors=None, last_n: int = 15) -> dict:
    """
    Simple linear-trend slope of each sensor over the most recent `last_n`
    cycles for a single engine, normalised by the sensor's own std so
    slopes are roughly comparable across sensors.
    """
    sensors = sensors or SELECTED_SENSORS
    tail = engine_df.sort_values("cycle").tail(last_n)
    slopes = {}
    for s in sensors:
        y = tail[s].values.astype(float)
        if len(y) < 2:
            slopes[s] = 0.0
            continue
        x = np.arange(len(y))
        std = y.std() if y.std() > 1e-6 else 1.0
        slope = np.polyfit(x, y, 1)[0]
        slopes[s] = float(slope / std)
    return slopes


def split_by_engine(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42):
    """Split engines (not rows) into train/validation to avoid leakage."""
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    units = units.copy()
    rng.shuffle(units)
    n_train = max(1, int(len(units) * train_frac))
    train_units = set(units[:n_train])
    val_units = set(units[n_train:])
    train_df = df[df["unit"].isin(train_units)].reset_index(drop=True)
    val_df = df[df["unit"].isin(val_units)].reset_index(drop=True)
    return train_df, val_df


def fit_scaler(df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].fillna(0.0))
    return scaler


def scale_features(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].fillna(0.0))
    return df
