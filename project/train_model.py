"""
train_model.py
----------------
Run this ONCE (or whenever data/ changes) before launching the Streamlit
dashboard. It:
  1. Loads data/train_FD001.txt
  2. Generates RUL labels + engineered features
  3. Splits by engine (no row-level leakage) into train/validation
  4. Trains a RandomForestRegressor for RUL prediction
  5. Trains an IsolationForest anomaly detector on relatively healthy rows
  6. Saves everything needed by app.py into models/

Usage:
    python train_model.py
"""

import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_train, SELECTED_SENSORS
from src.preprocessing import (
    add_train_rul,
    add_engineered_features,
    split_by_engine,
    fit_scaler,
    scale_features,
)
from src.rul_model import train_rul_model, evaluate
from src.anomaly_detection import train_anomaly_model, score_anomaly

DATA_DIR = Path("project\data")
MODELS_DIR = Path("project\models")


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading training data (data/train_FD001.txt)")
    print("=" * 60)
    train_df = load_train(DATA_DIR)
    print(f"Loaded {train_df.shape[0]} rows across {train_df['unit'].nunique()} engines.")

    print("\nSTEP 2: Generating RUL labels (max_cycle - current_cycle)")
    train_df = add_train_rul(train_df)

    print("\nSTEP 3: Engineering degradation features "
          "(rolling mean / std / diff for 7 selected sensors)")
    train_df, feature_cols = add_engineered_features(train_df)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    print("\nSTEP 4: Splitting by engine (80% train / 20% validation engines)")
    train_split, val_split = split_by_engine(train_df, train_frac=0.8, seed=42)
    print(f"Train engines: {train_split['unit'].nunique()} | "
          f"Validation engines: {val_split['unit'].nunique()}")

    print("\nSTEP 5: Fitting scaler + scaling features")
    scaler = fit_scaler(train_split, feature_cols)
    train_scaled = scale_features(train_split, feature_cols, scaler)
    val_scaled = scale_features(val_split, feature_cols, scaler)

    X_train, y_train = train_scaled[feature_cols], train_scaled["RUL"]
    X_val, y_val = val_scaled[feature_cols], val_scaled["RUL"]

    print("\nSTEP 6: Training RandomForestRegressor (RUL model)")
    model = train_rul_model(X_train, y_train)
    metrics = evaluate(model, X_val, y_val)
    print(f"Validation -> MAE: {metrics['MAE']:.2f} cycles | "
          f"RMSE: {metrics['RMSE']:.2f} cycles | R2: {metrics['R2']:.3f}")

    print("\nSTEP 7: Training IsolationForest anomaly detector on relatively "
          "healthy rows (RUL >= median)")
    healthy_mask = train_scaled["RUL"] >= train_scaled["RUL"].median()
    X_healthy = train_scaled.loc[healthy_mask, feature_cols]
    anomaly_model = train_anomaly_model(X_healthy, contamination=0.05)

    raw_scores_all, _ = score_anomaly(anomaly_model, train_scaled[feature_cols])
    score_min = float(np.percentile(raw_scores_all, 1))
    score_max = float(np.percentile(raw_scores_all, 99))
    warn_threshold = float(np.percentile(raw_scores_all, 10))
    anomaly_threshold = float(np.percentile(raw_scores_all, 2))

    print("\nSTEP 8: Saving model artifacts to models/")
    joblib.dump(model, MODELS_DIR / "rul_model.pkl")
    joblib.dump(anomaly_model, MODELS_DIR / "anomaly_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    meta = {
        "feature_cols": feature_cols,
        "sensors": SELECTED_SENSORS,
        "val_metrics": {"MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]},
        "score_min": score_min,
        "score_max": score_max,
        "warn_threshold": warn_threshold,
        "anomaly_threshold": anomaly_threshold,
        "rul_cap": float(train_df["RUL"].quantile(0.95)),
        "n_train_engines": int(train_split["unit"].nunique()),
        "n_val_engines": int(val_split["unit"].nunique()),
    }
    joblib.dump(meta, MODELS_DIR / "meta.pkl")

    print("\nDone. Saved:")
    for f in ["rul_model.pkl", "anomaly_model.pkl", "scaler.pkl", "meta.pkl"]:
        print(f"  models/{f}")
    print("\nYou can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
