"""
app.py
-------
AI-Enabled UAV Piston Engine Health Monitoring System -- SIH26054 Prototype

IMPORTANT (read before demoing):
This prototype uses the NASA C-MAPSS FD001 turbofan-degradation dataset
purely to validate the RUL / anomaly-detection / health-scoring PIPELINE.
It is NOT UAV piston-engine telemetry and the code never claims that it is.
Original C-MAPSS sensor IDs (s3, s4, s11, s12, s15, s20, s21) are preserved
throughout. See README.md for the full explanation.
"""

import sys
from pathlib import Path
import plotly.graph_objects as go
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent))
from src.data_loader import (
    SELECTED_SENSORS,
    SENSOR_INFO,
    load_train,
    load_test,
    load_test_rul,
    data_files_present,
)
from src.preprocessing import add_train_rul, add_engineered_features, sensor_trend_slope
from src.rul_model import predict_rul
from src.anomaly_detection import score_anomaly, anomaly_status
from src.health_score import compute_health_score, condition_from_score, CONDITION_COLORS
from src.recommendations import maintenance_recommendation, alerts, predictive_insight_text

DATA_DIR = "data"
MODELS_DIR = Path("models")

st.set_page_config(
    page_title="SIH26054 - UAV Engine Health Prototype",
    page_icon="✈️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .kpi-card {
        background: #11151c;
        border: 1px solid #2a2f3a;
        border-radius: 10px;
        padding: 18px 16px;
        text-align: center;
    }
    .kpi-label {
        color: #9aa4b2;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
    }
    .banner {
        background: #1b2230;
        border-left: 4px solid #4c8bf5;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #cbd3e0;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    required = ["rul_model.pkl", "anomaly_model.pkl", "scaler.pkl", "meta.pkl"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        return None
    model = joblib.load(MODELS_DIR / "rul_model.pkl")
    anomaly_model = joblib.load(MODELS_DIR / "anomaly_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    meta = joblib.load(MODELS_DIR / "meta.pkl")
    return {"rul_model": model, "anomaly_model": anomaly_model, "scaler": scaler, "meta": meta}


@st.cache_data(show_spinner=False)
def load_prepared_train():
    df = load_train(DATA_DIR)
    df = add_train_rul(df)
    df, feature_cols = add_engineered_features(df)
    return df, feature_cols


@st.cache_data(show_spinner=False)
def load_prepared_test():
    df = load_test(DATA_DIR)
    df, feature_cols = add_engineered_features(df)
    rul_df = load_test_rul(DATA_DIR)
    return df, feature_cols, rul_df


@st.cache_data(show_spinner=False)
def score_full_dataset(_models, df, feature_cols):
    """Scales features and computes RUL + anomaly score for every row."""
    scaler = _models["scaler"]
    model = _models["rul_model"]
    anomaly_model = _models["anomaly_model"]
    meta = _models["meta"]

    X = df[feature_cols].fillna(0.0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    pred_rul = predict_rul(model, X_scaled)
    raw_scores, _ = score_anomaly(anomaly_model, X_scaled)
    status = anomaly_status(raw_scores, meta["warn_threshold"], meta["anomaly_threshold"])

    out = df.copy()
    out["predicted_RUL"] = pred_rul
    out["anomaly_raw_score"] = raw_scores
    out["anomaly_status"] = status
    return out


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("AI-Enabled UAV Piston Engine Health Monitoring System")
st.caption("SIH26054 — Predictive Maintenance & Engine Health Prototype")

st.markdown(
    """
    <div class="banner">
    <b>Prototype Validation Dataset:</b> NASA C-MAPSS FD001 &nbsp;|&nbsp;
    Telemetry shown is C-MAPSS turbofan simulation data used for
    predictive-maintenance pipeline validation. The final SIH26054 system
    will use real MALE UAV aero piston-engine telemetry — see the
    "Future SIH System" section below.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Guard: data & model availability
# --------------------------------------------------------------------------
files_present = data_files_present(DATA_DIR)
if not all(files_present.values()):
    missing = [k for k, v in files_present.items() if not v]
    st.error(
        "Missing required NASA C-MAPSS FD001 file(s): "
        + ", ".join(missing)
        + ". Please place train_FD001.txt, test_FD001.txt and RUL_FD001.txt "
        "inside the 'data/' folder."
    )
    st.stop()

models = load_models()
if models is None:
    st.error(
        "Trained model artifacts not found in 'models/'. "
        "Please run:  python train_model.py  before launching the dashboard."
    )
    st.stop()

meta = models["meta"]

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Controls")

page = st.sidebar.radio(
    "Page",
    ["Live Dashboard", "Model Evaluation (Test Set)", "Future SIH Roadmap"],
)

dataset_choice = st.sidebar.selectbox("Dataset", ["Test FD001", "Train FD001"])

try:
    if dataset_choice == "Train FD001":
        raw_df, feature_cols = load_prepared_train()
        rul_lookup = None
    else:
        raw_df, feature_cols, rul_lookup = load_prepared_test()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

scored_df = score_full_dataset(models, raw_df, feature_cols)

engine_ids = sorted(scored_df["unit"].unique().tolist())
engine_id = st.sidebar.selectbox("Engine ID", engine_ids, index=0)

sensor_choice = st.sidebar.selectbox(
    "Sensor",
    SELECTED_SENSORS,
    format_func=lambda s: f"{s} — {SENSOR_INFO[s]['name']}",
)

engine_df = scored_df[scored_df["unit"] == engine_id].sort_values("cycle").reset_index(drop=True)
latest_row = engine_df.iloc[-1]

# Per-engine calculated values
current_cycle = int(latest_row["cycle"])
predicted_rul_val = float(latest_row["predicted_RUL"])
anomaly_stat_val = str(latest_row["anomaly_status"])
raw_anom_score = float(latest_row["anomaly_raw_score"])

slopes = sensor_trend_slope(engine_df, sensors=SELECTED_SENSORS, last_n=15)
health = compute_health_score(
    predicted_rul=predicted_rul_val,
    raw_anomaly_score=raw_anom_score,
    score_min=meta["score_min"],
    score_max=meta["score_max"],
    sensor_slopes=slopes,
    rul_cap=meta["rul_cap"],
)
health_score_val = health["score"]
condition = condition_from_score(health_score_val)
condition_color = CONDITION_COLORS[condition]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Cycle:** {current_cycle}")
st.sidebar.markdown(f"**Predicted RUL:** {predicted_rul_val:.0f} cycles")
st.sidebar.markdown(f"**Health Score:** {health_score_val:.0f} / 100")
st.sidebar.markdown(f"**Engine Condition:** {condition}")
st.sidebar.markdown(f"**Anomaly Status:** {anomaly_stat_val}")

st.sidebar.markdown("---")
with st.sidebar.expander("Optional: upload your own CSV"):
    uploaded = st.file_uploader(
        "Upload a C-MAPSS-format CSV (optional — demo works without this)",
        type=["csv"],
    )
    if uploaded is not None:
        st.info(
            "File received. Custom-upload scoring is not wired into this "
            "prototype's main flow — the primary demo uses the local "
            "data/ files as required by the problem statement."
        )


# ==========================================================================
# PAGE 1: LIVE DASHBOARD
# ==========================================================================
if page == "Live Dashboard":

    # ---- KPI cards ----
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, color in [
        (k1, "Engine Condition", condition, condition_color),
        (k2, "Predicted RUL", f"{predicted_rul_val:.0f} cycles", "#4c8bf5"),
        (k3, "Health Score", f"{health_score_val:.0f} / 100", condition_color),
        (k4, "Anomaly Status", anomaly_stat_val,
         "#e74c3c" if anomaly_stat_val == "ANOMALY" else ("#f1c40f" if anomaly_stat_val == "WARNING" else "#2ecc71")),
    ]:
        col.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{color};">{value}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # ---- Sensor overview ----
    st.subheader("Sensor Overview")
    overview_rows = []
    for s in SELECTED_SENSORS:
        cur_val = float(latest_row[s])
        slope = slopes.get(s, 0.0)
        trend_word = "Rising" if slope > 0.05 else ("Falling" if slope < -0.05 else "Stable")
        sensor_status = "ANOMALY" if anomaly_stat_val == "ANOMALY" else ("WARNING" if anomaly_stat_val == "WARNING" else "NORMAL")
        overview_rows.append(
            {
                "Sensor ID": s,
                "Name": SENSOR_INFO[s]["name"],
                "Description": SENSOR_INFO[s]["desc"],
                "Current Value": round(cur_val, 3),
                "Trend": trend_word,
                "Status": sensor_status,
            }
        )
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    st.write("")

    # ---- Two large charts ----
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sensor Degradation Trend")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=engine_df["cycle"], y=engine_df[sensor_choice],
                mode="lines+markers", name=sensor_choice,
                line=dict(color="#4c8bf5"),
            )
        )
        fig.update_layout(
            title=f"Engine {engine_id} — {sensor_choice} ({SENSOR_INFO[sensor_choice]['name']}) vs Cycle",
            xaxis_title="Cycle", yaxis_title=f"{sensor_choice} value",
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("RUL / Degradation Trend")
        fig2 = go.Figure()
        if dataset_choice == "Train FD001":
            fig2.add_trace(
                go.Scatter(x=engine_df["cycle"], y=engine_df["RUL"],
                            mode="lines", name="Actual RUL (ground truth)",
                            line=dict(color="#2ecc71"))
            )
        fig2.add_trace(
            go.Scatter(x=engine_df["cycle"], y=engine_df["predicted_RUL"],
                        mode="lines+markers", name="Predicted RUL",
                        line=dict(color="#e67e22", dash="dot"))
        )
        if dataset_choice == "Test FD001" and rul_lookup is not None:
            true_final = rul_lookup.loc[rul_lookup["unit"] == engine_id, "RUL"]
            if len(true_final):
                fig2.add_trace(
                    go.Scatter(
                        x=[current_cycle], y=[float(true_final.values[0])],
                        mode="markers", name="Actual RUL (final, from RUL_FD001.txt)",
                        marker=dict(color="#2ecc71", size=12, symbol="star"),
                    )
                )
        fig2.update_layout(
            title=f"Engine {engine_id} — RUL vs Cycle",
            xaxis_title="Cycle", yaxis_title="RUL (cycles)",
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.write("")

    # ---- Predictive insights ----
    st.subheader("Predictive Insights")
    avg_slope = float(np.mean(list(slopes.values()))) if slopes else 0.0
    trend_dir = "up" if avg_slope > 0.03 else ("down" if avg_slope < -0.03 else "flat")
    insight_text = predictive_insight_text(
        engine_id, predicted_rul_val, health_score_val, anomaly_stat_val, trend_dir
    )
    st.info(insight_text)

    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Predicted RUL", f"{predicted_rul_val:.0f} cycles")
    ic2.metric("Health Score", f"{health_score_val:.0f} / 100")
    ic3.metric("Anomaly Detection", anomaly_stat_val)
    ic4.metric("Degradation Trend", trend_dir.upper())

    st.write("")

    # ---- Maintenance recommendation ----
    st.subheader("Maintenance Recommendation")
    st.caption("Prototype Maintenance Recommendation — rule-based, for demonstration only.")
    rec = maintenance_recommendation(health_score_val, anomaly_stat_val, predicted_rul_val)
    st.markdown(f"**STATUS:** {rec['status']}")
    st.markdown(f"**REASON:** {rec['reason']}")
    st.markdown(f"**ACTION:** {rec['action']}")
    for extra_msg in rec["extra"]:
        st.markdown(f"- {extra_msg}")

    st.write("")

    # ---- Engine health history ----
    st.subheader("Engine Health History")
    health_hist = []
    for _, row in engine_df.iterrows():
        row_slopes = slopes  # reuse latest-window slopes as a light-weight proxy across the run
        h = compute_health_score(
            predicted_rul=float(row["predicted_RUL"]),
            raw_anomaly_score=float(row["anomaly_raw_score"]),
            score_min=meta["score_min"],
            score_max=meta["score_max"],
            sensor_slopes=row_slopes,
            rul_cap=meta["rul_cap"],
        )
        health_hist.append(h["score"])
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(x=engine_df["cycle"], y=health_hist, mode="lines+markers",
                    name="Health Score", line=dict(color="#9b59b6"))
    )
    fig3.add_hline(y=80, line_dash="dash", line_color="#2ecc71", annotation_text="Healthy")
    fig3.add_hline(y=60, line_dash="dash", line_color="#f1c40f", annotation_text="Warning")
    fig3.add_hline(y=40, line_dash="dash", line_color="#e67e22", annotation_text="Degraded")
    fig3.update_layout(
        title=f"Engine {engine_id} — Prototype Health Score vs Cycle",
        xaxis_title="Cycle", yaxis_title="Health Score (0-100)",
        template="plotly_dark", height=380, yaxis_range=[0, 100],
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.write("")

    # ---- Alerts ----
    st.subheader("Alerts")
    for level, msg in alerts(health_score_val, anomaly_stat_val, predicted_rul_val):
        if level == "CRITICAL":
            st.error(f"🔴 {msg}")
        elif level in ("WARNING", "ANOMALY"):
            st.warning(f"🟠 {msg}")
        else:
            st.success(f"🟢 {msg}")

    st.write("")

    # ---- Digital Twin Health Layer ----
    st.subheader("Digital Twin Health Layer (Conceptual)")
    st.caption(
        "A simplified conceptual layer comparing observed sensor behavior against "
        "expected (rolling baseline) behavior and the AI's degradation prediction — "
        "NOT a physics-based digital twin simulation."
    )
    dt_rows = []
    for s in SELECTED_SENSORS:
        observed = float(latest_row[s])
        expected = float(latest_row[f"{s}_roll_mean"])
        deviation = observed - expected
        dt_rows.append(
            {
                "Sensor": f"{s} ({SENSOR_INFO[s]['name']})",
                "Observed": round(observed, 3),
                "Expected (rolling baseline)": round(expected, 3),
                "Deviation": round(deviation, 3),
            }
        )
    st.dataframe(pd.DataFrame(dt_rows), use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 2: MODEL EVALUATION (TEST SET)
# ==========================================================================
elif page == "Model Evaluation (Test Set)":
    st.subheader("Model Evaluation on Unseen Test Engines (FD001)")
    st.caption(
        "For every test engine, the model predicts RUL at the FINAL observed "
        "cycle in test_FD001.txt and compares it against the true RUL in "
        "RUL_FD001.txt. These metrics are calculated live from the model — "
        "nothing here is hardcoded."
    )

    test_df, test_feat_cols, rul_df = load_prepared_test()
    test_scored = score_full_dataset(models, test_df, test_feat_cols)

    final_rows = test_scored.sort_values("cycle").groupby("unit").tail(1).reset_index(drop=True)
    eval_df = final_rows.merge(rul_df, on="unit", how="left")
    eval_df["Absolute Error"] = (eval_df["predicted_RUL"] - eval_df["RUL"]).abs()

    mae = mean_absolute_error(eval_df["RUL"], eval_df["predicted_RUL"])
    rmse = mean_squared_error(eval_df["RUL"], eval_df["predicted_RUL"]) ** 0.5
    r2 = r2_score(eval_df["RUL"], eval_df["predicted_RUL"])

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f} cycles")
    m2.metric("RMSE", f"{rmse:.2f} cycles")
    m3.metric("R²", f"{r2:.3f}")

    st.write("")
    st.markdown("**Per-Engine Results**")
    display_df = eval_df.rename(
        columns={"unit": "Engine ID", "cycle": "Final Cycle", "predicted_RUL": "Predicted RUL", "RUL": "Actual RUL"}
    )[["Engine ID", "Final Cycle", "Predicted RUL", "Actual RUL", "Absolute Error"]]
    display_df["Predicted RUL"] = display_df["Predicted RUL"].round(1)
    display_df["Absolute Error"] = display_df["Absolute Error"].round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

    st.write("")
    st.markdown("**Predicted RUL vs Actual RUL**")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eval_df["RUL"], y=eval_df["predicted_RUL"], mode="markers",
            marker=dict(color="#4c8bf5", size=8), name="Test engines",
        )
    )
    max_val = float(max(eval_df["RUL"].max(), eval_df["predicted_RUL"].max()))
    fig.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                    line=dict(color="#e74c3c", dash="dash"), name="Perfect prediction")
    )
    fig.update_layout(
        xaxis_title="Actual RUL (cycles)", yaxis_title="Predicted RUL (cycles)",
        template="plotly_dark", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Training vs. Validation performance (from train_model.py)"):
        st.json(meta["val_metrics"])
        st.caption(
            f"Trained on {meta['n_train_engines']} engines, validated on "
            f"{meta['n_val_engines']} held-out engines (engine-wise split, no row leakage)."
        )


# ==========================================================================
# PAGE 3: FUTURE SIH ROADMAP
# ==========================================================================
else:
    st.subheader("Future SIH System — Roadmap")
    st.caption(
        "This prototype (validated on C-MAPSS FD001) is the first stage of the "
        "full SIH26054 solution. The roadmap below shows the intended evolution "
        "toward a real, physics-informed UAV piston-engine Digital Twin."
    )

    roadmap_steps = [
        "Current Prototype (C-MAPSS pipeline validation)",
        "Real UAV Piston-Engine Telemetry Acquisition",
        "Physics-Based Engine Model",
        "Digital Twin (fused physics + AI)",
        "Real-Time Anomaly Detection",
        "Fault Classification (engine-specific fault modes)",
        "RUL Prediction (piston-engine specific)",
        "Mission Simulation & Reliability Scoring",
        "Edge AI Deployment (onboard / ground station)",
    ]

    for i, step in enumerate(roadmap_steps, start=1):
        st.markdown(f"**{i}. {step}**")
        if i < len(roadmap_steps):
            st.markdown("<div style='text-align:center; color:#9aa4b2;'>↓</div>", unsafe_allow_html=True)

    st.write("")
    st.info(
        "This prototype demonstrates the end-to-end predictive-maintenance "
        "pipeline (data → RUL → anomaly detection → health score → "
        "recommendation → dashboard) on a validated public dataset. The final "
        "SIH26054 system will replace C-MAPSS with real MALE UAV aero "
        "piston-engine telemetry and physics-informed models, without "
        "changing this overall architecture."
    )
