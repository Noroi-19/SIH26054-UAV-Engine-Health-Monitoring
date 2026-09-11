# AI-Enabled UAV Piston Engine Health Monitoring System — Prototype
### Smart India Hackathon 2026 — SIH26054

## Problem Statement
**SIH26054**: "AI-Enabled Real-Time Digital Twin System for Health Monitoring,
Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines
used in MALE UAVs."

## Prototype Objective
This is an **internal hackathon prototype**, not a production system. It
demonstrates the full predictive-maintenance pipeline end-to-end:

```
Dataset → Data Loading → Sensor Selection → Preprocessing →
Degradation Analysis → RUL Prediction → Anomaly Detection →
Health Score → Engine Condition → Maintenance Recommendation →
Interactive Dashboard
```

## Dataset — Important Note
**A suitable public dataset of MALE UAV aero piston-engine telemetry with
degradation/RUL labels could not be found.** This prototype therefore uses
the **NASA C-MAPSS FD001** turbofan-engine degradation simulation dataset
**only to validate the RUL / anomaly-detection / health-scoring pipeline**.

- C-MAPSS is **not** claimed to be UAV piston-engine data anywhere in this
  codebase or dashboard.
- Original C-MAPSS sensor names (`s1`...`s21`) are preserved exactly —
  they are never renamed to piston-engine terms like "Oil Pressure" or
  "RPM".
- The final SIH26054 system will replace this dataset with real UAV
  piston-engine telemetry and physics-based engine models, without changing
  the overall pipeline architecture (see "Future SIH System" in the
  dashboard).

Required files (place in `data/`):
```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```
Source: NASA Prognostics Center of Excellence — Turbofan Engine Degradation
Simulation Data Set (FD001 subset).

## Technology Stack
- Python
- Pandas / NumPy
- Scikit-learn (RandomForestRegressor, IsolationForest, StandardScaler)
- Plotly (interactive charts)
- Streamlit (dashboard)
- Joblib (model persistence)

No deep learning, no Kalman/particle filters, no databases, no message
brokers, and no cloud/container infrastructure are used — this keeps the
prototype simple, fast to build, and reliable to demo.

## Sensors Used (7 selected, original C-MAPSS IDs preserved)
| Sensor | Name | Description |
|---|---|---|
| s3  | T30  | Total temperature at HPC outlet |
| s4  | T50  | Total temperature at LPT outlet |
| s11 | Ps30 | Static pressure at HPC outlet |
| s12 | Phi  | Ratio of fuel flow to Ps30 |
| s15 | BPR  | Bypass ratio |
| s20 | W31  | HPT coolant bleed |
| s21 | W32  | LPT coolant bleed |

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Setup Data

1. Download the NASA C-MAPSS FD001 files.
2. Copy `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` into the
   `data/` folder (replacing the placeholder note file there).

## Train the Model (run once)

```bash
python train_model.py
```

This loads `train_FD001.txt`, builds RUL labels and engineered features,
splits engines 80/20 (train/validation, no row leakage), trains the RUL
model and the anomaly detector, and saves everything to `models/`.

## Run the Dashboard

```bash
streamlit run app.py
```

## How RUL Is Calculated
For every training row:
```
RUL = max_cycle_of_that_engine − current_cycle
```
Example: an engine that runs for 192 cycles has RUL 191 at cycle 1 and
RUL 0 at cycle 192. This generated label is the regression target.

## Model Methodology
1. **Feature engineering** (per engine, per sensor, over the 7 selected
   sensors): raw value, rolling mean, rolling std (5-cycle window), and
   cycle-to-cycle difference — 28 features total.
2. **Engine-wise train/validation split** (80% / 20% of engines) — this
   avoids leaking cycles from the same engine across train and validation.
3. **RUL model**: `RandomForestRegressor` trained on the engineered
   features, scaled with `StandardScaler`.
4. **Anomaly detector**: `IsolationForest` trained only on the healthier
   half of the training rows (RUL ≥ median), so it learns what "normal"
   looks like. Thresholds for NORMAL / WARNING / ANOMALY are calibrated
   from percentiles of the training score distribution.
5. **Health Score (0–100)** — a transparent, rule-based combination of
   three components (weights are simple constants in `src/health_score.py`,
   easy to tune):
   - RUL component (closer to 0 cycles → lower score)
   - Anomaly component (normalised IsolationForest score)
   - Trend component (average magnitude of recent normalised sensor
     slopes — fast-moving sensors reduce the score)

   This is explicitly labelled **"Prototype Engine Health Score"** in the
   dashboard — it is **not** a certified or industrial health index.
6. **Engine condition** from health score: HEALTHY (80–100) / WARNING
   (60–79) / DEGRADED (40–59) / CRITICAL (0–39).
7. **Maintenance recommendation**: rule-based text generated from the
   calculated health score, anomaly status, and predicted RUL — labelled
   **"Prototype Maintenance Recommendation"**. The system never claims to
   autonomously authorize aircraft operation.

## Dashboard Features
- **Live Dashboard**: KPI cards (Engine Condition, Predicted RUL, Health
  Score, Anomaly Status), sensor overview table, sensor trend chart,
  RUL/degradation trend chart, predictive insights (auto-generated text),
  maintenance recommendation, engine health history chart, alerts panel,
  and a conceptual "Digital Twin Health Layer" (Observed vs Expected vs
  Deviation).
- **Model Evaluation (Test Set)**: runs the trained model on every unseen
  test engine, predicts RUL at the final observed cycle, compares against
  `RUL_FD001.txt`, and displays MAE / RMSE / R², a per-engine results
  table, and a Predicted-vs-Actual scatter plot.
- **Future SIH Roadmap**: shows how this prototype extends toward the full
  SIH26054 Digital Twin system.
- Sidebar controls: Dataset (Train/Test), Engine ID, Sensor selection, plus
  an optional CSV upload (not required for the primary demo).

All values shown are computed live from the dataset and trained model —
nothing is hardcoded.

## How to Demonstrate to Judges
1. Open the app — show the title and the C-MAPSS validation-dataset banner.
2. Sidebar → Dataset: **Test FD001**.
3. Pick an engine ID with a low predicted RUL (sort mentally by trying a
   few, or check the Model Evaluation table first).
4. Point out the 4 KPI cards (Condition / RUL / Health Score / Anomaly).
5. Open the Sensor Trend chart — show degradation over cycles.
6. Open the RUL trend chart — show predicted RUL approaching the actual
   RUL marker (from `RUL_FD001.txt`).
7. Switch to **Model Evaluation (Test Set)** — show MAE/RMSE/R² computed
   on unseen engines, and the Predicted-vs-Actual scatter plot.
8. Show the Maintenance Recommendation section and Alerts panel.
9. Switch to **Future SIH Roadmap** — explain that this pipeline will be
   re-targeted at real UAV piston-engine telemetry.

## Known Limitations
- C-MAPSS FD001 is turbofan (jet engine) simulation data, not piston-engine
  data — it is used solely to validate the ML pipeline.
- FD001 has no labelled fault types (e.g. injector fault, misfire,
  lubrication failure), so anomaly detection reports *"abnormal sensor
  behavior"*, not a specific diagnosed fault.
- The Health Score and its weights are a simple, transparent prototype
  construct, not a certified/industrial index.
- The "Digital Twin Health Layer" is conceptual (observed vs. rolling
  baseline vs. deviation), not a physics-based simulation.
- No real-time streaming telemetry is implemented in this prototype.

## Future Improvements for the Final SIH System
- Replace C-MAPSS with real MALE UAV aero piston-engine telemetry.
- Add a physics-based (thermodynamic/mechanical) piston-engine model to
  fuse with the AI model for a true digital twin.
- Introduce labelled fault-mode data for fault classification (not just
  anomaly detection).
- Real-time ingestion from onboard sensors and ground-station telemetry.
- Mission-reliability simulation incorporating flight profile and
  environmental conditions.
- Edge AI deployment for onboard inference with constrained compute.
