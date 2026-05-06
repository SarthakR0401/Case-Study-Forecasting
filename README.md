# End-to-End Time Series Forecasting System (v2.0)

A production-ready forecasting service that trains four algorithms per US state, auto-selects the best performer, and serves 8-week sales predictions through a REST API with confidence intervals.

---

## What changed in v2.0 (bug fixes)

| File | Original bug | Fix applied |
|---|---|---|
| `preprocessing.py` | `fillna(0)` created false seasonal troughs | Linear interpolation + ffill/bfill |
| `preprocessing.py` | Hard-coded column names crashed on header variations | Robust auto-detection with fallback substring matching |
| `features.py` | Holiday flag absent (required by spec) | Added via `holidays` library (US federal) |
| `features.py` | `rolling_std_30` / `rolling_mean_30` created but never used | Added to `FEATURE_COLS` used by XGBoost |
| `prophet_model.py` | `daily_seasonality=True` → severe overfitting on daily data | Set to `False`; added `multiplicative` seasonality mode |
| `xgboost_model.py` | `predict(None)` silently returned `[0.0]*56` at API time | Recursive feature builder reconstructs lag/rolling from training tail |
| `lstm_model.py` | No normalization — unstable gradients on large sales values | `MinMaxScaler` fit on train, inverse-transform on predict |
| `lstm_model.py` | `relu` activation in LSTM gates — exploding gradients risk | Switched to `tanh` (standard); added `Dropout(0.2)` |
| `evaluate.py` | sMAPE produced `NaN` when both `y_true` and `y_pred` were 0 | Replace `0+0` denominator with `1e-8` guard |
| `main.py` | `load_and_preprocess()` called on every request (~2 s I/O) | `lifespan` startup cache — loaded once, dict lookup per request |
| `main.py` | XGBoost always returned zeros at inference | Calls fixed `model.predict(None)` → recursive inference |
| `main.py` | No confidence intervals in response | SARIMA & Prophet expose `lower` / `upper` arrays |
| `schemas.py` | No input validation on `state` — empty strings accepted | `field_validator` strips whitespace, rejects empty, title-cases |
| `plot_results.py` | String dates → uneven x-axis spacing | Converted to `pd.Timestamp`; `mdates.DateFormatter` for ticks |
| `plot_results.py` | No confidence interval visualization | `ax.fill_between()` for CI band |
| `requirements.txt` | Missing `holidays`, `pytest`, `httpx`, `plotly` | All added with pinned versions |
| *(missing)* | No package `__init__.py` files → import errors | Added for `api/`, `models/`, `utils/` |
| `train.py` | `tail_df` never passed to XGBoost → inference always zeros | Passed explicitly during training |
| `train.py` | No metrics export → model selection unauditable | Exports `saved_models/metrics_report.csv` |

---

## Project structure

```
forecasting_system/
│
├── train.py                    # Run once: trains all models, selects best per state
├── plot_results.py             # Visualization client (matplotlib + Plotly)
├── requirements.txt
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app (startup cache, /predict, /states, /metrics)
│   ├── model_loader.py         # ModelRegistry singleton
│   └── schemas.py              # Pydantic request/response models
│
├── models/
│   ├── __init__.py
│   ├── sarima_model.py         # SARIMAX(1,1,1)(1,1,1)[7] + CI support
│   ├── prophet_model.py        # Prophet yearly+weekly, multiplicative
│   ├── xgboost_model.py        # XGBoost + recursive inference
│   └── lstm_model.py           # LSTM + MinMaxScaler normalization
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Load Excel, interpolate gaps, ensure date continuity
│   ├── features.py             # Lag, rolling, calendar, holiday features
│   └── evaluate.py             # RMSE, MAE, MAPE, sMAPE (NaN-safe)
│
├── data/
│   └── Forecasting Case- Study.xlsx
│
└── saved_models/               # Created by train.py
    ├── <State>_best_model.joblib
    ├── state_model_map.joblib
    └── metrics_report.csv      # Per-state, per-model RMSE leaderboard
```

---

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data
```
data/Forecasting Case- Study.xlsx
```

### 3. Train all models
```bash
python train.py
```

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Visualize
```bash
python plot_results.py
```
