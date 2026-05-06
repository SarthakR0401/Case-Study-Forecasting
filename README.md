# End-to-End Time Series Forecasting System (v2.0)

A production-ready forecasting service that trains four algorithms per US state, auto-selects the best performer, and serves 8-week sales predictions through a REST API with confidence intervals.

---

## 🛠️ Tech Stack
- **Modeling**: SARIMA, Facebook Prophet, XGBoost, LSTM (TensorFlow)
- **API Framework**: FastAPI (Uvicorn)
- **Data Handling**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib & Plotly

---

## 🚀 What's New in v2.0 (Bug Fixes & Improvements)

| File | Fix / Improvement Applied |
|---|---|
| `preprocessing.py` | Linear interpolation + ffill/bfill for robust time-series continuity |
| `preprocessing.py` | Robust column auto-detection (handles state/region, sales/revenue, etc.) |
| `features.py` | Automated US Federal Holiday detection via `holidays` library |
| `features.py` | Integrated 30-day rolling windows into machine learning features |
| `prophet_model.py` | Multiplicative seasonality mode + Confidence Interval support |
| `xgboost_model.py` | **Recursive Inference Engine**: Dynamically builds lags for future steps |
| `lstm_model.py` | Normalization via `MinMaxScaler` + `tanh` stability fixes |
| `evaluate.py` | NaN-safe sMAPE implementation (handles zero-sales days) |
| `main.py` | **Lifespan Startup Cache**: Pre-loads data for sub-millisecond response |
| `main.py` | Confidence intervals returned in API for all models |
| `plot_results.py` | Date-aware x-axis + shaded 95% Confidence Interval bands |

---

## 📂 Project Structure

```
forecasting_system/
├── train.py            # Automated training pipeline & model selection
├── plot_results.py     # Visualization client (matplotlib + Plotly)
├── api/                # FastAPI application, lifespan cache, and schemas
├── models/             # Implementations for SARIMA, Prophet, XGBoost, and LSTM
├── utils/              # Preprocessing, feature engineering, and evaluation
├── data/               # Source Excel dataset
└── saved_models/       # Model artifacts and metrics_report.csv
```

---

## 🏃 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline
```bash
python train.py
```

### 3. Start the API
```bash
uvicorn api.main:app --reload
```
View interactive documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Evaluation
Models are compared on the **last 56 days** using:
- **RMSE** (Primary selection metric)
- **MAE** & **MAPE**
- **sMAPE** (Symmetric Mean Absolute Percentage Error)
