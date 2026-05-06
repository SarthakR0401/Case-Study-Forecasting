# Production-Ready Time Series Forecasting System (v2.0)

A high-performance forecasting engine designed to predict 8 weeks of sales for multiple geographic regions. The system automatically evaluates statistical, additive, and deep learning models to select the most accurate forecaster per entity.

---

## 🚀 Key Improvements in v2.0

This version introduces significant architectural upgrades and critical bug fixes to ensure production stability:

- **Recursive Forecasting Engine**: Machine learning models (XGBoost) now use a recursive inference loop, building future features (lags/rolling) dynamically from their own predictions.
- **Ultra-Fast API Performance**: Implemented a **Lifespan Startup Cache** that pre-loads and pre-processes data once, reducing API response times to sub-milliseconds.
- **Robust Data Resilience**: Enhanced column auto-detection and smarter missing value handling (linear interpolation + ffill/bfill) to handle real-world messy datasets.
- **Confidence Intervals**: Full support for uncertainty bands in SARIMA and Prophet, providing business users with "Lower" and "Upper" sales bounds.
- **Stability Fixes**: Corrected LSTM gradient instability (tanh activation + scaling) and Prophet daily seasonality overfitting.

---

## 🛠️ Tech Stack

- **Core**: Python 3.11, Pandas, NumPy
- **Modeling**: SARIMA (Statsmodels), Facebook Prophet, XGBoost, LSTM (TensorFlow)
- **API**: FastAPI (Uvicorn)
- **Visualization**: Matplotlib, Plotly (Interactive)

---

## 📂 Project Structure

```
forecasting_system/
├── api/                # FastAPI logic, lifespan cache, and Pydantic schemas
├── models/             # Logic for all 4 forecasting algorithms (v2.0 recursive engines)
├── utils/              # Data cleaning, feature engineering (lags, holidays), and metrics
├── data/               # Source Excel datasets
├── saved_models/       # Persisted "Best-in-Class" model artifacts and metrics reports
├── train.py            # Automated training & per-state model selection pipeline
├── plot_results.py     # Multi-state visualization dashboard with CI bands
└── generate_docs.py    # Formal documentation (.docx) generator
```

---

## 🧠 Intelligence & Feature Engineering

The system extracts deep temporal patterns using:
1. **Lags**: (t-1, t-7, t-30) to capture daily, weekly, and monthly dependencies.
2. **Rolling Stats**: 7-day and 30-day moving averages and volatility (std).
3. **External Context**: Automated US Federal Holiday detection via `holidays` library.
4. **Calendar Flags**: Day-of-week, month, and weekend indicators.

---

## 🏃 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training & Selection
```bash
python train.py
```
*This will evaluate all models per state and export a `metrics_report.csv` in `saved_models/`.*

### 3. Launch the Prediction API
```bash
uvicorn api.main:app --reload
```
View interactive Swagger docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Evaluation Metrics
Models are compared using a strict **Time-Series Hold-out Validation** (last 56 days) on:
- **RMSE** (Primary Selection Metric)
- **MAE** & **MAPE**
- **sMAPE** (NaN-safe implementation)

---

## 📄 Documentation
A formal technical document explaining the methodology and insights can be generated locally:
```bash
python generate_docs.py
```
This produces `Forecasting_System_Documentation.docx`.
