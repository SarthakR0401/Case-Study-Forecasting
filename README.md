# End-to-End Time Series Forecasting System

A production-ready forecasting system that trains multiple algorithms, compares performance, and serves the best model via a REST API.

## 🚀 Objective
The goal of this system is to forecast the next 8 weeks (56 days) of sales for multiple states using historical data, while handling seasonality, trends, and missing values.

## 🛠️ Tech Stack
- **Language**: Python 3.11
- **Modeling**: SARIMA, Facebook Prophet, XGBoost, LSTM (TensorFlow)
- **API Framework**: FastAPI (Uvicorn)
- **Data Handling**: Pandas, NumPy, Scikit-learn
- **Containerization**: Docker

## 📂 Project Structure
- `api/`: FastAPI application and schemas.
- `models/`: Wrapper classes for the 4 mandatory forecasting algorithms.
- `utils/`: Data preprocessing, feature engineering, and evaluation metrics.
- `data/`: Source dataset (Excel).
- `saved_models/`: Persisted "Best Model" artifacts for each state.
- `train.py`: The automated model selection and training pipeline.

## 🧠 Key Features
1. **Automated Model Selection**: The training pipeline evaluates all 4 models per state using a time-series validation split and selects the winner based on the lowest **RMSE**.
2. **Feature Engineering**: Implements Lag features (t-1, t-7, t-30), rolling statistics, and calendar flags (day of week, month).
3. **Data Quality**: Automatically handles missing dates and ensures time-series continuity.
4. **Production API**: Serves 8-week forecasts through a validated REST endpoint.

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

## 📊 Results
The system successfully captures weekly and monthly seasonality. For example, in the California dataset, the **SARIMA** model was selected for its superior ability to handle high-frequency cyclical patterns.
