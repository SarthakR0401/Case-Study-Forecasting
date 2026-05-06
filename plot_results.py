import requests
import matplotlib.pyplot as plt
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
STATES_TO_PLOT = ["California", "Texas", "New York", "Florida", "Wyoming", "Alabama"]

def plot_state_forecast(state, ax):
    print(f"Fetching forecast for {state}...")
    try:
        response = requests.post(API_URL, json={"state": state})
        if response.status_code == 200:
            data = response.json()
            
            # Convert to datetime for proper spacing
            hist_dates = pd.to_datetime(data['history_dates'])
            fc_dates = pd.to_datetime(data['dates'])
            
            ax.plot(hist_dates, data['history'], color='gray', linestyle='--', alpha=0.6, label='History')
            ax.plot(fc_dates, data['forecast'], color='blue', marker='o', markersize=4, label=f"Forecast ({data['model_used'].upper()})")
            
            ax.axvline(x=hist_dates[-1], color='red', linestyle='-', alpha=0.3)
            ax.set_title(f"State: {state}", fontsize=12, fontweight='bold')
            
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(fontsize=8)
        else:
            print(f"Error for {state}: {response.text}")
    except Exception as e:
        print(f"Connection failed for {state}: {e}")

plt.rcParams['figure.dpi'] = 100
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
fig.suptitle("Forecasting System: Production Visualization (History vs Forecast)", fontsize=18, fontweight='bold', y=0.98)

axes = axes.flatten()
for i, state in enumerate(STATES_TO_PLOT):
    plot_state_forecast(state, axes[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
