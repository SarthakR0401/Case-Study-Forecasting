import requests
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

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
            
            # Plot history
            ax.plot(hist_dates, data['history'], color='black', alpha=0.3, label='History')
            
            # Plot forecast
            ax.plot(fc_dates, data['forecast'], color='blue', linewidth=2, label=f"Forecast ({data['model_used'].upper()})")
            
            # Plot Confidence Interval
            if data.get('lower') and data.get('upper'):
                ax.fill_between(fc_dates, data['lower'], data['upper'], color='blue', alpha=0.1, label='95% CI')
            
            ax.axvline(x=hist_dates[-1], color='red', linestyle='--', alpha=0.5)
            ax.set_title(f"State: {state}", fontsize=12, fontweight='bold')
            
            # Formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(fontsize=7, loc='upper left')
        else:
            print(f"Error for {state}: {response.text}")
    except Exception as e:
        print(f"Connection failed for {state}: {e}")

def run_visualization():
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    fig.suptitle("Forecasting System v2.0: History vs Predicted (with CI)", fontsize=18, fontweight='bold', y=0.98)

    axes = axes.flatten()
    for i, state in enumerate(STATES_TO_PLOT):
        plot_state_forecast(state, axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('forecast_grid.png')
    print("Dashboard saved as forecast_grid.png")
    plt.show()

if __name__ == "__main__":
    run_visualization()
