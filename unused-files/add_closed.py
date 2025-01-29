import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Load the predictions file
predictions_file = "it_industry_predictions.csv"
try:
    predictions_df = pd.read_csv(predictions_file)
    predictions_df["Last Updated"] = pd.to_datetime(predictions_df["Last Updated"])
except FileNotFoundError:
    print("Predictions file not found. Ensure 'it_industry_predictions.csv' exists.")
    exit()

# Ensure Actual Close column exists
if "Actual Close" not in predictions_df.columns:
    predictions_df["Actual Close"] = None

# Fetch yesterday's actual close prices for each ticker
print("Fetching actual close prices for yesterday...")
yesterday = datetime.now() - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

for index, row in predictions_df.iterrows():
    ticker = row["Ticker"]
    try:
        # Fetch historical data up to yesterday
        ticker_data = yf.Ticker(ticker).history(start=yesterday_str, end=row["Last Updated"].strftime("%Y-%m-%d"))
        if not ticker_data.empty:
            # Get the most recent close price before yesterday
            actual_close = ticker_data.iloc[-1]["Close"]
            predictions_df.at[index, "Actual Close"] = actual_close
        else:
            print(f"No data available for ticker {ticker} up to {yesterday_str}.")
    except Exception as e:
        print(f"Could not fetch data for ticker {ticker}: {e}")

# Save updated predictions file
predictions_df.to_csv(predictions_file, index=False)
print("Updated predictions saved with actual close prices.")
