import pandas as pd
import numpy as np

# Dataset 1: Stock Price Data
def feature_engineering_stock(data):
    # Ensure 'Date' column has no whitespace or unexpected characters
    data['Date'] = data['Date'].str.strip()

    # Convert 'Date' to datetime format, forcing timezone-naive datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', utc=True).dt.tz_localize(None)

    # Drop rows with invalid or missing dates
    data = data.dropna(subset=['Date'])

    # Extract time-based features without using .dt
    data['Year'] = data['Date'].apply(lambda x: x.year)
    data['Month'] = data['Date'].apply(lambda x: x.month)
    data['Day'] = data['Date'].apply(lambda x: x.day)
    data['DayOfWeek'] = data['Date'].apply(lambda x: x.weekday())

    # Calculate moving averages
    data['5D_MA_Close'] = data['Close'].rolling(window=5).mean()
    data['30D_MA_Close'] = data['Close'].rolling(window=30).mean()

    # Calculate rate of change
    data['Daily_Return'] = data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()

    # Technical Indicators
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    # Lag features
    for lag in [1, 3, 5]:
        data[f'Lag_{lag}_Close'] = data['Close'].shift(lag)

    return data


# Dataset 2: Fundamental Data
def feature_engineering_fundamental(data):
    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Drop rows with invalid or missing dates
    data = data.dropna(subset=['Date'])

    # Sort by Symbol and Date (ascending order for chronological processing)
    data = data.sort_values(by=['Symbol', 'Date'])

    # Calculate financial ratios
    data['Gross_Profit_Margin'] = data['Gross_Profit'] / data['Total_Revenue']
    data['Operating_Margin'] = data['Operating_Income'] / data['Total_Revenue']
    data['Net_Profit_Margin'] = data['Net_Income'] / data['Total_Revenue']

    # Calculate growth rates (YoY Change)
    for col in ['Total_Revenue', 'Gross_Profit', 'Operating_Income', 'Net_Income']:
        data[f'{col}_YoY_Change'] = data.groupby('Symbol')[col].pct_change()

    # Valuation ratios
    data['P/E_Ratio'] = data['Market_Price'] / data['EPS']
    data['P/S_Ratio'] = data['Market_Price'] / data['Total_Revenue']

    # Rolling averages for stability
    for col in ['Total_Revenue', 'Gross_Profit']:
        data[f'{col}_Rolling_Mean'] = (
            data.groupby('Symbol')[col]
            .rolling(window=4, min_periods=1)  # Rolling window of 4; NaN for rows with insufficient data
            .mean()
            .reset_index(0, drop=True)
        )

    # Lag features
    for lag in [1, 2, 3]:
        for col in ['Total_Revenue', 'Net_Income']:
            data[f'Lag_{lag}_{col}'] = data.groupby('Symbol')[col].shift(lag)

    # Retain NaN where calculations are not possible for early years
    return data



# Load datasets
stock_data = pd.read_csv("preprocessed_historical_data.csv")
fundamental_data = pd.read_csv("preprocessed_fundamental_data.csv")

# Apply feature engineering
try:
    stock_data = feature_engineering_stock(stock_data)
    fundamental_data = feature_engineering_fundamental(fundamental_data)

    # Save the processed data
    stock_data.to_csv("feature_engineered_stock_data.csv", index=False)
    fundamental_data.to_csv("feature_engineered_fundamental_data.csv", index=False)
    print("Feature engineering complete!")
except Exception as e:
    print(f"Error during feature engineering: {e}")
