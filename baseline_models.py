import yfinance as yf
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from concurrent.futures import ThreadPoolExecutor
import os
import pickle

# Database setup
DB_FILE = "sp500_analysis.db"
MODEL_FILE = "baseline_models.pkl"


# Industry-specific growth and discount rates
industry_rates = {
    "Information Technology": {"growth_rate": 12, "discount_rate": 9},
    "Health Care": {"growth_rate": 10, "discount_rate": 8},
    "Consumer Discretionary": {"growth_rate": 9, "discount_rate": 8.5},
    "Utilities": {"growth_rate": 4, "discount_rate": 10},
    "Financials": {"growth_rate": 6, "discount_rate": 11},
    "Materials": {"growth_rate": 8, "discount_rate": 9},
    "Real Estate": {"growth_rate": 6, "discount_rate": 9.5},
    "Consumer Staples": {"growth_rate": 7, "discount_rate": 9},
    "Energy": {"growth_rate": 5, "discount_rate": 12},
    "Industrials": {"growth_rate": 7, "discount_rate": 10},
    "Telecommunication Services": {"growth_rate": 8, "discount_rate": 9.5},
}

# Function to fetch data from Yahoo Finance
def fetch_data_from_yahoo(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        eps = info.get("trailingEps", None)
        market_price = info.get("regularMarketPrice") or info.get("previousClose")
        revenue_per_share = info.get("revenuePerShare", None)
        ps_ratio = info.get("priceToSalesTrailing12Months", None)

        return eps, market_price, revenue_per_share, ps_ratio
    except Exception as e:
        print(f"Yahoo Finance error for {symbol}: {e}")
        return None, None, None, None

# Function to calculate intrinsic value (using corrected approach)
def calculate_intrinsic_value(row):
    sector = row["Sector"]
    rates = industry_rates.get(sector, None)

    if rates is None:
        return None  # Skip calculation if sector is unknown

    growth_rate = rates["growth_rate"] / 100  # Convert to decimal
    discount_rate = rates["discount_rate"] / 100  # Convert to decimal

    # Calculate next year's EPS
    if row["EPS"] and row["EPS"] > 0:
        next_year_eps = row["EPS"] * (1 + growth_rate)

        # Safeguard for small denominator
        denominator = discount_rate - growth_rate
        if denominator > 0.01:  # Ensure denominator is not too small
            intrinsic_value = next_year_eps / denominator
        else:
            # Use alternative one-year forward DCF approach
            intrinsic_value = next_year_eps / (1 + discount_rate)
    else:
        intrinsic_value = None  # EPS is invalid

    # If intrinsic value is invalid, use P/S ratio approach
    if intrinsic_value is None or intrinsic_value <= 0 or pd.isna(intrinsic_value):
        if row["P/S_Ratio"] and row["P/S_Ratio"] > 0 and row["Revenue_Per_Share"] and row["Revenue_Per_Share"] > 0:
            intrinsic_value = row["P/S_Ratio"] * row["Revenue_Per_Share"]
        else:
            intrinsic_value = row["Market_Price"]  # Fallback to market price

    return max(intrinsic_value, 0)  # Ensure positive intrinsic value

# Function to fetch and process data for each stock
def fetch_and_process(row):
    symbol = row["Symbol"]
    industry = row["Sector"]

    eps, market_price, revenue_per_share, ps_ratio = fetch_data_from_yahoo(symbol)

    if market_price and market_price > 0:
        # Store retrieved data in row for intrinsic value calculation
        row["EPS"] = eps
        row["Market_Price"] = market_price
        row["Revenue_Per_Share"] = revenue_per_share
        row["P/S_Ratio"] = ps_ratio

        intrinsic_value = calculate_intrinsic_value(row)

        return {
            "Symbol": symbol,
            "Intrinsic_Value": intrinsic_value if intrinsic_value else None,
            "Market_Price": market_price,
            "Method": "DCF" if intrinsic_value and intrinsic_value != market_price else "P/S",
            "Last_Updated": datetime.now(),
        }

    print(f"Skipping {symbol}: No valid data available.")
    return None


# Function to fetch all data in parallel
def fetch_all_data_in_parallel(df):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_and_process, df.to_dict("records")))
    return pd.DataFrame([res for res in results if res])

# Function to apply baseline models
def apply_baseline_models(df):
    df = df.dropna(subset=["Intrinsic_Value", "Market_Price"])
    X = df[["Market_Price"]]
    y = df["Intrinsic_Value"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting Machine": GradientBoostingRegressor(random_state=42),
    }

    trained_models = {}
    for model_name, model in models.items():
        model.fit(X, y)
        trained_models[model_name] = model
        df.loc[:, f"{model_name}_Prediction"] = model.predict(X)
        print(f"\n{model_name} Predictions:")
        print(df[["Symbol", "Intrinsic_Value", f"{model_name}_Prediction"]].head(5))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(trained_models, f)

    return df

def main():
    print("Fetching and updating data...")
    sp500_data = pd.read_csv("sp500_tickers.csv")
    sp500_data["Symbol"] = sp500_data["Symbol"].str.replace(".", "-", regex=False)  # Normalize tickers

    updated_data = fetch_all_data_in_parallel(sp500_data)

    print("\nApplying baseline models...")
    updated_data = apply_baseline_models(updated_data)

    # Identify top undervalued and overvalued stocks
    updated_data["Undervalued"] = updated_data["Intrinsic_Value"] > updated_data["Market_Price"]
    updated_data["Overvalued"] = updated_data["Intrinsic_Value"] < updated_data["Market_Price"]

    # Display tickers using DCF and P/S methods
    dcf_tickers = updated_data[updated_data["Method"] == "DCF"]
    ps_tickers = updated_data[updated_data["Method"] == "P/S"]

    print("\nTickers Using DCF Method:")
    print(dcf_tickers[["Symbol", "Intrinsic_Value", "Market_Price", "Undervalued", "Overvalued"]].head(10))

    print("\nTickers Using P/S Method:")
    print(ps_tickers[["Symbol", "Intrinsic_Value", "Market_Price", "Undervalued", "Overvalued"]].head(10))

    print("\nTop 5 Undervalued Stocks:")
    print(updated_data[updated_data["Undervalued"]].sort_values("Intrinsic_Value", ascending=False).head(5))

    print("\nTop 5 Overvalued Stocks:")
    print(updated_data[updated_data["Overvalued"]].sort_values("Intrinsic_Value", ascending=True).head(5))

    # Save results to a CSV file
    csv_file = "sp500_analysis.csv"
    updated_data.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Results saved to {csv_file}.")

if __name__ == "__main__":
    main()
