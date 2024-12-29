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

# API Keys
# Finnhub API Key
FINNHUB_API_KEY = "cthpu71r01qm2t954n3gcthpu71r01qm2t954n40"

# Database setup
DB_FILE = "sp500_analysis.db"
MODEL_FILE = "baseline_models.pkl"
engine = create_engine(f"sqlite:///{DB_FILE}")

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

# Function to fetch data from Finnhub API
def fetch_data_from_finnhub(symbol):
    try:
        metric_response = requests.get(
            f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_API_KEY}"
        )
        metric_data = metric_response.json()
        eps = metric_data.get("metric", {}).get("epsTTM", None)
        revenue_per_share = metric_data.get("metric", {}).get("revenuePerShareTTM", None)
        ps_ratio = metric_data.get("metric", {}).get("psTTM", None)

        quote_response = requests.get(
            f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        )
        quote_data = quote_response.json()
        market_price = quote_data.get("c", None)

        return eps, market_price, revenue_per_share, ps_ratio
    except Exception as e:
        print(f"Finnhub error for {symbol}: {e}")
        return None, None, None, None

# Function to calculate DCF-based intrinsic value
def calculate_dcf(eps, growth_rate, discount_rate, years=10):
    intrinsic_value = 0
    for year in range(1, years + 1):
        projected_eps = eps * (1 + growth_rate / 100) ** year
        discounted_value = projected_eps / (1 + discount_rate / 100) ** year
        intrinsic_value += discounted_value
    return intrinsic_value

# Function to calculate P/S-based intrinsic value
def calculate_ps_valuation(revenue_per_share, ps_ratio):
    return revenue_per_share * ps_ratio

# Function to fetch and process data
def fetch_and_process(row):
    symbol = row["Symbol"]
    industry = row["Sector"]

    if symbol in ["CHK", "GPS"]:  # Use Finnhub for specific tickers
        eps, market_price, revenue_per_share, ps_ratio = fetch_data_from_finnhub(symbol)
    else:  # Use Yahoo Finance for others
        eps, market_price, revenue_per_share, ps_ratio = fetch_data_from_yahoo(symbol)

    if market_price and market_price > 0:
        if eps and eps > 0.01:  # Use DCF for positive EPS
            rates = industry_rates.get(industry, {"growth_rate": 8, "discount_rate": 10})
            intrinsic_value = calculate_dcf(eps, rates["growth_rate"], rates["discount_rate"])
            method = "DCF"
        elif revenue_per_share and revenue_per_share > 0 and ps_ratio and ps_ratio > 0:  # Use P/S ratio
            intrinsic_value = calculate_ps_valuation(revenue_per_share, ps_ratio)
            method = "P/S"
        else:
            intrinsic_value, method = None, "None"

        return {
            "Symbol": symbol,
            "Intrinsic_Value": intrinsic_value,
            "Market_Price": market_price,
            "Method": method,
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

# Main execution
def main():
    print("Fetching and updating data...")
    sp500_data = pd.read_csv("sp500_tickers.csv")
    sp500_data["Symbol"] = sp500_data["Symbol"].str.replace(".", "-", regex=False)  # Normalize tickers

    updated_data = fetch_all_data_in_parallel(sp500_data)

    print("\nApplying baseline models...")
    updated_data = apply_baseline_models(updated_data)

    # Identify top undervalued and overvalued stocks
    updated_data.loc[:, "Undervalued"] = updated_data["Intrinsic_Value"] > updated_data["Market_Price"]
    updated_data.loc[:, "Overvalued"] = updated_data["Intrinsic_Value"] < updated_data["Market_Price"]

     # Display tickers using DCF and P/S methods
    dcf_tickers = updated_data[updated_data["Method"] == "DCF"]
    ps_tickers = updated_data[updated_data["Method"] == "P/S"]

    print("\nTickers Using DCF Method:")
    print(dcf_tickers[["Symbol", "Intrinsic_Value", "Market_Price", "Undervalued", "Overvalued"]].head(10))

    print("\nTickers Using P/S Method:")
    print(ps_tickers[["Symbol", "Intrinsic_Value", "Market_Price", "Undervalued", "Overvalued"]].head(10))

    print("\nTop 5 Undervalued Stocks:")
    print(updated_data.loc[updated_data["Undervalued"]].sort_values("Intrinsic_Value", ascending=False).head(5))

    print("\nTop 5 Overvalued Stocks:")
    print(updated_data.loc[updated_data["Overvalued"]].sort_values("Intrinsic_Value", ascending=True).head(5))

    # Save results to database
    updated_data.to_sql("sp500_analysis", engine, if_exists="replace", index=False)
    print("Results saved to database.")

    with open("cron_log.txt", "a") as log_file:
        log_file.write(f"Script executed at {datetime.now()}\n")


if __name__ == "__main__":
    main()
