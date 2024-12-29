"""import yfinance as yf
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

QUANDL_API_KEY = xRWqNvh4jACTQF9y6ckc

# Function to fetch data from Yahoo Finance
def fetch_data_from_yahoo(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # Fetch financial data
        eps = info.get("trailingEps", None)
        market_price = info.get("regularMarketPrice") or info.get("previousClose")
        revenue_per_share = info.get("revenuePerShare", None)
        ps_ratio = info.get("priceToSalesTrailing12Months", None)

        return eps, market_price, revenue_per_share, ps_ratio
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None, None, None, None

# Function to fetch data for multiple tickers with error handling and rate limiting
def fetch_all_data_in_parallel(df):
    def fetch_and_process(row):
        symbol = row["Symbol"]
        industry = row["Sector"]

        # Retry logic for 429 Too Many Requests
        for attempt in range(3):  # Retry up to 5 times
            try:
                eps, market_price, revenue_per_share, ps_ratio = fetch_data_from_yahoo(symbol)
                if market_price and market_price > 0:  # Ensure valid market price
                    return {
                        "Symbol": symbol,
                        "EPS": eps,
                        "Market_Price": market_price,
                        "Revenue_Per_Share": revenue_per_share,
                        "P/S_Ratio": ps_ratio,
                        "Sector": industry,
                        "Last_Updated": datetime.now(),
                    }
                else:
                    print(f"Skipping {symbol}: Invalid Market Price")
                    return None
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limit hit for {symbol}, retrying in 5 seconds...")
                    time.sleep(10)  # Wait before retrying
                else:
                    print(f"Error fetching {symbol}: {e}")
                    return None

        print(f"Failed to fetch {symbol} after multiple attempts.")
        return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_and_process, df.to_dict("records")))

    # Filter out None values and return DataFrame
    return pd.DataFrame([res for res in results if res])

# Main function
if __name__ == "__main__":
    # Load tickers from CSV
    sp500_data = pd.read_csv("sp500_tickers.csv")
    sp500_data["Symbol"] = sp500_data["Symbol"].str.replace(".", "-", regex=False)  # Normalize tickers

    # Fetch data in parallel
    print("Fetching data in parallel with rate limiting...")
    valid_data = fetch_all_data_in_parallel(sp500_data)

    # Output valid tickers and count
    print(f"\nNumber of valid tickers: {len(valid_data)}")
    print("\nSample of Valid Data:")
    print(valid_data.head(10))

    # Save valid tickers to a CSV
    valid_data.to_csv("valid_sp500_tickers_with_intrinsic.csv", index=False)
    print("\nValid tickers saved to 'valid_sp500_tickers_with_intrinsic.csv'")"""
import requests
import pandas as pd
import time
from datetime import datetime

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "PVISE221PEPXLHYB"

# Function to fetch data from Alpha Vantage
def fetch_alpha_vantage_data(symbol):
    try:
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            eps = data.get("EPS", None)
            market_price = data.get("MarketCapitalization", None)  # Placeholder; adjust if needed
            revenue_per_share = data.get("RevenueTTM", None)  # Placeholder; adjust if needed
            ps_ratio = data.get("PriceToSalesRatioTTM", None)  # Placeholder; adjust if needed
            return {
                "Symbol": symbol,
                "EPS": eps,
                "Market_Price": market_price,
                "Revenue_Per_Share": revenue_per_share,
                "P/S_Ratio": ps_ratio,
                "Last_Updated": datetime.now()
            }
        else:
            print(f"Alpha Vantage API error for {symbol}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Load missing tickers
missing_tickers = [
    "ATVI", "AET", "ALXN", "AGN", "ADS", "ABC", "APC", "ANDV", "ANTM", "ARNC",
    "BHGE", "BLL", "BBT", "COG", "CBG", "CBS", "CELG", "CTL", "CERN", "XEC",
    "CTXS", "CXO", "CSRA", "DISCA", "DISCK"
]  # Replace with your missing tickers list

# Limit to 25 tickers (Alpha Vantage daily limit)
selected_tickers = missing_tickers[:25]

# Fetch data for selected tickers
def main():
    results = []
    for symbol in selected_tickers:
        print(f"Fetching data for {symbol}...")
        data = fetch_alpha_vantage_data(symbol)
        if data:
            results.append(data)
        else:
            print(f"Skipping {symbol}: No valid data")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("alpha_vantage_results.csv", index=False)
        print("Results saved to alpha_vantage_results.csv")
    else:
        print("No valid data fetched.")

if __name__ == "__main__":
    main()
