import yfinance as yf
import pandas as pd
import time
from datetime import datetime

# Input and output files
INPUT_FILE = "valid_sp500_tickers_with_intrinsic.csv"
OUTPUT_FILE = "comprehensive_historical_data.csv"

# Fields to collect
FIELDS = [
    "trailingEps",
    "priceToEarningsRatio",
    "priceToSalesTrailing12Months",
    "revenuePerShare",
    "totalRevenue",
    "grossProfit",
    "operatingIncome",
    "netIncome",
    "earningsDate",
    "regularMarketPrice",
]

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

import pandas as pd

# Define the industry rates (same as in your script)
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

# Load the CSV file with historical data
csv_file = "comprehensive_historical_data.csv"  # Replace with your CSV file
output_file = "updated_stock_data_with_industry.csv"

# Load the S&P 500 ticker data (contains ticker-to-industry mapping)
sp500_ticker_file = "valid_sp500_tickers_with_intrinsic.csv"  # Replace with your ticker-to-industry file
ticker_data = pd.read_csv(sp500_ticker_file)

# Create a dictionary to map tickers to industries
ticker_to_industry = pd.Series(ticker_data["Sector"].values, index=ticker_data["Symbol"]).to_dict()

# Read the historical stock data
stock_data = pd.read_csv(csv_file)

# Map industry to each ticker
stock_data["Industry"] = stock_data["Symbol"].map(ticker_to_industry)

# Map discount rate based on industry
stock_data["Discount_Rate"] = stock_data["Industry"].map(
    lambda x: industry_rates.get(x, {}).get("discount_rate", None)
)

# Save the updated DataFrame
stock_data.to_csv(output_file, index=False)

print(f"Updated stock data with industry and discount rate saved to '{output_file}'.")


# Below was used for gathering the data, then i needed to add industry data and discount and growth rates 
"""
# Function to fetch historical data for a ticker
def fetch_historical_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        historical_data = []

        # Income statement and fundamental metrics
        income_statement = stock.financials.T if stock.financials is not None else pd.DataFrame()
        if income_statement.empty:
            print(f"No financial data for {symbol}")
            return []

        for date, data in income_statement.iterrows():
            record = {
                "Symbol": symbol,
                "Date": date,
                "Total_Revenue": data.get("Total Revenue", None),
                "Gross_Profit": data.get("Gross Profit", None),
                "Operating_Income": data.get("Operating Income", None),
                "Net_Income": data.get("Net Income", None),
                "EPS": data.get("EPS", None),
                "Cost_of_Revenue": data.get("Cost of Revenue", None),
                "R&D_Expenses": data.get("Research and Development", None),
                "SG&A_Expenses": data.get("Selling General and Administrative", None),
                "Tax_Provision": data.get("Tax Provision", None),
                "Operating_Expenses": data.get("Operating Expenses", None),
                "Market_Price": stock.info.get("regularMarketPrice", None),
            }
            historical_data.append(record)

        return historical_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return []

# Main function
def main():
    # Read tickers
    sp500_data = pd.read_csv(INPUT_FILE)
    tickers = sp500_data["Symbol"].tolist()

    # Collect data in batches
    all_data = []
    batch_size = 10

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{len(tickers) // batch_size + 1}")

        for ticker in batch:
            print(f"Fetching data for {ticker}")
            data = fetch_historical_data(ticker)
            all_data.extend(data)

        # Write data incrementally to avoid data loss
        if all_data:
            pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)

        # Respect API limits
        print("Sleeping for 10 seconds to avoid rate limits...")
        time.sleep(10)

    print("Data collection complete.")

if __name__ == "__main__":
    main()
"""