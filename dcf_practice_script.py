import pandas as pd
import yfinance as yf

# Load the tickers and sectors from the CSV file
sp500_tickers = pd.read_csv("sp500_tickers.csv")

# Define industry rates
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

# Prepare the new dataframe to store results
columns = ["Year", "Ticker", "Industry", "EPS", "Growth Rate", "Discount Rate", "Intrinsic Value"]
result_df = pd.DataFrame(columns=columns)

projection_years = 1  # Number of years for DCF projection

# Iterate over each ticker in the CSV
for index, row in sp500_tickers.iterrows():
    ticker = row["Symbol"]
    industry = row["Sector"]

    # Retrieve financial data from Yahoo Finance
    try:
        stock = yf.Ticker(ticker)

        # Fetch historical income statement and balance sheet
        income_statement = stock.financials
        balance_sheet = stock.balance_sheet

        # Check for required data
        if "Net Income" in income_statement.index and "Ordinary Shares Number" in balance_sheet.index:
            net_income = income_statement.loc["Net Income"]
            shares_outstanding = balance_sheet.loc["Ordinary Shares Number"]

            # Calculate historical EPS
            eps_values = (net_income / shares_outstanding).dropna()

            # Filter out negative EPS values
            eps_values = eps_values[eps_values > 0]

            # Calculate DCF intrinsic value for each available EPS
            new_rows = []
            for year, eps in eps_values.items():
                growth_rate = industry_rates[industry]["growth_rate"]
                discount_rate = industry_rates[industry]["discount_rate"]
                intrinsic_value = 0

                for future_year in range(1, projection_years + 1):
                    projected_eps = eps * (1 + growth_rate / 100) ** future_year
                    discounted_value = projected_eps / (1 + discount_rate / 100) ** future_year
                    intrinsic_value += discounted_value

                # Create a new row
                new_rows.append({
                    "Year": year,
                    "Ticker": ticker,
                    "Industry": industry,
                    "EPS": eps,
                    "Growth Rate": growth_rate,
                    "Discount Rate": discount_rate,
                    "Intrinsic Value": intrinsic_value,
                })

            # Append new rows to the dataframe
            result_df = pd.concat([result_df, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            print(f"Error: Could not fetch historical EPS values for {ticker}.")

    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")

# Sort the results by Ticker and Year in ascending order
result_df["Year"] = pd.to_datetime(result_df["Year"])
result_df = result_df.sort_values(by=["Ticker", "Year"], ascending=[True, True])

# Save the result to a new CSV file
result_df.to_csv("sp500_eps_data.csv", index=False)
print("Data collection complete. Saved to sp500_eps_data.csv")
