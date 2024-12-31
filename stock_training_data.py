import yfinance as yf
import pandas as pd
import time

# Function to fetch historical data for a single ticker
def fetch_historical_data(ticker, start_date="2010-01-01", end_date=None):
    """
    Fetch historical stock data for a given ticker.
    
    :param ticker: Stock ticker symbol
    :param start_date: Start date for historical data
    :param end_date: End date for historical data (default is None for today)
    :return: DataFrame with historical data
    """
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(start=start_date, end=end_date)
        historical_data["Ticker"] = ticker  # Add ticker column for reference
        return historical_data.reset_index()  # Reset index to include the Date column
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Main function to fetch data for multiple tickers
def fetch_historical_data_for_tickers(ticker_list, output_file="historical_data.csv", batch_size=10, sleep_time=30):
    """
    Fetch historical data for multiple tickers and save to a CSV file.
    
    :param ticker_list: List of stock tickers
    :param output_file: Name of the CSV file to save data
    :param batch_size: Number of tickers to process per batch to avoid rate limits
    :param sleep_time: Sleep time (in seconds) between batches
    """
    all_data = []

    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(ticker_list) // batch_size + 1}...")

        for ticker in batch:
            print(f"Fetching historical data for {ticker}...")
            data = fetch_historical_data(ticker)
            if not data.empty:
                all_data.append(data)

        print(f"Completed batch {i // batch_size + 1}. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

    # Combine all data and save to CSV
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(output_file, index=False)
        print(f"Historical data saved to {output_file}.")
    else:
        print("No data fetched.")

# Example usage
if __name__ == "__main__":
    # Replace with your list of valid S&P 500 tickers
    tickers = pd.read_csv("valid_sp500_tickers_with_intrinsic.csv")["Symbol"].tolist()
    
    # Fetch and save historical data
    fetch_historical_data_for_tickers(
        ticker_list=tickers,
        output_file="sp500_historical_data.csv",
        batch_size=10,
        sleep_time=10
    )
