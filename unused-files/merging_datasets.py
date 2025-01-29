import pandas as pd

def merge_datasets(historical_file, updated_file, output_file):
    """
    Merge historical data with updated stock data including industries.

    :param historical_file: Path to the historical data CSV.
    :param updated_file: Path to the updated stock data CSV.
    :param output_file: Path to save the merged CSV.
    """
    # Load the datasets
    historical_data = pd.read_csv(historical_file)
    updated_data = pd.read_csv(updated_file)

    # Ensure 'Ticker' column is present in both datasets
    historical_data.rename(columns={"Ticker": "Symbol"}, inplace=True)
    updated_data.rename(columns={"Ticker": "Symbol"}, inplace=True)

    # Fill missing Industry data in the updated dataset
    industry_mapping = updated_data.set_index("Symbol")["Industry"].to_dict()
    historical_data["Industry"] = historical_data["Symbol"].map(industry_mapping)

    # Merge datasets on 'Symbol' column
    merged_data = pd.merge(historical_data, updated_data, on="Symbol", how="outer")

    # Save the merged dataset
    merged_data.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}.")

# Example usage
if __name__ == "__main__":
    merge_datasets(
        historical_file="sp500_historical_data.csv",
        updated_file="updated_stock_data_with_industry.csv",
        output_file="merged_sp500_data.csv"
    )
