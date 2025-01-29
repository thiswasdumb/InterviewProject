import pandas as pd
from datetime import datetime

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

def calculate_intrinsic_value(row):
    # Extract the sector-specific growth and discount rates
    sector = row["Sector"]
    rates = industry_rates.get(sector, None)

    if rates is None:
        return None  # Skip calculation if sector is not in the dictionary

    growth_rate = rates["growth_rate"] / 100  # Convert to decimal
    discount_rate = rates["discount_rate"] / 100  # Convert to decimal

    # Calculate next year's EPS
    next_year_eps = row["EPS"] * (1 + growth_rate)

    # Safeguard for small denominator
    denominator = discount_rate - growth_rate
    if denominator > 0.01:  # Ensure denominator is not too small
        intrinsic_value = next_year_eps / denominator
    else:
        # Use alternative one-year forward DCF approach
        intrinsic_value = next_year_eps / (1 + discount_rate)

    # If intrinsic value is still invalid, use P/S ratio approach
    if intrinsic_value <= 0 or pd.isna(intrinsic_value):
        if row["P/S_Ratio"] > 0 and row["Revenue_Per_Share"] > 0:
            intrinsic_value = row["P/S_Ratio"] * row["Revenue_Per_Share"]
        else:
            # Fallback to market price if all else fails
            intrinsic_value = row["Market_Price"]

    return max(intrinsic_value, 0)  # Ensure positive intrinsic value


# Load the CSV file
file_path = "valid_sp500_tickers_with_intrinsic.csv"
data = pd.read_csv(file_path)

# Calculate intrinsic values
results = []
for _, row in data.iterrows():
    intrinsic_value = calculate_intrinsic_value(row)
    if intrinsic_value is not None:
        results.append({
            "Ticker": row["Symbol"],
            "Intrinsic Value": round(intrinsic_value, 2),
            "Date Calculated": datetime.now().strftime("%Y-%m-%d")
        })

# Convert results to DataFrame and save to a new CSV file
results_df = pd.DataFrame(results)
output_file = "intrinsic_values_calculated.csv"
results_df.to_csv(output_file, index=False)

print(f"Intrinsic values saved to {output_file}")
