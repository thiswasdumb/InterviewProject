import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Initialize scalers
scaler = MinMaxScaler()
encoder = LabelEncoder()

# Preprocessing for Dataset 1: Historical Stock Data
def preprocess_historical_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Ticker", "Date"])
    numerical_cols = ["Open", "High", "Low", "Close", "Volume"]
    
    # Normalize numerical columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("Preprocessed Historical Data Sample:")
    print(df.head())
    return df

# Preprocessing for Dataset 2: Fundamental Stock Data
def preprocess_fundamental_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Industry"] = encoder.fit_transform(df["Industry"])
    
    # Drop rows where more than 50% of the columns are missing
    df = df.dropna(thresh=int(df.shape[1] * 0.5), axis=0)
    
    # Fill remaining missing values with column means
    numerical_cols = [
        "Total_Revenue", "Gross_Profit", "Operating_Income", "Net_Income", "EPS",
        "Cost_of_Revenue", "R&D_Expenses", "SG&A_Expenses", "Tax_Provision",
        "Operating_Expenses", "Market_Price"
    ]
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Preprocessed Fundamental Data Sample:")
    print(df.head())
    return df

# Load datasets
historical_data = pd.read_csv("sp500_historical_data.csv")
fundamental_data = pd.read_csv("updated_stock_data_with_industry.csv")

# Apply preprocessing
preprocessed_historical_data = preprocess_historical_data(historical_data)
preprocessed_fundamental_data = preprocess_fundamental_data(fundamental_data)

# Save results
preprocessed_historical_data.to_csv("preprocessed_historical_data.csv", index=False)
preprocessed_fundamental_data.to_csv("preprocessed_fundamental_data.csv", index=False)
