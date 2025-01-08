import datetime
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Define the LSTM Model
class IndustryStockPriceLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(IndustryStockPriceLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only take the last time step's output
        return out

# Load the trained model
input_dim = 6  # Number of features
hidden_dim = 200
output_dim = 1
num_layers = 2
dropout = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IndustryStockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
model.load_state_dict(torch.load("it_lstm_model.pth", map_location=device))
model.eval()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fixed predict_future_close function
def predict_future_close(data, scaler, model):
    """
    Predict the next close price (scaled and then converted to the original scale).
    """
    # Add a dummy Ticker_Encoded feature (e.g., 0) for prediction
    ticker_encoded_dummy = np.zeros((data.shape[0], 1))
    data_with_ticker = np.hstack((data, ticker_encoded_dummy))

    # Convert to PyTorch tensor
    recent_data = torch.tensor(data_with_ticker, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict the scaled close price
    with torch.no_grad():
        prediction_scaled = model(recent_data).item()

    # Inverse transform the predicted value back to the original scale
    prediction_original = scaler.inverse_transform([[0, 0, 0, prediction_scaled, 0]])[0, 3]
    return prediction_original

# Load the dataset
sp500_info = pd.read_csv("sp500_tickers.csv")  # Assumes this file contains Tickers and Sectors
raw_data = pd.read_csv("sp500_historical_data.csv")
industry = "Information Technology"

# Filter for tickers in the selected industry
tickers_in_industry = sp500_info[sp500_info["Sector"] == industry]["Symbol"]
industry_data = raw_data[raw_data["Ticker"].isin(tickers_in_industry)].copy()

# Scale the data
numerical_cols = ["Open", "High", "Low", "Close", "Volume"]
industry_data[numerical_cols] = scaler.fit_transform(industry_data[numerical_cols])

# Predict the next close price for each ticker
results = []
sequence_length = 60

for ticker in tickers_in_industry:
    ticker_data = industry_data[industry_data["Ticker"] == ticker]
    if len(ticker_data) < sequence_length:
        continue

    # Get the most recent data for the ticker
    recent_data = ticker_data.iloc[-sequence_length:][numerical_cols].values

    # Predict the next close price
    next_close = predict_future_close(recent_data, scaler, model)

    # Fetch the actual close price from Yahoo Finance
    try:
        actual_close = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        actual_close = None
        print(f"Could not fetch actual close for {ticker}: {e}")

    results.append({"Ticker": ticker, "Predicted Close": next_close, "Actual Close": actual_close, "Date Updated": str(datetime.date.today())})

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

joblib.dump(scaler, "scaler.pkl")

# Save results to a CSV file
results_df.to_csv("it_industry_predictions.csv", index=False)
print("Predictions saved to 'it_industry_predictions.csv'")
