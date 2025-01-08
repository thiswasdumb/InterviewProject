import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime
from torch.utils.data import DataLoader, TensorDataset
import joblib


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
input_dim = 6  # Number of features (ensure the input matches this)
hidden_dim = 200
output_dim = 1
num_layers = 2
dropout = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IndustryStockPriceLSTM(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
model.load_state_dict(torch.load("it_lstm_model.pth", map_location=device))
model.eval()

import joblib

# Load scaler for consistent scaling
try:
    scaler = joblib.load("it_scaler.pkl")
except FileNotFoundError:
    print("Scaler file not found. Using a new scaler.")
    scaler = MinMaxScaler()
    joblib.dump(scaler, 'it_scaler.pkl')


# Function to prepare data for LSTM
def prepare_lstm_data(data, sequence_length, scaler):
    data.loc[:, "Date"] = pd.to_datetime(data["Date"], errors="coerce")  # Convert to datetime
    data = data.dropna(subset=["Date"])  # Drop rows with invalid dates
    data = data.sort_values(by="Date").reset_index(drop=True)
    data = data.fillna(method="ffill").fillna(method="bfill")  # Handle missing values

    data["Dummy"] = 0  # Placeholder column to ensure input_dim matches model
    numerical_cols = ["Open", "High", "Low", "Close", "Volume", "Dummy"]

    # Scale numerical columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length][numerical_cols].values
        target = data.iloc[i+sequence_length]["Close"]
        sequences.append(sequence)
        targets.append(target)

    return np.array(sequences), np.array(targets)

# Function to update the model
def update_model(model, data, scaler, sequence_length, learning_rate, num_epochs):
    X, y = prepare_lstm_data(data, sequence_length, scaler)
    
    # Convert to PyTorch tensors
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Training setup
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (sequences, targets) in enumerate(dataloader):
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss/len(dataloader):.4f}")

# Load last recorded date and historical data
try:
    predictions_df = pd.read_csv("it_industry_predictions.csv")
    last_date = predictions_df["Last Updated"].max()
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()
except (FileNotFoundError, KeyError):
    last_date = datetime.date.today() - datetime.timedelta(days=5)  # Default to 5 days ago

print(f"Last recorded date: {last_date}")

# Load ticker and sector info
sp500_info = pd.read_csv("sp500_tickers.csv")
raw_data = pd.read_csv("sp500_historical_data.csv")
industry = "Information Technology"

# Filter for tickers in the selected industry
tickers_in_industry = sp500_info[sp500_info["Sector"] == industry]["Symbol"]
new_data = []

# Fixed predict_future_close function
def predict_future_close(data, scaler, model):
    """
    Predict the next close price (scaled and then converted to the original scale).
    """
    recent_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict the scaled close price
    with torch.no_grad():
        prediction_scaled = model(recent_data).item()

    # Inverse transform the predicted value back to the original scale
    prediction_original = scaler.inverse_transform([[0, 0, 0, prediction_scaled, 0, 0]])[0, 3]
    return prediction_original

print(f"Fetching new data for tickers in {industry}...")

for i, ticker in enumerate(tickers_in_industry):
    try:
        ticker_data = yf.Ticker(ticker).history(start=str(last_date))
        ticker_data.reset_index(inplace=True)
        ticker_data["Ticker"] = ticker
        new_data.append(ticker_data)
        print(f"Fetched data for ticker {ticker} ({i+1}/{len(tickers_in_industry)})")
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")

if new_data:
    new_data = pd.concat(new_data)
    new_data.loc[:, "Date"] = pd.to_datetime(new_data["Date"], errors="coerce")  # Ensure consistent date format
    new_data = new_data.dropna(subset=["Date"])
    raw_data = pd.concat([raw_data, new_data])

    # Update the model with new data
    industry_data = raw_data[raw_data["Ticker"].isin(tickers_in_industry)]
    print("Updating the model with new data...")
    update_model(model, industry_data, scaler, sequence_length=60, learning_rate=0.0005, num_epochs=5)

    # Save updated predictions
    results = []
    for ticker in tickers_in_industry:
        ticker_data = industry_data[industry_data["Ticker"] == ticker]
        if len(ticker_data) < 60:
            continue

            # Ensure consistent features (including the Dummy column) in recent data for prediction
        recent_data = ticker_data.iloc[-60:][["Open", "High", "Low", "Close", "Volume"]].values
        # Add the Dummy column
        dummy_col = np.zeros((recent_data.shape[0], 1))  # Placeholder column
        recent_data = np.hstack((recent_data, dummy_col))  # Add dummy column to match input_dim

        # Scale recent data
        recent_data = scaler.transform(recent_data)

        # Predict the next close price
        next_close = predict_future_close(recent_data, scaler, model)

        results.append({"Ticker": ticker, "Actual Close": ticker_data.iloc[-1]["Close"], "Predicted Close": next_close, "Last Updated": str(datetime.date.today())})

    results_df = pd.DataFrame(results)
    results_df.to_csv("it_industry_predictions.csv", index=False)
    print("Predictions updated and saved.")
else:
    print("No new data to update.")
