import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Load the raw data and filter for the industry
sp500_info = pd.read_csv("sp500_tickers.csv")  # Assumes this file contains Tickers and Sectors
raw_data = pd.read_csv("sp500_historical_data.csv")
industry = "Information Technology"

# Filter for tickers in the selected industry
tickers_in_industry = sp500_info[sp500_info["Sector"] == industry]["Symbol"]
industry_data = raw_data[raw_data["Ticker"].isin(tickers_in_industry)].copy()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Prepare LSTM Data
def prepare_lstm_data(data, sequence_length=60, scaler=None):
    data = data.sort_values(by=["Ticker", "Date"])
    data = data.fillna(method="ffill").fillna(method="bfill")  # Handle missing values

    # Scale numerical columns
    numerical_cols = ["Open", "High", "Low", "Close", "Volume"]
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Encode ticker as a feature
    data["Ticker_Encoded"] = data["Ticker"].astype("category").cat.codes

    # Prepare sequences
    sequences, targets = [], []
    for ticker in data["Ticker"].unique():
        ticker_data = data[data["Ticker"] == ticker]
        for i in range(len(ticker_data) - sequence_length):
            sequence = ticker_data.iloc[i:i+sequence_length][numerical_cols + ["Ticker_Encoded"]].values
            target = ticker_data.iloc[i+sequence_length]["Close"]
            sequences.append(sequence)
            targets.append(target)

    return np.array(sequences), np.array(targets), scaler

# Prepare the data
sequence_length = 60
X, y, scaler = prepare_lstm_data(industry_data, sequence_length, scaler)

# Split into training and validation
split_idx = int(0.8 * len(X))  # 80% training, 20% validation
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

# Convert data to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the LSTM Model
class IndustryStockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(IndustryStockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only take the last time step's output
        return out

# Hyperparameters
input_dim = X.shape[2]
hidden_dim = 200
output_dim = 1
num_layers = 2
dropout = 0.3
learning_rate = 0.0005
num_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, and optimizer
model = IndustryStockPriceLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader):.4f}")

train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs)

# Save the industry model
torch.save(model.state_dict(), f"{industry}_lstm_model.pth")
print(f"Model for {industry} saved as {industry}_lstm_model.pth")

# Evaluate the model for all tickers in the industry
model.eval()
def predict_future_close(data, scaler, model):
    """
    Predict the next close price (scaled and then converted to original scale).
    """
    recent_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction_scaled = model(recent_data).item()

    # Inverse transform the predicted value back to the original scale
    prediction_original = scaler.inverse_transform([[0, 0, 0, prediction_scaled, 0]])[0, 3]
    return prediction_original

results = []
for ticker in tickers_in_industry:
    ticker_data = industry_data[industry_data["Ticker"] == ticker]
    if len(ticker_data) < sequence_length:
        continue

    recent_close = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    recent_data = ticker_data.iloc[-sequence_length:][["Open", "High", "Low", "Close", "Volume"]].values
    recent_data = scaler.transform(recent_data)
    next_close = predict_future_close(recent_data, scaler, model)

    results.append({"Ticker": ticker, "Actual Close": recent_close, "Predicted Close": next_close})

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# Save results
results_df.to_csv("industry_predictions.csv", index=False)
