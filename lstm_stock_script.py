import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load feature-engineered stock data
stock_data = pd.read_csv("feature_engineered_stock_data.csv")

# Initialize scalers and encoders
scaler = MinMaxScaler()
ticker_encoder = LabelEncoder()

def prepare_lstm_data(data, sequence_length=60):
    """
    Prepare data for LSTM model training.

    Parameters:
        data (pd.DataFrame): Feature-engineered stock data.
        sequence_length (int): Number of time steps in each sequence.

    Returns:
        X (np.array): Input sequences.
        y (np.array): Target values (next Close price).
    """
    # Encode Ticker as categorical variable
    data["Ticker"] = ticker_encoder.fit_transform(data["Ticker"])

    # Sort data by Ticker and Date
    data = data.sort_values(by=["Ticker", "Date"])

    # Handle missing values with forward fill
    data = data.fillna(method="ffill")

    # Create sequences
    sequences = []
    targets = []
    tickers = data["Ticker"].unique()

    for ticker in tickers:
        ticker_data = data[data["Ticker"] == ticker]
        for i in range(len(ticker_data) - sequence_length):
            sequence = ticker_data.iloc[i:i+sequence_length].drop(columns=["Date", "Ticker"]).values
            target = ticker_data.iloc[i+sequence_length]["Close"]
            sequences.append(sequence)
            targets.append(target)

    return np.array(sequences), np.array(targets)

# Prepare data for LSTM
sequence_length = 60  # Example sequence length
X, y = prepare_lstm_data(stock_data, sequence_length=sequence_length)

# Save prepared data for training
np.save("lstm_X.npy", X)
np.save("lstm_y.npy", y)

print("LSTM data preparation complete!")
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Define the LSTM Model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(StockPriceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Only take the last time step's output
        return out

# Training Function
def train_model(train_loader, model, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

# Hyperparameters
input_dim = X.shape[2]  # Automatically infer number of features from data
hidden_dim = 50
output_dim = 1  # Predicting one value (e.g., Close price)
num_layers = 2
dropout = 0.2
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Convert Data to PyTorch Dataloader
def create_dataloader(X, y):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load Data for Training
X_train = np.load("lstm_X.npy")
y_train = np.load("lstm_y.npy")
train_loader = create_dataloader(X_train, y_train)

# Initialize Model, Criterion, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockPriceLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model
train_model(train_loader, model, criterion, optimizer, num_epochs)

# Save the Model
torch.save(model.state_dict(), "lstm_stock_model.pth")
print("Model saved to 'lstm_stock_model.pth'")
