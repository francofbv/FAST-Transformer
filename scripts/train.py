import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import torch.optim as optim
import numpy as np
from models.transformer import TimeSeriesTransformer
from config.config import config
import yfinance as yf
from utils.dataloader import TimeSeriesDataset
from tqdm import tqdm
from evaluate import evaluate

try:
    # Download more recent data
    data = yf.download("NVDA", start="2022-01-01", end="2023-12-31")
    if data.empty:
        raise ValueError("No data downloaded from yfinance")
    
    close_prices = data['Close'].values.reshape(-1, 1)
    print(f"Downloaded {len(close_prices)} days of NVDA data")
    
except Exception as e:
    print(f"Error downloading data: {e}")
    # Create some dummy data for testing
    print("Using dummy data instead")
    close_prices = np.random.randn(100, 1) * 10 + 100  # Random prices around 100

# Split data into train and test
train_size = int(len(close_prices) * 0.8)
train_data = close_prices[:train_size]
test_data = close_prices[train_size:]

# Create datasets
train_dataset = TimeSeriesDataset(train_data)
test_dataset = TimeSeriesDataset(test_data)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = TimeSeriesTransformer()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
for epoch in tqdm(range(config.NUM_EPOCHS), desc="Training Progress"):
    model.train()
    epoch_loss = 0
    
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False):
        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch.squeeze())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}")

# Save model weights and scaler
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler
}, 'checkpoints/model.pth')

# Evaluate the model
print("\nEvaluating model on test set...")
mae, rmse, r2 = evaluate(model, test_loader, train_dataset.scaler)