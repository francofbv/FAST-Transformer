import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import yfinance as yf
from utils.dataloader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.fast_nn import FactorAugmentedSparseThroughput
from config.config import config

def main():
    # Get some sample data (either real or dummy)
    try:
        data = yf.download("NVDA", start="2022-01-01", end="2023-12-31")
        if data.empty:
            raise ValueError("No data downloaded from yfinance")
        
        close_prices = data['Close'].values.reshape(-1, 1)
        print(f"Downloaded {len(close_prices)} days of NVDA data")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Using dummy data instead")
        close_prices = np.random.randn(100, 1) * 10 + 100  # Random prices around 100

    # Create dataset and dataloader
    dataset = TimeSeriesDataset(close_prices)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize Fast-NN model
    model = FactorAugmentedSparseThroughput(
        input_dim=1,  # Single feature (close price)
        r_bar=config.R_BAR,
        width=config.WIDTH if hasattr(config, 'WIDTH') else 32  # Default width if not in config
    )

    # Set model to eval mode
    model.eval()

    # Get a single batch of data
    X_batch, _ = next(iter(dataloader))
    print("\nInput shape:", X_batch.shape)

    # Forward pass through Fast-NN
    with torch.no_grad():
        x1, x2 = model(X_batch, is_training=False)
        print("\nFast-NN Output shapes:")
        print("x1 shape (factor output):", x1.shape)
        print("x2 shape (sparse output):", x2.shape)
        print("\nCombined output shape if concatenated:", 
              torch.cat([x1, x2], dim=-1).shape)

if __name__ == "__main__":
    main() 