import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import yfinance as yf
from utils.dataloader import TimeSeriesDataset
from torch.utils.data import DataLoader
from models.fast_nn import FactorAugmentedSparseThroughput
from config.config import config
import pandas as pd

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

    # Calculate returns
    returns = np.diff(close_prices, axis=0) / close_prices[:-1]

    # Calculate rolling volatility (e.g., using a 20-day window)
    window = 20
    volatility = pd.Series(returns.flatten()).rolling(window=window).std().values

    # Reshape volatility to match the shape of close_prices
    volatility = volatility.reshape(-1, 1)

    # Ensure that the lengths match by trimming close_prices
    # Since volatility has window-1 fewer elements due to the rolling window, we need to trim close_prices accordingly
    combined_data = np.hstack((close_prices[window:], volatility[window-1:]))

    # Create dataset and dataloader
    dataset = TimeSeriesDataset(combined_data)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Create random matrix for diversified projection
    input_dim = 1  # Single feature (close price)
    dp_mat = np.random.randn(input_dim, config.R_BAR)  # Generate random matrix (p x r_bar)
    print("\nDiversified projection matrix shape:", dp_mat.shape)
    print("Random matrix values:\n", dp_mat)

    # Initialize Fast-NN model
    model = FactorAugmentedSparseThroughput(
        input_dim=config.INPUT_DIM,
        r_bar=config.R_BAR,
        width=config.WIDTH,
        dp_mat=dp_mat,
        sparsity=None,
        rs_mat=None
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