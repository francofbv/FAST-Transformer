import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import torch.optim as optim
import numpy as np
from scipy.sparse.linalg import eigsh
import argparse
from models.transformer import TimeSeriesTransformer
from models.fastnn_transformer import FastNNTransformer
from config.config import config
import yfinance as yf
from utils.dataloader import TimeSeriesDataset
from tqdm import tqdm
from evaluate import evaluate
import logging
from datetime import datetime
import shutil
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a time series model')
parser.add_argument('--fast_nn', action='store_true', help='Use Fast-NN Transformer model instead of Transformer')
args = parser.parse_args()

try:
    # Download more recent data
    data = yf.download("NVDA", start="2019-01-01", end="2024-12-31")
    if data.empty:
        raise ValueError("No data downloaded from yfinance")
    
    close_prices = data['Close'].values.reshape(-1, 1)
    print(f"Downloaded {len(close_prices)} days of NVDA data")
    
except Exception as e:
    print(f"Error downloading data: {e}")
    # Create some dummy data for testing
    print("Using dummy data instead")
    close_prices = np.random.randn(100, 1) * 10 + 100  # Random prices around 100

# Calculate returns
returns = np.diff(close_prices, axis=0) / close_prices[:-1]

# Calculate rolling volatility (e.g., using a 20-day window)
window = 20
volatility = pd.Series(returns.flatten()).rolling(window=window).std().values

# Reshape volatility to match the shape of close_prices
volatility = volatility.reshape(-1, 1)

# Print shapes for debugging
print(f"close_prices shape: {close_prices.shape}")
print(f"returns shape: {returns.shape}")
print(f"volatility shape: {volatility.shape}")

# Combine close_prices and volatility into a new dataset
# Adjust indices to ensure matching lengths
combined_data = np.hstack((close_prices[window:], volatility[window-1:]))

print(f"combined_data shape: {combined_data.shape}")

# Now combined_data has shape (n_samples, 2), where the first column is close prices and the second is volatility

# Split data into train and test
train_size = int(len(combined_data) * 0.8)
train_data = combined_data[:train_size]
test_data = combined_data[train_size:]

# Create datasets
train_dataset = TimeSeriesDataset(train_data)
test_dataset = TimeSeriesDataset(test_data)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

def compute_dp_mat(x, r_bar=config.R_BAR):
    print(x.shape)
    p = np.shape(x)[1]
    covariance_matrix = x.T @ x
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)
    #print(dp_matrix)
    #print(dp_matrix.shape)
    #exit()

    return dp_matrix
    
# Initialize model based on command line argument
if args.fast_nn: 
    dp_mat = compute_dp_mat(train_data)
    model = FastNNTransformer(
        r_bar=config.R_BAR,
        width=config.WIDTH,
        dp_mat=dp_mat,
        sparsity=None,
        rs_mat=None
    )
    logging.info("Using Fast-NN Transformer model")
else: 
    model = TimeSeriesTransformer()
    logging.info("Using standard Transformer model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logging.info(f"Using device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)



def train(data_loader, model, criterion, optimizer, reg_lambda, reg_tau, device):
    model.train()
    loss_dict = {'l2_loss': 0.0} # initialize l2_loss (MSE) to 0

    if reg_tau: loss_dict['reg_loss'] = 0.0 # if we're using regularization, initialize reg_loss to 0

    
    progress_bar = tqdm(data_loader, desc='Training')
    for batch, (X_batch, y_batch) in enumerate(progress_bar):
        # Move data to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        pred = model(X_batch).squeeze()
        
        loss = criterion(pred, y_batch.squeeze())
        loss_dict['l2_loss'] += loss.item()

        if reg_tau and args.fast_nn:
            reg_loss = model.regularization_loss(model=model, tau=reg_tau)
            loss_dict['reg_loss'] += reg_lambda * reg_loss.item()
            loss += reg_loss * reg_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'l2_loss': f"{loss_dict['l2_loss']/(batch+1):.4f}",
            'reg_loss': f"{loss_dict['reg_loss']/(batch+1):.4f}" if reg_tau else "N/A"
        })
    
    loss_dict['l2_loss'] /= len(data_loader)
    if reg_tau: loss_dict['reg_loss'] /= len(data_loader)
    
    return loss_dict

def test(data_loader, model, criterion, reg_lambda, reg_tau, device):
    model.eval()
    loss_sum = 0
    reg_loss_sum = 0
    
    progress_bar = tqdm(data_loader, desc='Testing')
    with torch.no_grad():
        for x, y in progress_bar:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            if args.fast_nn:
                pred = model(x, is_training=False).squeeze()
            else:
                pred = model(x).squeeze()
            loss = criterion(pred, y.squeeze())
            
            loss_sum += loss.item()
            
            if reg_tau and args.fast_nn:
                reg_loss = model.regularization_loss(model=model, tau=reg_tau)
                reg_loss_sum += reg_lambda * reg_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_sum/(progress_bar.n+1):.4f}",
                'reg_loss': f"{reg_loss_sum/(progress_bar.n+1):.4f}" if reg_tau else "N/A"
            })

    loss_dict = {'l2_loss': loss_sum / len(data_loader)}
    if reg_tau and args.fast_nn: loss_dict['reg_loss'] = reg_lambda * model.regularization_loss(model=model, tau=reg_tau).item()
    
    return loss_dict

# train loop
try:
    anneal_rate = (config.HP_TAU * 10 - config.HP_TAU) / config.NUM_EPOCHS
    anneal_tau = config.HP_TAU * 10

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    logging.info(f"Using device: {device}")

    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        logging.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        logging.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        anneal_tau -= anneal_rate
        train_losses = train(train_loader, model, criterion, optimizer, reg_lambda=0.1, reg_tau=anneal_tau, device=device)
        scheduler.step()
        test_losses = test(test_loader, model, criterion, reg_lambda=0.1, reg_tau=anneal_tau, device=device)
        
        # Log losses
        logging.info(f"Train Loss: {train_losses['l2_loss']:.4f}")
        if anneal_tau and args.fast_nn:
            logging.info(f"Train Reg Loss: {train_losses['reg_loss']:.4f}")
        logging.info(f"Test Loss: {test_losses['l2_loss']:.4f}")
        if anneal_tau and args.fast_nn:
            logging.info(f"Test Reg Loss: {test_losses['reg_loss']:.4f}")
        
        # Save best model
        if test_losses['l2_loss'] < best_val_loss:
            best_val_loss = test_losses['l2_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': train_dataset.scaler,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'checkpoints/best_model.pth')
            logging.info("Saved new best model!")

except Exception as e:
    logging.error(f"Error during training: {e}")
    raise

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler
}, 'checkpoints/final_model.pth')

# Evaluate the model
logging.info("\nEvaluating model on test set...")
mae, rmse, r2 = evaluate(model, test_loader, train_dataset.scaler)
logging.info(f"Final Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")