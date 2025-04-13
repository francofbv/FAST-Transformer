import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
import yfinance as yf
from tqdm import tqdm

from config.config import config
from models.transformer import TimeSeriesTransformer
from models.fastnn_transformer import FastNNTransformer
from utils.dataloader import TimeSeriesDataset
from evaluate import evaluate

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
    '''
    Compute the pretrained dp (diversified projection)matrix for the fast-nn transformer

    x: data to compute dp matrix for
    r_bar: number of eigenvalues to keep
    '''
    print(x.shape)
    p = np.shape(x)[1]
    covariance_matrix = x.T @ x
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)

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
    # this lowk doesnt work so carefull lol
    model = TimeSeriesTransformer()
    logging.info("Using standard Transformer model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logging.info(f"Using device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) # from fast-nn paper



def train(data_loader, model, criterion, optimizer, reg_lambda, reg_tau, device):
    '''
    Train the model

    data_loader: data loader
    model: model to train
    criterion: loss function
    optimizer: optimizer
    reg_lambda: regularization parameter
    '''
    model.train()
    loss_dict = {'l2_loss': 0.0} # initialize l2_loss (MSE) to 0

    if reg_tau: loss_dict['reg_loss'] = 0.0 # if we're using regularization, initialize reg_loss to 0

    
    progress_bar = tqdm(data_loader, desc='Training')
    for batch, (X_batch, y_batch) in enumerate(progress_bar):
        # Move data to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        pred = model(X_batch).squeeze() # remove singleton dimension
        
        loss = criterion(pred, y_batch.squeeze()) # compute loss
        loss_dict['l2_loss'] += loss.item() # add MSE to dict

        if reg_tau and args.fast_nn:
            reg_loss = model.regularization_loss(model=model, tau=reg_tau)
            loss_dict['reg_loss'] += reg_lambda * reg_loss.item() # add reg loss to dict
            loss += reg_loss * reg_lambda # add reg loss to total loss for backprop

        optimizer.zero_grad() # zero gradients
        loss.backward() # backprop
        optimizer.step() # update weights
        
        # Update progress bar
        progress_bar.set_postfix({
            'l2_loss': f"{loss_dict['l2_loss']/(batch+1):.4f}",
            'reg_loss': f"{loss_dict['reg_loss']/(batch+1):.4f}" if reg_tau else "N/A"
        })
    
    loss_dict['l2_loss'] /= len(data_loader) # normalize loss
    if reg_tau: loss_dict['reg_loss'] /= len(data_loader) # normalize reg loss
    
    return loss_dict

def test(data_loader, model, criterion, reg_lambda, reg_tau, device):
    '''
    Test the model

    data_loader: data loader
    model: model to test
    criterion: loss function
    reg_lambda: regularization parameter
    '''
    model.eval()
    loss_sum = 0
    reg_loss_sum = 0
    
    progress_bar = tqdm(data_loader, desc='Testing')
    with torch.no_grad():
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            if args.fast_nn:
                pred = model(x, is_training=False).squeeze()
            else:
                pred = model(x).squeeze()
            loss = criterion(pred, y.squeeze())
            
            loss_sum += loss.item()
            
            if reg_tau and args.fast_nn:
                reg_loss = model.regularization_loss(model=model, tau=reg_tau) # compute reg loss
                reg_loss_sum += reg_lambda * reg_loss.item() # add reg loss to total loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_sum/(progress_bar.n+1):.4f}",
                'reg_loss': f"{reg_loss_sum/(progress_bar.n+1):.4f}" if reg_tau else "N/A"
            })

    loss_dict = {'l2_loss': loss_sum / len(data_loader)} # normalize loss
    if reg_tau and args.fast_nn: loss_dict['reg_loss'] = reg_lambda * model.regularization_loss(model=model, tau=reg_tau).item() # compute reg loss
    
    return loss_dict

# train loop
try:
    anneal_rate = (config.HP_TAU * 10 - config.HP_TAU) / config.NUM_EPOCHS # anneal rate for tau
    anneal_tau = config.HP_TAU * 10 # annealed tau value (tau scales regularization loss, reg loss gets smaller as tau increases)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    logging.info(f"Using device: {device}")

    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        logging.info(f"\nEpoch: {epoch+1}/{config.NUM_EPOCHS}")
        logging.info(f"Current lr: {scheduler.get_last_lr()[0]:.6f}")
        
        anneal_tau -= anneal_rate # update tau
        train_losses = train(train_loader, model, criterion, optimizer, reg_lambda=0.005, reg_tau=anneal_tau, device=device) # compute losses
        scheduler.step()
        test_losses = test(test_loader, model, criterion, reg_lambda=0.1, reg_tau=anneal_tau, device=device) # compute test losses
        
        # Log losses
        logging.info(f"Train Loss: {train_losses['l2_loss']:.4f}")
        if anneal_tau and args.fast_nn:
            logging.info(f"Train Reg Loss: {train_losses['reg_loss']:.4f}")
        logging.info(f"Test Loss: {test_losses['l2_loss']:.4f}")
        if anneal_tau and args.fast_nn:
            logging.info(f"Test Reg Loss: {test_losses['reg_loss']:.4f}")
        
        # check if model is better than previous best
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