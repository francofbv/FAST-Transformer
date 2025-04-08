import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import torch.optim as optim
import numpy as np
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

# Initialize model based on command line argument
if args.fast_nn: 
    dp_mat = np.random.randn(1, config.R_BAR)  # Generate random matrix (p x r_bar)
    model = FastNNTransformer(
        input_dim=1,
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

        if reg_tau:
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
            
            pred = model(x, is_training=False).squeeze()
            loss = criterion(pred, y.squeeze())
            loss_sum += loss.item()
            
            if reg_tau:
                reg_loss = model.regularization_loss(model=model, tau=reg_tau)
                reg_loss_sum += reg_lambda * reg_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_sum/(progress_bar.n+1):.4f}",
                'reg_loss': f"{reg_loss_sum/(progress_bar.n+1):.4f}" if reg_tau else "N/A"
            })

    loss_dict = {'l2_loss': loss_sum / len(data_loader)}
    if reg_tau: loss_dict['reg_loss'] = reg_lambda * model.regularization_loss(model=model, tau=reg_tau).item()
    
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
        if anneal_tau:
            logging.info(f"Train Reg Loss: {train_losses['reg_loss']:.4f}")
        logging.info(f"Test Loss: {test_losses['l2_loss']:.4f}")
        if anneal_tau:
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