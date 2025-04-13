import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import config
from models.transformer import TimeSeriesTransformer
from models.fastnn_transformer import FastNNTransformer
from utils.optiver_dataloader import OptiverDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'optiver_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model for Optiver Trading at the Close')
    parser.add_argument('--fast_nn', action='store_true', help='Use Fast-NN Transformer model')
    parser.add_argument('--data_path', type=str, default=config.TRAIN_DATA_PATH, help='Path to training data')
    args = parser.parse_args()

    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        dataset = OptiverDataset(args.data_path, seq_len=config.SEQ_LEN)
        
        # Split into train and validation
        train_size = int(len(dataset) * (1 - config.VALIDATION_SPLIT))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        if args.fast_nn:
            # Compute DP matrix for Fast-NN
            dp_mat = compute_dp_mat(train_dataset.dataset.features)
            model = FastNNTransformer(
                dp_mat=dp_mat,
                input_dim=config.INPUT_DIM,
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                num_layers=config.NUM_LAYERS,
                r_bar=config.R_BAR,
                width=config.WIDTH
            )
            logging.info("Using Fast-NN Transformer model")
        else:
            model = TimeSeriesTransformer(
                input_dim=config.INPUT_DIM,
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                num_layers=config.NUM_LAYERS
            )
            logging.info("Using standard Transformer model")
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info(f"Using device: {device}")
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.NUM_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch, (X, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')):
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(X)
                loss = nn.MSELoss()(output, y)
                
                if args.fast_nn:
                    reg_loss = model.regularization_loss(model, config.HP_TAU)
                    loss += config.CHOICE_LAMBDA[0] * reg_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    output = model(X)
                    loss = nn.MSELoss()(output, y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Log metrics
            logging.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'scaler': dataset.scaler
                }, 'checkpoints/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logging.info("Early stopping triggered")
                    break
        
        logging.info("Training completed!")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 