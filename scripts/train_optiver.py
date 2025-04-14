import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.config import config
from models.transformer import TimeSeriesTransformer
from models.fastnn_transformer import FastNNTransformer
from utils.optiver_dataloader import OptiverDataset
from scripts.evaluate import evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'optiver_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Add safe globals for model loading
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, feature_scaler, target_scaler, path):
    """Save model checkpoint with only necessary data"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'feature_scaler': {
            'mean': feature_scaler.mean_,
            'scale': feature_scaler.scale_
        },
        'target_scaler': {
            'mean': target_scaler.mean_,
            'scale': target_scaler.scale_
        } if target_scaler else None
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint with proper error handling"""
    try:
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise

def compute_dp_mat(x, r_bar=config.R_BAR):
    '''
    Compute the pretrained dp (diversified projection)matrix for the fast-nn transformer

    x: data to compute dp matrix for
    r_bar: number of eigenvalues to keep
    '''
    p = np.shape(x)[1]
    covariance_matrix = x.T @ x
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)

    return dp_matrix

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
                    reg_loss = reg_loss / (config.BATCH_SIZE * config.SEQ_LEN)
                    loss += config.CHOICE_LAMBDA[0] * reg_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_metrics = evaluate(
                model, 
                val_loader, 
                feature_scaler=dataset.scaler,
                target_scaler=config.TARGET_SCALER,
                fast_nn=args.fast_nn
            )
            val_loss = val_metrics['loss']
            
            # Log metrics
            logging.info(f"Epoch {epoch+1}:")
            logging.info(f"  Train Loss = {train_loss:.4f}")
            logging.info(f"  Val Loss = {val_loss:.4f}")
            logging.info(f"  Val MAE = {val_metrics['mae']:.4f}")
            logging.info(f"  Val RMSE = {val_metrics['rmse']:.4f}")
            logging.info(f"  Val R² = {val_metrics['r2']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    feature_scaler=dataset.scaler,
                    target_scaler=config.TARGET_SCALER,
                    path='checkpoints/best_model.pth'
                )
                logging.info("  Saved new best model!")
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        logging.info("Training completed!")
        
        # Load and evaluate best model
        checkpoint = load_checkpoint('checkpoints/best_model.pth', model)
        final_metrics = evaluate(
            model, 
            val_loader, 
            feature_scaler=dataset.scaler,
            target_scaler=config.TARGET_SCALER,
            fast_nn=args.fast_nn
        )
        
        logging.info("\nFinal Evaluation of Best Model:")
        logging.info(f"  Val Loss = {final_metrics['loss']:.4f}")
        logging.info(f"  Val MAE = {final_metrics['mae']:.4f}")
        logging.info(f"  Val RMSE = {final_metrics['rmse']:.4f}")
        logging.info(f"  Val R² = {final_metrics['r2']:.4f}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 