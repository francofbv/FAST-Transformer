import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from scipy.sparse.linalg import eigsh

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.config import config
from models.fastnn_transformer import FastNNTransformer
from utils.dataloader import ETTh1Dataset
from scripts.evaluate import evaluate_etth1

def compute_dp_mat(data, r_bar=config.R_BAR):
    '''
    Compute the diversified projection matrix for FAST-NN
    
    data: input data to compute covariance from
    r_bar: number of eigenvalues to keep
    '''
    p = data.shape[1]
    covariance_matrix = data.T @ data
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)
    return dp_matrix

def create_data_loaders(data_path, pred_len, multivariate=False):
    '''Create train, validation, and test data loaders for ETTh1'''
    
    # Create datasets
    train_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='train', multivariate=multivariate)
    val_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='val', multivariate=multivariate)
    test_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='test', multivariate=multivariate)
    
    # Share training scaler with val/test datasets
    val_dataset.set_scaler(train_dataset.scaler)
    test_dataset.set_scaler(train_dataset.scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset

def train_model(model, train_loader, val_loader, pred_len, device, multivariate=False):
    '''Train the FAST-Transformer model'''
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    forecast_type = "Multivariate" if multivariate else "Univariate"
    print(f"Training FAST-Transformer for {forecast_type} forecasting, prediction length {pred_len}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(X, is_training=True)
            loss = criterion(output, y)
            
            # Add FAST-NN regularization
            reg_loss = model.regularization_loss(model, config.HP_TAU)
            total_loss = loss + config.LAMBDA * reg_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def main(multivariate=False):
    '''Main training function for ETTh1 benchmarking'''
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = config.DATA_PATH
    forecast_type = "multivariate" if multivariate else "univariate"
    output_dim = config.INPUT_DIM if multivariate else 1
    
    print(f"\nTraining FAST-Transformer for {forecast_type} forecasting")
    print(f"Output dimensions: {output_dim}")
    
    # Create results directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    results = {}
    
    # Train and evaluate for each prediction horizon
    for pred_len in config.PRED_LENS:
        print(f"\n{'='*50}")
        print(f"Training for prediction horizon: {pred_len}")
        print(f"{'='*50}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(data_path, pred_len, multivariate)
        
        # Compute diversified projection matrix from training data
        train_features = []
        for X, _ in train_loader:
            batch_size, seq_len, num_features = X.shape
            train_features.append(X.reshape(-1, num_features).numpy())
        train_features = np.vstack(train_features)
        
        dp_mat = compute_dp_mat(train_features)
        
        # Initialize model
        model = FastNNTransformer(
            dp_mat=dp_mat,
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            r_bar=config.R_BAR,
            width=config.WIDTH,
            pred_len=pred_len,
            output_dim=output_dim
        )
        
        # Train model
        model, best_val_loss = train_model(model, train_loader, val_loader, pred_len, device, multivariate)
        
        # Evaluate on test set
        test_metrics = evaluate_etth1(model, test_loader, device, multivariate)
        
        # Store results
        results[pred_len] = {
            'val_loss': best_val_loss,
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae']
        }
        
        # Save model
        model_path = f'checkpoints/fast_transformer_etth1_{forecast_type}_{pred_len}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        print(f"Results for horizon {pred_len}:")
        print(f"  Test MSE: {test_metrics['mse']:.6f}")
        print(f"  Test MAE: {test_metrics['mae']:.6f}")
    
    # Print final results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'MSE':<12} {'MAE':<12}")
    print("-" * 34)
    
    for pred_len in config.PRED_LENS:
        mse = results[pred_len]['test_mse']
        mae = results[pred_len]['test_mae']
        print(f"{pred_len:<10} {mse:<12.6f} {mae:<12.6f}")
    
    # Save results to file
    import json
    results_file = f'results/etth1_{forecast_type}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train FAST-Transformer on ETTh1')
    parser.add_argument('--multivariate', action='store_true', 
                       help='Train for multivariate forecasting (default: univariate)')
    args = parser.parse_args()
    
    main(multivariate=args.multivariate)