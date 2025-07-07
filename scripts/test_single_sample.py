import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.config import config
from models.fastnn_transformer import FastNNTransformer
from utils.dataloader import ETTh1Dataset
from scipy.sparse.linalg import eigsh

def compute_dp_mat(data, r_bar=config.R_BAR):
    '''Compute the diversified projection matrix for FAST-NN'''
    p = data.shape[1]
    covariance_matrix = data.T @ data
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)
    return dp_matrix

def test_single_sample(pred_len=96, sample_idx=0):
    '''Test model on a single sample and print prediction vs ground truth'''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    data_path = config.DATA_PATH
    train_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='train')
    test_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='test')
    
    # Set scaler for test dataset
    test_dataset.set_scaler(train_dataset.scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
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
        pred_len=pred_len
    )
    
    # Skip loading pre-trained model - use random initialization
    model = model.to(device)
    print("Using randomly initialized model (no pre-trained weights)")
    model.eval()
    
    # Get single sample
    if sample_idx >= len(test_dataset):
        print(f"Sample index {sample_idx} out of range. Test dataset has {len(test_dataset)} samples.")
        return
    
    X, y_true = test_dataset[sample_idx]
    X = X.unsqueeze(0).to(device)  # Add batch dimension
    y_true = y_true.numpy()
    
    # Make prediction
    with torch.no_grad():
        y_pred = model(X).cpu().numpy().squeeze()
    
    # Print results
    print(f"\n{'='*80}")
    print(f"SINGLE SAMPLE PREDICTION TEST")
    print(f"{'='*80}")
    print(f"Prediction horizon: {pred_len} timesteps")
    print(f"Sample index: {sample_idx}")
    print(f"Target variable: OT (Oil Temperature)")
    print(f"\nInput sequence shape: {X.shape}")
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Ground truth shape: {y_true.shape}")
    
    # Print first 10 and last 10 predictions vs ground truth
    print(f"\n{'Timestep':<10} {'Prediction':<12} {'Ground Truth':<12} {'Error':<12}")
    print("-" * 50)
    
    # First 10 timesteps
    for i in range(min(10, len(y_pred))):
        error = abs(y_pred[i] - y_true[i])
        print(f"{i+1:<10} {y_pred[i]:<12.6f} {y_true[i]:<12.6f} {error:<12.6f}")
    
    # Add separator if we have more than 10 timesteps
    if len(y_pred) > 20:
        print("...")
    
    # Last 10 timesteps (if more than 10 total)
    if len(y_pred) > 10:
        start_idx = max(10, len(y_pred) - 10)
        for i in range(start_idx, len(y_pred)):
            error = abs(y_pred[i] - y_true[i])
            print(f"{i+1:<10} {y_pred[i]:<12.6f} {y_true[i]:<12.6f} {error:<12.6f}")
    
    # Calculate metrics
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    
    print(f"\n{'='*50}")
    print(f"METRICS FOR THIS SAMPLE")
    print(f"{'='*50}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    
    # Print some statistics about the predictions
    print(f"\n{'='*50}")
    print(f"STATISTICS")
    print(f"{'='*50}")
    print(f"Ground Truth - Mean: {np.mean(y_true):.6f}, Std: {np.std(y_true):.6f}")
    print(f"Prediction   - Mean: {np.mean(y_pred):.6f}, Std: {np.std(y_pred):.6f}")
    print(f"Error        - Mean: {np.mean(np.abs(y_pred - y_true)):.6f}, Std: {np.std(np.abs(y_pred - y_true)):.6f}")

if __name__ == "__main__":
    # Test with different prediction horizons and sample indices
    pred_len = 96  # Change this to test different horizons: 96, 192, 336, 720
    sample_idx = 0  # Change this to test different samples
    
    print("Testing single sample prediction...")
    test_single_sample(pred_len=pred_len, sample_idx=sample_idx)