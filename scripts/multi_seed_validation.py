import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
import json
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.config import config
from models.fastnn_transformer import FastNNTransformer
from utils.dataloader import ETTh1Dataset
from scripts.evaluate import evaluate_etth1

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_dp_mat(data, r_bar=config.R_BAR):
    '''Compute the diversified projection matrix for FAST-NN'''
    p = data.shape[1]
    covariance_matrix = data.T @ data
    eigen_values, eigen_vectors = eigsh(covariance_matrix, r_bar, which='LM')
    dp_matrix = eigen_vectors / np.sqrt(p)
    return dp_matrix

def create_data_loaders(data_path, pred_len, seed):
    '''Create train, validation, and test data loaders for ETTh1'''
    
    # Create datasets
    train_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='train')
    val_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='val')
    test_dataset = ETTh1Dataset(data_path, seq_len=config.SEQ_LEN, pred_len=pred_len, split='test')
    
    # Share training scaler with val/test datasets
    val_dataset.set_scaler(train_dataset.scaler)
    test_dataset.set_scaler(train_dataset.scaler)
    
    # Create data loaders with fixed generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset

def train_single_run(seed, pred_len=96):
    '''Train model with a specific seed and return test metrics'''
    
    print(f"\n{'='*60}")
    print(f"Training Run with Seed: {seed}")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    data_path = config.DATA_PATH
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(data_path, pred_len, seed)
    
    # Compute diversified projection matrix from training data
    print("Computing diversified projection matrix...")
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
    
    model = model.to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training variables
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Training"):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X, is_training=True)
            
            # Calculate loss
            loss = criterion(y_pred, y)
            
            # Add regularization loss
            reg_loss = model.regularization_loss(model, config.HP_TAU)
            total_loss = loss + config.LAMBDA * reg_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if config.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X, is_training=False)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'checkpoints/best_model_seed_{seed}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f'checkpoints/best_model_seed_{seed}.pth'))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss = 0.0
    test_batches = 0
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X, is_training=False)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            test_batches += 1
            
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    avg_test_loss = test_loss / test_batches
    
    # Calculate additional metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (avoiding division by zero)
    mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
    
    results = {
        'seed': seed,
        'test_mse': float(mse),
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'test_mape': float(mape),
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(avg_train_loss),
        'epochs_trained': epoch + 1
    }
    
    print(f"Test Results for Seed {seed}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return results

def run_multi_seed_validation(seeds=[42, 123, 456, 789, 999], pred_len=96):
    '''Run training with multiple seeds and analyze results'''
    
    print(f"\n{'='*80}")
    print(f"MULTI-SEED VALIDATION FOR ETTH1 - HORIZON {pred_len}")
    print(f"{'='*80}")
    print(f"Running {len(seeds)} independent training runs...")
    print(f"Seeds: {seeds}")
    
    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    all_results = []
    
    for seed in seeds:
        try:
            result = train_single_run(seed, pred_len)
            all_results.append(result)
        except Exception as e:
            print(f"Error training with seed {seed}: {e}")
            continue
    
    # Analyze results
    if len(all_results) == 0:
        print("No successful training runs!")
        return
    
    print(f"\n{'='*80}")
    print(f"MULTI-SEED VALIDATION RESULTS")
    print(f"{'='*80}")
    
    # Extract metrics
    mse_values = [r['test_mse'] for r in all_results]
    mae_values = [r['test_mae'] for r in all_results]
    rmse_values = [r['test_rmse'] for r in all_results]
    mape_values = [r['test_mape'] for r in all_results]
    
    # Calculate statistics
    mse_stats = {
        'mean': np.mean(mse_values),
        'std': np.std(mse_values),
        'min': np.min(mse_values),
        'max': np.max(mse_values),
        'median': np.median(mse_values)
    }
    
    mae_stats = {
        'mean': np.mean(mae_values),
        'std': np.std(mae_values),
        'min': np.min(mae_values),
        'max': np.max(mae_values),
        'median': np.median(mae_values)
    }
    
    # Print detailed results
    print(f"\nIndividual Results:")
    print(f"{'Seed':<8} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Epochs':<8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['seed']:<8} {r['test_mse']:<10.6f} {r['test_mae']:<10.6f} {r['test_rmse']:<10.6f} {r['test_mape']:<10.2f} {r['epochs_trained']:<8}")
    
    print(f"\nStatistical Summary:")
    print(f"MSE  - Mean: {mse_stats['mean']:.6f} ± {mse_stats['std']:.6f}")
    print(f"     - Range: [{mse_stats['min']:.6f}, {mse_stats['max']:.6f}]")
    print(f"     - Median: {mse_stats['median']:.6f}")
    
    print(f"MAE  - Mean: {mae_stats['mean']:.6f} ± {mae_stats['std']:.6f}")
    print(f"     - Range: [{mae_stats['min']:.6f}, {mae_stats['max']:.6f}]")
    print(f"     - Median: {mae_stats['median']:.6f}")
    
    # Reproducibility analysis
    cv_mse = mse_stats['std'] / mse_stats['mean'] * 100
    cv_mae = mae_stats['std'] / mae_stats['mean'] * 100
    
    print(f"\nReproducibility Analysis:")
    print(f"MSE Coefficient of Variation: {cv_mse:.2f}%")
    print(f"MAE Coefficient of Variation: {cv_mae:.2f}%")
    
    if cv_mse < 5:
        print("✓ Excellent reproducibility (CV < 5%)")
    elif cv_mse < 10:
        print("✓ Good reproducibility (CV < 10%)")
    elif cv_mse < 20:
        print("⚠ Moderate reproducibility (CV < 20%)")
    else:
        print("✗ Poor reproducibility (CV > 20%)")
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'prediction_horizon': pred_len,
        'seeds': seeds,
        'individual_results': all_results,
        'statistics': {
            'mse': mse_stats,
            'mae': mae_stats,
            'reproducibility': {
                'mse_cv': cv_mse,
                'mae_cv': cv_mae
            }
        }
    }
    
    with open(f'multi_seed_results_horizon_{pred_len}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: multi_seed_results_horizon_{pred_len}.json")
    
    return results_summary

if __name__ == "__main__":
    # Run multi-seed validation
    seeds = [42, 123, 456, 789, 999]
    results = run_multi_seed_validation(seeds=seeds, pred_len=96)