import torch 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from config.config import config

'''
Primary evaluation methods for both Optiver and ETTh1 datasets
'''

def create_scaler_from_params(mean, scale):
    """
    Create a scaler from saved parameters

    mean: mean of the scaler
    scale: scale of the scaler
    """
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_samples_seen_ = 1
    return scaler

def evaluate(model, test_loader, feature_scaler=None, target_scaler=None, fast_nn=False):
    '''
    Evaluate the model on the test set

    model: model to evaluate
    test_loader: test data loader
    feature_scaler: scaler used for features (optional)
    target_scaler: scaler used for targets (optional)
    fast_nn: whether the model is a Fast-NN Transformer
    '''

    model.eval()
    device = next(model.parameters()).device  
    all_preds = []
    all_targets = []
    total_loss = 0.0

    # from original FAST-NN repo
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch)
            
            loss = torch.nn.MSELoss()(preds, y_batch)
            
            if fast_nn:
                reg_loss = model.regularization_loss(model, config.HP_TAU)
                reg_loss = reg_loss / (config.BATCH_SIZE * config.SEQ_LEN)
                loss += config.LAMBDA * reg_loss
            
            total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    all_preds = all_preds.reshape(-1, 1)
    all_targets = all_targets.reshape(-1, 1)

    if target_scaler is not None:
        if isinstance(target_scaler, dict):
            scaler = create_scaler_from_params(target_scaler['mean'], target_scaler['scale'])
        else:
            scaler = target_scaler
        preds = scaler.inverse_transform(all_preds)
        ground_truth = scaler.inverse_transform(all_targets)
    else:
        preds = all_preds
        ground_truth = all_targets

    mae = mean_absolute_error(ground_truth, preds)
    rmse = np.sqrt(mean_squared_error(ground_truth, preds))
    
    print("Eval Metrics:")
    avg_loss = total_loss / len(test_loader)
    print(f"Average Loss: {avg_loss}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
    }

def evaluate_etth1(model, test_loader, device, multivariate=False):
    '''
    Evaluate FAST-Transformer on ETTh1 dataset
    
    model: trained model
    test_loader: test data loader
    device: torch device
    multivariate: whether this is multivariate forecasting
    '''
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            preds = model(X_batch)
            
            # Compute loss
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # IMPORTANT: Inverse transform to original scale for accurate metrics
    # Get the scaler from the test dataset
    test_dataset = test_loader.dataset
    if hasattr(test_dataset, 'scaler') and test_dataset.scaler is not None:
        if multivariate:
            # For multivariate: all_preds and all_targets have shape (n_samples, pred_len, n_vars)
            original_shape = all_preds.shape
            # Reshape to (n_samples * pred_len, n_vars) for inverse transform
            preds_2d = all_preds.reshape(-1, all_preds.shape[-1])
            targets_2d = all_targets.reshape(-1, all_targets.shape[-1])
            
            # Inverse transform
            preds_original = test_dataset.scaler.inverse_transform(preds_2d)
            targets_original = test_dataset.scaler.inverse_transform(targets_2d)
            
            # Reshape back to original shape
            all_preds = preds_original.reshape(original_shape)
            all_targets = targets_original.reshape(original_shape)
        else:
            # For univariate: reshape to 2D for inverse transform
            preds_2d = all_preds.reshape(-1, 1)
            targets_2d = all_targets.reshape(-1, 1)
            
            # Inverse transform only the target variable (last column typically OT)
            target_scaler = test_dataset.scaler
            preds_original = target_scaler.inverse_transform(preds_2d)
            targets_original = target_scaler.inverse_transform(targets_2d)
            
            # Reshape back
            all_preds = preds_original.reshape(all_preds.shape)
            all_targets = targets_original.reshape(all_targets.shape)
    
    # Compute metrics on original scale
    if multivariate:
        # For multivariate, compute metrics across all variables
        mse = mean_squared_error(all_targets.reshape(-1), all_preds.reshape(-1))
        mae = mean_absolute_error(all_targets.reshape(-1), all_preds.reshape(-1))
        
        # Also compute per-variable metrics
        n_vars = all_preds.shape[-1]
        var_metrics = {}
        feature_cols = test_dataset.feature_cols
        for i in range(n_vars):
            var_mse = mean_squared_error(all_targets[:, :, i].reshape(-1), all_preds[:, :, i].reshape(-1))
            var_mae = mean_absolute_error(all_targets[:, :, i].reshape(-1), all_preds[:, :, i].reshape(-1))
            var_name = feature_cols[i] if i < len(feature_cols) else f'var_{i}'
            var_metrics[var_name] = {'mse': var_mse, 'mae': var_mae}
    else:
        # For univariate
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        var_metrics = None
    
    rmse = np.sqrt(mse)
    avg_loss = total_loss / len(test_loader)
    
    forecast_type = "Multivariate" if multivariate else "Univariate"
    print(f"{forecast_type} Test Evaluation Results (Original Scale):")
    print(f"Overall MSE: {mse:.6f}")
    print(f"Overall MAE: {mae:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"Average Loss: {avg_loss:.6f}")
    
    if multivariate and var_metrics:
        print("\nPer-variable metrics:")
        for var_name, metrics in var_metrics.items():
            print(f"  {var_name}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
    
    result = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'loss': avg_loss
    }
    
    if var_metrics:
        result['var_metrics'] = var_metrics
    
    return result

def benchmark_etth1_all_horizons(model_dict, test_loaders, device):
    '''
    Benchmark FAST-Transformer on all ETTh1 prediction horizons
    
    model_dict: dictionary of {pred_len: model} for each horizon
    test_loaders: dictionary of {pred_len: test_loader} for each horizon  
    device: torch device
    '''
    results = {}
    
    print("\nBenchmarking FAST-Transformer on ETTh1")
    print("=" * 50)
    
    for pred_len in config.PRED_LENS:
        print(f"\nEvaluating prediction horizon: {pred_len}")
        print("-" * 30)
        
        if pred_len not in model_dict or pred_len not in test_loaders:
            print(f"Model or test loader not found for horizon {pred_len}")
            continue
            
        model = model_dict[pred_len]
        test_loader = test_loaders[pred_len]
        
        # Evaluate model
        metrics = evaluate_etth1(model, test_loader, device)
        results[pred_len] = metrics
    
    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Horizon':<10} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 46)
    
    for pred_len in config.PRED_LENS:
        if pred_len in results:
            metrics = results[pred_len]
            print(f"{pred_len:<10} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f}")
    
    return results