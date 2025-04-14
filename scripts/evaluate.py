import torch 
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from config.config import config

def create_scaler_from_params(mean, scale):
    """Create a scaler from saved parameters"""
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

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            preds = model(X_batch)
            
            # Calculate loss
            loss = torch.nn.MSELoss()(preds, y_batch)
            
            if fast_nn:
                # Add regularization loss if using Fast-NN
                reg_loss = model.regularization_loss(model, config.HP_TAU)
                reg_loss = reg_loss / (config.BATCH_SIZE * config.SEQ_LEN)
                loss += config.CHOICE_LAMBDA[0] * reg_loss
            
            total_loss += loss.item()
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    # Convert to numpy arrays and reshape
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # If predictions are 2D (batch_size, num_stocks), flatten them
    if len(all_preds.shape) > 1:
        all_preds = all_preds.reshape(-1)
        all_targets = all_targets.reshape(-1)
    
    # Reshape to 2D arrays for inverse transform
    all_preds = all_preds.reshape(-1, 1)
    all_targets = all_targets.reshape(-1, 1)

    # Inverse transform if scalers are provided
    if target_scaler is not None:
        if isinstance(target_scaler, dict):
            # If scaler is provided as parameters
            scaler = create_scaler_from_params(target_scaler['mean'], target_scaler['scale'])
        else:
            # If scaler is provided as StandardScaler object
            scaler = target_scaler
        inv_preds = scaler.inverse_transform(all_preds)
        inv_targets = scaler.inverse_transform(all_targets)
    else:
        inv_preds = all_preds
        inv_targets = all_targets

    # Calculate metrics
    mae = mean_absolute_error(inv_targets, inv_preds)
    rmse = root_mean_squared_error(inv_targets, inv_preds)
    r2 = r2_score(inv_targets, inv_preds)
    
    # Print detailed metrics
    print("\nEvaluation Metrics:")
    print(f"Average Loss: {total_loss / len(test_loader):.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Target Range: [{inv_targets.min():.4f}, {inv_targets.max():.4f}]")
    print(f"Prediction Range: [{inv_preds.min():.4f}, {inv_preds.max():.4f}]")
    print(f"Target Mean: {inv_targets.mean():.4f}")
    print(f"Prediction Mean: {inv_preds.mean():.4f}")

    return {
        'loss': total_loss / len(test_loader),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'target_range': [inv_targets.min(), inv_targets.max()],
        'prediction_range': [inv_preds.min(), inv_preds.max()],
        'target_mean': inv_targets.mean(),
        'prediction_mean': inv_preds.mean()
    }