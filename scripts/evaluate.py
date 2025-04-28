import torch 
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from config.config import config

'''
primary evaluation method, scales the predictions and ground truth back to the original scale for testing
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
    rmse = root_mean_squared_error(ground_truth, preds)
    
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