import torch 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, test_loader, scaler):
    '''
    Evaluate the model on the test set

    model: model to evaluate
    test_loader: test data loader
    scaler: scaler to inverse transform the predictions and targets
    '''

    model.eval()
    device = next(model.parameters()).device  
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            preds = model(X_batch).squeeze() # squeeze to remove singleton dimension
            y_true = y_batch.squeeze()
            
            all_preds.extend(preds.cpu().numpy()) # move to cpu and convert to numpy
            all_targets.extend(y_true.cpu().numpy())
            
    all_preds = np.array(all_preds).reshape(-1, 1) # reshape to 2D array
    all_targets = np.array(all_targets).reshape(-1, 1)

    dummy_vol = np.zeros_like(all_preds) # dummy vector
    
    preds_with_dummy = np.hstack([all_preds, dummy_vol]) # reconstruct to original shape to make compatible with scaler
    targets_with_dummy = np.hstack([all_targets, dummy_vol])

    inv_preds = scaler.inverse_transform(preds_with_dummy)[:, 0].reshape(-1, 1) # convert transformed data to original scale
    inv_targets = scaler.inverse_transform(targets_with_dummy)[:, 0].reshape(-1, 1)

    mae = mean_absolute_error(inv_targets, inv_preds)
    rmse = mean_squared_error(inv_targets, inv_preds, squared=False)
    r2 = r2_score(inv_targets, inv_preds)
    
    # compute metrics
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


    return mae, rmse, r2