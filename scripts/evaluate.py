import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, test_loader, scaler):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).squeeze().cpu().numpy()
            y_true = y_batch.squeeze().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(y_true)
            
    # Convert to numpy arrays and reshape
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)

    # Create dummy volatility column (zeros) to match original data shape
    dummy_vol = np.zeros_like(all_preds)
    
    # Add dummy volatility column to match scaler's expected shape
    preds_with_dummy = np.hstack([all_preds, dummy_vol])
    targets_with_dummy = np.hstack([all_targets, dummy_vol])

    # Inverse transform predictions and targets
    inv_preds = scaler.inverse_transform(preds_with_dummy)[:, 0].reshape(-1, 1)  # Take only first column (price)
    inv_targets = scaler.inverse_transform(targets_with_dummy)[:, 0].reshape(-1, 1)  # Take only first column (price)

    # Evaluation metrics
    mae = mean_absolute_error(inv_targets, inv_preds)
    rmse = mean_squared_error(inv_targets, inv_preds, squared=False)
    r2 = r2_score(inv_targets, inv_preds)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(inv_targets, label='True')
    plt.plot(inv_preds, label='Predicted')
    plt.title("NVIDIA Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return mae, rmse, r2