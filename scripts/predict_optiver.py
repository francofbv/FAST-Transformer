import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from config.config import config
from models.transformer import TimeSeriesTransformer
from models.fastnn_transformer import FastNNTransformer
from utils.optiver_dataloader import OptiverDataset

def main():
    # Load the best model
    checkpoint = torch.load('checkpoints/best_model.pth')
    
    # Initialize model
    if 'dp_mat' in checkpoint:
        model = FastNNTransformer(
            dp_mat=checkpoint['dp_mat'],
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            r_bar=config.R_BAR,
            width=config.WIDTH
        )
    else:
        model = TimeSeriesTransformer(
            input_dim=config.INPUT_DIM,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS
        )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load test data
    test_dataset = OptiverDataset(config.TEST_DATA_PATH, seq_len=config.SEQ_LEN, is_training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for X in tqdm(test_loader, desc="Making predictions"):
            X = X.to(device)
            output = model(X)
            predictions.extend(output.cpu().numpy())
    
    # Create submission file
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'target': predictions
    })
    
    # Save submission
    os.makedirs(config.SUBMISSION_PATH, exist_ok=True)
    submission_path = os.path.join(config.SUBMISSION_PATH, f'submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main() 