import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from config.config import config

class ETTh1Dataset(Dataset):
    def __init__(self, data_path, seq_len=config.SEQ_LEN, pred_len=96, split='train', target='OT'):
        '''
        Initialize ETTh1 dataset for time series forecasting following standard benchmarks
        
        Args:
            data_path: Path to the CSV file
            seq_len: Length of input sequence (lookback window)
            pred_len: Length of prediction horizon (96, 192, 336, 720)
            split: 'train', 'val', or 'test'
            target: Target variable to predict ('OT' for univariate)
        '''
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.target = target

        # Load and preprocess data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Standard ETT splits: 12 months train, 4 months val, 4 months test
        # For hourly data (8760 hours/year), this is approximately:
        # Train: 12 months = 8760 hours
        # Val: 4 months = 2920 hours  
        # Test: 4 months = 2920 hours
        total_len = len(self.df)
        train_end = int(total_len * 0.6)    # ~12 months
        val_end = int(total_len * 0.8)      # ~16 months total
        
        # Apply split (no overlap)
        if split == 'train':
            self.df = self.df[:train_end]
        elif split == 'val':
            self.df = self.df[train_end:val_end]
        else:  # test
            self.df = self.df[val_end:]
        
        # Feature columns (all except date)
        self.feature_cols = [col for col in self.df.columns if col != 'date']
        
        # Normalize features using training data statistics only
        if split == 'train':
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.df[self.feature_cols].values)
        else:
            # For val/test, we need the training scaler - this will be set externally (in training script)
            self.scaler = None
            self.data = self.df[self.feature_cols].values
        
        # Target column index for univariate prediction
        self.target_col_idx = self.feature_cols.index(self.target)
        
    def set_scaler(self, scaler):
        '''Set scaler from training set for val/test sets'''
        self.scaler = scaler
        self.data = self.scaler.transform(self.df[self.feature_cols].values)
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)
    
    def __getitem__(self, idx):
        '''
        Get a sequence for training/testing
        Returns:
            x: input sequence of shape (seq_len, n_features) 
            y: target sequence of shape (pred_len,) for univariate prediction
        '''
        # Input sequence: seq_len timesteps of all features
        x = self.data[idx:idx + self.seq_len]
        
        # Target sequence: pred_len timesteps of target variable only
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col_idx]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)