import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from config.config import config

class ETTh1Dataset(Dataset):
    def __init__(self, data_path, seq_len=config.SEQ_LEN, pred_len=96, split='train', target='OT', multivariate=False):
        '''
        Initialize ETTh1 dataset for time series forecasting following standard benchmarks
        
        Args:
            data_path: Path to the CSV file
            seq_len: Length of input sequence (lookback window)
            pred_len: Length of prediction horizon (96, 192, 336, 720)
            split: 'train', 'val', or 'test'
            target: Target variable to predict ('OT' for univariate, ignored for multivariate)
            multivariate: If True, predict all variables; if False, predict only target
        '''
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.target = target
        self.multivariate = multivariate

        # Load and preprocess data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Standard ETTh1 academic splits using fixed sample counts
        # Train: 8,640 samples (12 months * 30 days * 24 hours)
        # Val: 2,880 samples (4 months * 30 days * 24 hours)  
        # Test: 2,880 samples (4 months * 30 days * 24 hours)
        # Total: 14,400 samples
        
        total_samples = len(self.df)
        train_samples = 8640
        val_samples = 2880
        test_samples = 2880
        
        # Ensure we have enough data
        if total_samples < train_samples + val_samples + test_samples:
            print(f"Warning: Dataset has {total_samples} samples, need {train_samples + val_samples + test_samples}")
            # Use proportional splits if dataset is smaller
            train_end = int(0.6 * total_samples)
            val_end = int(0.8 * total_samples)
        else:
            train_end = train_samples
            val_end = train_samples + val_samples
        
        # Apply standard academic splits
        if split == 'train':
            self.df = self.df.iloc[:train_end]
        elif split == 'val':
            self.df = self.df.iloc[train_end:val_end]
        else:  # test
            self.df = self.df.iloc[val_end:val_end + test_samples]
        
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
        
        # Target column indices for prediction
        if self.multivariate:
            # For multivariate, predict all features
            self.target_col_indices = list(range(len(self.feature_cols)))
        else:
            # For univariate, predict only the target variable
            self.target_col_indices = [self.feature_cols.index(self.target)]
        
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
            y: target sequence of shape (pred_len, n_targets) for multivariate or (pred_len,) for univariate
        '''
        # Input sequence: seq_len timesteps of all features
        x = self.data[idx:idx + self.seq_len]
        
        # Target sequence: pred_len timesteps of target variable(s)
        if self.multivariate:
            # Multivariate: predict all variables
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col_indices]
        else:
            # Univariate: predict only target variable
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_col_indices[0]]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)