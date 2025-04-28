import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from config.config import config

'''
Custom dataloader for the Optiver dataset, most of this code is taken from the pytorch documentation

https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''

class OptiverDataset(Dataset): # inherit from torch.utils.data.Dataset
    def __init__(self, data_path, seq_len=config.SEQ_LEN, is_training=True):
        """
        Initialize Optiver dataset
        
        Args:
            data_path: Path to the CSV file
            seq_len: Length of the sequence for the transformer
            is_training: Whether this is training data (affects target calculation)
        """
        self.seq_len = seq_len
        self.is_training = is_training
        
        self.df = pd.read_csv(data_path)
        self.df = self.df.sort_values(['time_id', 'stock_id'])
        
        self.df = self._engineer_features(self.df)
        
        self.scaler = StandardScaler()
        feature_columns = []
        for column in self.df.columns:
            if column not in ['time_id', 'stock_id', 'target', 'row_id']:
                feature_columns.append(column)
        self.features = self.scaler.fit_transform(self.df[feature_columns])
        
        if is_training:
            self.targets = self.df['target'].values
            self.target_scaler = StandardScaler()
            self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
            config.TARGET_SCALER = self.target_scaler
        
        self.X, self.y = self.create_sequences()
        
        
    def _engineer_features(self, df):
        """
        Engineer additional features from the raw data
        """
        df['price_change'] = df.groupby('time_id')['reference_price'].pct_change()
        
        for window in config.LAGS:
            df[f'price_moving_avg{window}'] = df.groupby('time_id')['reference_price'].rolling(window).mean().reset_index(0, drop=True)
            df[f'price_moving_std{window}'] = df.groupby('time_id')['reference_price'].rolling(window).std().reset_index(0, drop=True)
            df[f'volume_moving_avg{window}'] = df.groupby('time_id')['matched_size'].rolling(window).mean().reset_index(0, drop=True)
        df['volatility'] = df.groupby('time_id')['price_change'].rolling(20).std().reset_index(0, drop=True)
        df['spread'] = (df['ask_price'] - df['bid_price']) / df['reference_price']
        
        df = df.fillna(-9e10) # replace NaN values with -9e10 (from discussion post from 9th place team)
        
        return df
    
    # took this from other repo
    def create_sequences(self):
        """
        Create sequences for the transformer model using overlapping windows
        """
        X, y = [], []
        
        unique_times = sorted(self.df['time_id'].unique())
        num_features = self.features.shape[1]
        
        print(f"Creating sequences with {len(unique_times)} unique time points")
        print(f"Number of features: {num_features}")
        print(f"Sequence length: {self.seq_len}")
        
        stride = 1 
        for i in range(0, len(unique_times) - self.seq_len, stride):
            time_window = unique_times[i:i+self.seq_len]
            
            window_mask = self.df['time_id'].isin(time_window)
            window_data = self.features[window_mask]
            
            unique_stocks = self.df[window_mask]['stock_id'].unique()
            num_stocks = len(unique_stocks)
            
            window_sequence = np.zeros((self.seq_len, num_stocks, num_features))
            
            for t, time_id in enumerate(time_window):
                time_mask = self.df['time_id'] == time_id
                time_data = self.features[time_mask]
                window_sequence[t, :len(time_data)] = time_data
            
            X.append(window_sequence)
            
            if self.is_training:
                next_time = unique_times[i+self.seq_len]
                next_time_mask = self.df['time_id'] == next_time
                y.append(self.targets[next_time_mask])
        
        return np.array(X), np.array(y) if self.is_training else None
    
    # took both of these from pytorch documentation
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        if self.is_training:
            y = torch.FloatTensor(self.y[idx])
            return x, y
        return x 