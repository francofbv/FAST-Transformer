import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

class OptiverDataset(Dataset):
    def __init__(self, data_path, seq_len=120, is_training=True):
        """
        Initialize the Optiver dataset
        
        Inherits from torch.utils.data.Dataset
        
        Args:
            data_path: Path to the CSV file
            seq_len: Length of the sequence for the transformer
            is_training: Whether this is training data (affects target calculation)
        """
        self.seq_len = seq_len # seq_len = number of lags
        self.is_training = is_training
        
        df = pd.read_csv(data_path) # load data
        df['time_id'] = pd.to_datetime(df['time_id']) # time to datetime
        
        df = df.sort_values('time_id') # sort by time_id (dataset already sorted but this is a sanity check)
        
        df = self._engineer_features(df) # feature engineering
        
        self.scaler = StandardScaler() # normalize features
        feature_cols = [col for col in df.columns if col not in ['time_id', 'stock_id', 'target']]
        self.features = self.scaler.fit_transform(df[feature_cols])
        
        if is_training: self.targets = df['target'].values # store targets if training
        
        self.X, self.y = self._create_sequences() # create sequences (lags)
        
    def _engineer_features(self, df):
        """
        Engineer additional features from the raw data
        """

        df['price_change'] = df.groupby('stock_id')['reference_price'].pct_change() # compute price changes
        
        for window in [5, 10, 20, 50]: # rolling windows
            df[f'price_ma_{window}'] = df.groupby('stock_id')['reference_price'].rolling(window).mean().reset_index(0, drop=True)
            df[f'price_std_{window}'] = df.groupby('stock_id')['reference_price'].rolling(window).std().reset_index(0, drop=True)
            df[f'volume_ma_{window}'] = df.groupby('stock_id')['matched_size'].rolling(window).mean().reset_index(0, drop=True)
        
        df['volatility'] = df.groupby('stock_id')['price_change'].rolling(20).std().reset_index(0, drop=True) # compute volatility
        
        df['spread'] = (df['ask_price'] - df['bid_price']) / df['reference_price'] # compute spread (difference between bid and ask price of asset, normalized by reference price)
        
        df = df.fillna(-9e10) # fill NaN values with -9e10 (from 9th place solution)
        
        return df
    
    def _create_sequences(self):
        """
        Create sequences for the transformer model
        """
        X, y = [], []
        
        # Group by stock_id
        for stock_id in pd.unique(self.df['stock_id']):
            stock_data = self.df[self.df['stock_id'] == stock_id]
            features = self.features[stock_data.index]
            
            # Create sequences
            for i in range(len(features) - self.seq_len):
                X.append(features[i:i+self.seq_len])
                if self.is_training:
                    y.append(self.targets[i+self.seq_len])
        
        return np.array(X), np.array(y) if self.is_training else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        if self.is_training:
            y = torch.FloatTensor([self.y[idx]])
            return x, y
        return x 