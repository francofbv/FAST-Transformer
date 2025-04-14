import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
from config.config import config

class OptiverDataset(Dataset):
    def __init__(self, data_path, seq_len=config.SEQ_LEN, is_training=True):
        """
        Initialize the Optiver dataset
        
        Args:
            data_path: Path to the CSV file
            seq_len: Length of the sequence for the transformer
            is_training: Whether this is training data (affects target calculation)
        """
        self.seq_len = seq_len
        self.is_training = is_training
        
        # Load and preprocess data
        self.df = pd.read_csv(data_path)
        self.df = self.df.sort_values(['time_id', 'stock_id'])
        
        # Print debug information
        print(f"Number of rows in dataset: {len(self.df)}")
        print(f"Number of unique time_ids: {len(self.df['time_id'].unique())}")
        print(f"Number of unique stocks: {len(self.df['stock_id'].unique())}")
        
        # Feature engineering
        self.df = self._engineer_features(self.df)
        
        # Normalize features
        self.scaler = StandardScaler()
        feature_cols = [col for col in self.df.columns if col not in ['time_id', 'stock_id', 'target', 'row_id']]
        self.features = self.scaler.fit_transform(self.df[feature_cols])
        
        # Store targets if training
        if is_training:
            self.targets = self.df['target'].values
            if config.NORMALIZE_TARGETS:
                self.target_scaler = StandardScaler()
                self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1)).flatten()
                config.TARGET_SCALER = self.target_scaler
        
        # Create sequences
        self.X, self.y = self._create_sequences()
        
        # Print debug information
        print(f"Number of sequences created: {len(self.X)}")
        if is_training:
            print(f"Number of targets: {len(self.y)}")
        
    def _engineer_features(self, df):
        """
        Engineer additional features from the raw data
        """
        # Calculate price changes within each time_id
        df['price_change'] = df.groupby('time_id')['reference_price'].pct_change()
        
        # Calculate rolling statistics within each time_id
        for window in [5, 10, 20, 50]:
            df[f'price_ma_{window}'] = df.groupby('time_id')['reference_price'].rolling(window).mean().reset_index(0, drop=True)
            df[f'price_std_{window}'] = df.groupby('time_id')['reference_price'].rolling(window).std().reset_index(0, drop=True)
            df[f'volume_ma_{window}'] = df.groupby('time_id')['matched_size'].rolling(window).mean().reset_index(0, drop=True)
        
        # Calculate volatility within each time_id
        df['volatility'] = df.groupby('time_id')['price_change'].rolling(20).std().reset_index(0, drop=True)
        
        # Calculate spread
        df['spread'] = (df['ask_price'] - df['bid_price']) / df['reference_price']
        
        # Fill NaN values
        df = df.fillna(-9e10)
        
        return df
    
    def _create_sequences(self):
        """
        Create sequences for the transformer model using overlapping windows
        """
        X, y = [], []
        
        # Get unique time_ids
        unique_times = sorted(self.df['time_id'].unique())
        num_features = self.features.shape[1]
        
        print(f"Creating sequences with {len(unique_times)} unique time points")
        print(f"Number of features: {num_features}")
        print(f"Sequence length: {self.seq_len}")
        
        # Create sequences with overlapping windows
        stride = 1  # Overlap between sequences
        for i in range(0, len(unique_times) - self.seq_len, stride):
            # Get the time window
            time_window = unique_times[i:i+self.seq_len]
            
            # Get all data points in this time window
            window_mask = self.df['time_id'].isin(time_window)
            window_data = self.features[window_mask]
            
            # Get unique stocks in this window
            unique_stocks = self.df[window_mask]['stock_id'].unique()
            num_stocks = len(unique_stocks)
            
            # Create a 3D array with padding for missing stocks
            window_sequence = np.zeros((self.seq_len, num_stocks, num_features))
            
            # Fill the sequence with actual data
            for t, time_id in enumerate(time_window):
                time_mask = self.df['time_id'] == time_id
                time_data = self.features[time_mask]
                window_sequence[t, :len(time_data)] = time_data
            
            X.append(window_sequence)
            
            if self.is_training:
                # Get the target for the next time step
                next_time = unique_times[i+self.seq_len]
                next_time_mask = self.df['time_id'] == next_time
                y.append(self.targets[next_time_mask])
            
            if i % 100 == 0:  # Print progress every 100 windows
                print(f"Processed window {i}/{len(unique_times) - self.seq_len}")
        
        if not X:
            raise ValueError("No valid sequences could be created. Check if the sequence length is appropriate for your data size.")
            
        print(f"\nFinal shapes:")
        print(f"X shape: {np.array(X).shape}")
        if self.is_training:
            print(f"y shape: {np.array(y).shape}")
            
        return np.array(X), np.array(y) if self.is_training else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        if self.is_training:
            y = torch.FloatTensor(self.y[idx])
            return x, y
        return x 